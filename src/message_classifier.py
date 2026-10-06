"""Post-generation malicious/benign message classification.

The evaluator already knows which agent IDs are controlled attackers.  This
module uses that metadata only for labels; the classifier sees the generated
message text and a frozen sentence embedding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class MessageSample:
    text: str
    label: int
    group_id: str
    step: Any
    agent_id: int
    round_id: int
    attack_type: str


def _message_text(response: str) -> str:
    """Keep the generated reasoning and payload, removing format wrappers."""
    text = str(response or "")
    text = re.sub(r"(?is)<REASON>\s*:\s*", "", text)
    text = re.sub(r"(?im)^\s*<ANSWER>\s*:\s*.*$", "", text)
    return " ".join(text.split()).strip()


def extract_message_samples(row: Dict[str, Any]) -> List[MessageSample]:
    """Extract one labelled sample for every agent and communication round."""
    attack_type = str(row.get("attack_type") or "none")
    attacker_ids = {int(x) for x in (row.get("attacker_ids") or [])}
    group_id = str(row.get("question") or row.get("step") or "")
    samples: List[MessageSample] = []
    histories = row.get("agent_histories") or []
    for agent_id, history in enumerate(histories):
        for round_id, interaction in enumerate(history or []):
            if not isinstance(interaction, dict):
                continue
            text = _message_text(interaction.get("response", ""))
            if not text:
                continue
            samples.append(
                MessageSample(
                    text=text,
                    label=int(attack_type != "none" and agent_id in attacker_ids),
                    group_id=group_id,
                    step=row.get("step"),
                    agent_id=agent_id,
                    round_id=round_id + 1,
                    attack_type=attack_type,
                )
            )
    return samples


def summarize_classification(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]
) -> Dict[str, Any]:
    """Return stable binary classification metrics, including edge-case status."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = [int(x) for x in y_true]
    predicted = [int(x) for x in y_pred]
    scores = [float(x) for x in y_score]
    classes = sorted(set(labels))
    cm = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    if len(classes) < 2:
        return {
            "status": "insufficient_classes",
            "n": len(labels),
            "class_counts": {str(c): labels.count(c) for c in classes},
            "accuracy": float(accuracy_score(labels, predicted)) if labels else None,
            "balanced_accuracy": float(recall_score(labels, predicted, labels=classes, average="macro", zero_division=0)) if labels else None,
            "precision": float(precision_score(labels, predicted, zero_division=0)) if labels else None,
            "recall": float(recall_score(labels, predicted, zero_division=0)) if labels else None,
            "f1": float(f1_score(labels, predicted, zero_division=0)) if labels else None,
            "roc_auc": None,
            "confusion_matrix": cm,
        }
    return {
        "status": "ok",
        "n": len(labels),
        "class_counts": {"0": labels.count(0), "1": labels.count(1)},
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": cm,
    }


def _split_groups(samples: Sequence[MessageSample], test_size: float, seed: int) -> Tuple[List[int], List[int]]:
    from sklearn.model_selection import GroupShuffleSplit

    groups = [s.group_id for s in samples]
    y = [s.label for s in samples]
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("message classification needs at least two question groups")
    # Try several deterministic splits because a random group split can contain
    # only benign or only attacker samples in one side on a small run.
    for offset in range(50):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        train_idx, test_idx = next(splitter.split(samples, y, groups))
        if {y[i] for i in train_idx} == {0, 1} and {y[i] for i in test_idx} == {0, 1}:
            return train_idx.tolist(), test_idx.tolist()
    raise ValueError("could not create a group split containing both benign and malicious messages")


def _group_metrics(samples: Sequence[MessageSample], indices: Sequence[int], scores: Sequence[float], preds: Sequence[int]) -> Dict[str, Any]:
    y_true = [samples[i].label for i in indices]
    y_score = [scores[i] for i in indices]
    y_pred = [preds[i] for i in indices]
    return summarize_classification(y_true, y_pred, y_score)


def classify_messages(
    samples: Sequence[MessageSample],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fit a frozen MiniLM + linear classifier and return test-set evidence."""
    if not samples:
        return {"status": "no_messages", "n": 0}
    if len({s.label for s in samples}) < 2:
        return {
            "status": "insufficient_classes",
            "n": len(samples),
            "class_counts": {str(c): sum(s.label == c for s in samples) for c in sorted({s.label for s in samples})},
        }
    try:
        train_idx, test_idx = _split_groups(samples, test_size, seed)
    except ValueError as exc:
        return {"status": "insufficient_groups", "n": len(samples), "error": str(exc)}

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "Message classification requires sentence-transformers and scikit-learn. "
            "Install requirements-classification.txt."
        ) from exc

    encoder = SentenceTransformer(model_name)
    texts = [s.text for s in samples]
    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    classifier.fit(embeddings[train_idx], [samples[i].label for i in train_idx])
    scores_all = classifier.predict_proba(embeddings)[:, 1]
    preds_all = (scores_all >= 0.5).astype(int).tolist()
    overall = _group_metrics(samples, test_idx, scores_all, preds_all)

    by_attack: Dict[str, Any] = {}
    by_round: Dict[str, Any] = {}
    for key, target in (("attack_type", by_attack), ("round_id", by_round)):
        values = sorted({str(getattr(samples[i], key)) for i in test_idx})
        for value in values:
            idx = [i for i in test_idx if str(getattr(samples[i], key)) == value]
            target[value] = _group_metrics(samples, idx, scores_all, preds_all) if len({samples[i].label for i in idx}) == 2 else {
                "status": "insufficient_classes",
                "n": len(idx),
                "class_counts": {str(c): sum(samples[i].label == c for i in idx) for c in sorted({samples[i].label for i in idx})},
            }

    predictions = []
    for i in test_idx:
        predictions.append({
            "step": samples[i].step,
            "agent_id": samples[i].agent_id,
            "round_id": samples[i].round_id,
            "attack_type": samples[i].attack_type,
            "label": samples[i].label,
            "predicted": preds_all[i],
            "score_malicious": float(scores_all[i]),
        })
    return {
        "status": "ok",
        "model": model_name,
        "n": len(samples),
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "overall": overall,
        "by_attack_type": by_attack,
        "by_round": by_round,
        "predictions": predictions,
    }


def write_report(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_separability_report(
    report: Dict[str, Any],
    path: str | Path,
    context: Dict[str, Any] | None = None,
) -> None:
    """Write a compact, paper-ready separability summary separate from raw logs."""
    context = context or {}
    lines = [
        "# Benign/Malicious Message Separability",
        "",
        "## Experiment configuration",
        "",
    ]
    config_rows = [
        ("LLM model", context.get("llm_model", "unknown")),
        ("Dataset", context.get("dataset", "unknown")),
        ("Attack", context.get("attack", "unknown")),
        ("Questions", context.get("questions", "unknown")),
        ("Agents / rounds", f"{context.get('n_agents', 'unknown')} / {context.get('n_rounds', 'unknown')}"),
        ("Embedding model", report.get("model", "unknown")),
        ("Split", f"question-group holdout ({context.get('test_size', 'unknown')})"),
    ]
    lines.extend(["| Item | Value |", "|---|---|"])
    lines.extend(f"| {key} | {value} |" for key, value in config_rows)

    if report.get("status") != "ok":
        lines.extend(["", f"**Status:** `{report.get('status', 'unknown')}`"])
        if report.get("error"):
            lines.append(f"\n**Error:** {report['error']}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    overall = report["overall"]
    lines.extend([
        "",
        "## Test-set results",
        "",
        "Labels: `0 = benign agent message`, `1 = attacker-controlled message`.",
        "",
        f"Test messages: **{report['test_n']}** (train: {report['train_n']}; total: {report['n']})",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {overall['accuracy']:.3f} |",
        f"| Balanced accuracy | {overall['balanced_accuracy']:.3f} |",
        f"| Precision | {overall['precision']:.3f} |",
        f"| Recall | {overall['recall']:.3f} |",
        f"| F1 | {overall['f1']:.3f} |",
        f"| ROC-AUC | **{overall['roc_auc']:.3f}** |",
        f"| Confusion matrix | `{overall['confusion_matrix']}` |",
    ])

    by_round = report.get("by_round", {})
    if by_round:
        lines.extend(["", "## Results by communication round", "", "| Round | N | Balanced accuracy | F1 | ROC-AUC |", "|---:|---:|---:|---:|---:|"])
        for round_id, metrics in by_round.items():
            if metrics.get("status") == "ok":
                lines.append(
                    f"| {round_id} | {metrics['n']} | {metrics['balanced_accuracy']:.3f} | "
                    f"{metrics['f1']:.3f} | {metrics['roc_auc']:.3f} |"
                )

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def format_report(report: Dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return f"message classification: {report.get('status')} ({report.get('error', '')})".strip()
    m = report["overall"]
    lines = [
        "message classification (test set): "
        f"n={report['test_n']} accuracy={m['accuracy']:.3f} "
        f"balanced_accuracy={m['balanced_accuracy']:.3f} f1={m['f1']:.3f} "
        f"roc_auc={m['roc_auc']:.3f} confusion_matrix={m['confusion_matrix']}"
    ]
    for key, values in (("attack_type", report.get("by_attack_type", {})), ("round", report.get("by_round", {}))):
        for value, metrics in values.items():
            if metrics.get("status") == "ok":
                lines.append(
                    f"  {key}={value}: n={metrics['n']} "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
                    f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}"
                )
            else:
                lines.append(f"  {key}={value}: {metrics.get('status')}")
    return "\n".join(lines)
