param(
    [string]$RepoPath = $PSScriptRoot,
    [string]$DataFile = '',
    [string]$Model = 'qwen3.5-35b-a3b',
    [string]$PythonExe = 'python',
    [string]$EmbeddingModel = 'sentence-transformers/all-MiniLM-L6-v2',
    [string]$PayloadTokenizerPath = '',
    [int]$MaxConcurrency = 8,
    [int]$EvalConcurrency = 4,
    [int]$NSamples = 20,
    [int]$Seed = 2026,
    [switch]$ClassifyMessages
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $RepoPath -PathType Container)) { throw "Repository directory not found: $RepoPath" }
$RepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
if (-not $DataFile) { $DataFile = Join-Path $RepoPath 'datasets/raw_backup/mmlu_test_full.jsonl' }
if (-not (Test-Path -LiteralPath $RepoPath -PathType Container)) { throw "Repository directory not found: $RepoPath" }
$Evaluate = Join-Path $RepoPath 'evaluate.py'
if (-not (Test-Path -LiteralPath $Evaluate -PathType Leaf)) { throw "evaluate.py not found: $Evaluate" }
if (-not (Test-Path -LiteralPath $DataFile -PathType Leaf)) { throw "MMLU data file not found: $DataFile" }
if ([string]::IsNullOrWhiteSpace($PayloadTokenizerPath) -or -not (Test-Path -LiteralPath $PayloadTokenizerPath -PathType Container)) {
    throw 'Pass -PayloadTokenizerPath with a local tokenizer for the requested model. Wrapper requires token counts. This cloud profile is diagnostic; serving-tokenizer identity is unverified.'
}
$DataFile = (Resolve-Path -LiteralPath $DataFile).Path
$PayloadTokenizerPath = (Resolve-Path -LiteralPath $PayloadTokenizerPath).Path
if (Test-Path -LiteralPath $EmbeddingModel -PathType Container) { $EmbeddingModel = (Resolve-Path -LiteralPath $EmbeddingModel).Path }

$ResultsDir = Join-Path $RepoPath ("results/repaired/bailian_" + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $ResultsDir) { throw "Choose a fresh results directory: $ResultsDir" }
New-Item -ItemType Directory -Path $ResultsDir | Out-Null
$CheckpointDir = Join-Path $ResultsDir 'checkpoints'
$PreviousKey = $env:OPENAI_API_KEY
$PreviousLocation = Get-Location

try {
    if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
        $SecureKey = Read-Host 'Enter Bailian API key (not saved to disk)' -AsSecureString
        $KeyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
        try { $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($KeyPointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($KeyPointer) }
    }
    if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw 'API key cannot be empty' }
    Set-Location -LiteralPath $RepoPath

    function Invoke-Experiment {
        param([string]$Name, [string[]]$ExtraArgs)
        $OutFile = Join-Path $ResultsDir ("mmlu_s" + $Seed + "_" + $Name + ".jsonl")
        Write-Host ("`nseed=" + $Seed + " " + $Name) -ForegroundColor Cyan
        $Common = @(
            '--data_file', $DataFile, '--dataset', 'mmlu',
            '--llm_transport', 'direct',
            '--base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '--model', $Model, '--disable_thinking',
            '--embedding_model', $EmbeddingModel,
            '--payload_tokenizer_path', $PayloadTokenizerPath,
            '--n_agents', '5', '--n_rounds', '3', '--temperature', '0.3',
            '--max_tokens', '512', '--max_concurrency', [string]$MaxConcurrency,
            '--eval_concurrency', [string]$EvalConcurrency,
            '--topology', 'full', '--topology_seed_strategy', 'fixed',
            '--attacker_placement', 'fixed', '--n_attackers', '2',
            '--attack_candidates', '4', '--attack_temperature', '0.9',
            '--attack_max_tokens', '512',
            '--n_samples', [string]$NSamples, '--seed', [string]$Seed,
            '--top_k_conf', '10', '--top_logprobs', '5',
            '--confidence_entropy_mode', 'top_logprobs_tail_bucket',
            '--out_file', $OutFile, '--checkpoint_dir', $CheckpointDir
        ) + $ExtraArgs
        if ($ClassifyMessages) { $Common += '--classify_messages' }
        & $PythonExe $Evaluate @Common
        if ($LASTEXITCODE -ne 0) {
            throw "Condition failed or incomplete: $Name (exit $LASTEXITCODE). Inspect summary/checkpoint; do not count it as a complete experiment."
        }
    }

    Write-Host "Cloud diagnostic: seed=$Seed, 11 settings, $NSamples samples; model/data/entropy differ from formal paper runs." -ForegroundColor Yellow
    Invoke-Experiment 'clean' @('--attack','none','--defense','none')
    Invoke-Experiment 'overt' @('--attack','overt','--defense','none')
    foreach ($AttackName in @('slow_drift','benign_wrapper','chaos_seeding')) {
        Invoke-Experiment $AttackName @('--attack',$AttackName,'--defense','none')
        Invoke-Experiment ("${AttackName}_pruning_d04") @('--attack',$AttackName,'--defense','pruning','--prune_threshold','0.4')
        Invoke-Experiment ("${AttackName}_downweight") @('--attack',$AttackName,'--defense','downweight')
    }
    Write-Host "All 11 diagnostic conditions completed. Results: $ResultsDir" -ForegroundColor Green
}
finally {
    $env:OPENAI_API_KEY = $PreviousKey
    Set-Location -LiteralPath $PreviousLocation.Path
}
