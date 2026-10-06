param(
    [string]$RepoPath = 'D:\codex project\rethinking-mas-defense',
    [string]$DataFile = 'D:\codex project\rethinking-mas-defense\datasets\MMLU\mmlu_test.jsonl',
    [string]$Model = 'qwen3.5-35b-a3b',
    [string]$PythonExe = 'python',
    [int]$MaxConcurrency = 8,
    [int]$EvalConcurrency = 4,
    [int]$NSamples = 20,
    [int]$Seed = 2026
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $RepoPath -PathType Container)) { throw "Repository directory not found: $RepoPath" }
$Evaluate = Join-Path $RepoPath 'evaluate.py'
if (-not (Test-Path -LiteralPath $Evaluate -PathType Leaf)) { throw "evaluate.py not found: $Evaluate" }
if (-not (Test-Path -LiteralPath $DataFile -PathType Leaf)) { throw "MMLU data file not found: $DataFile" }

$ApiKey = Read-Host 'Enter Bailian API key (not saved to disk)'
if ([string]::IsNullOrWhiteSpace($ApiKey)) { throw 'API key cannot be empty' }

Set-Location -LiteralPath $RepoPath
$ResultsDir = Join-Path $RepoPath 'results'
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$HelpText = (& $PythonExe $Evaluate --help 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) { throw "Python cannot run evaluate.py. Check dependencies or pass -PythonExe. Current value: $PythonExe`n$HelpText" }
$ThinkingArgs = @()
if ($HelpText -match '--disable_thinking') { $ThinkingArgs = @('--disable_thinking') }

function Invoke-Experiment {
    param([string]$Name, [string[]]$ExtraArgs)
    $OutFile = Join-Path $ResultsDir "mmlu_${Model}_s${Seed}_${Name}.jsonl"
    if (Test-Path -LiteralPath $OutFile -PathType Leaf) { Remove-Item -LiteralPath $OutFile -Force }
    Write-Host "`n===== seed=$Seed $Name =====" -ForegroundColor Cyan
    $Common = @(
        '--data_file', $DataFile, '--dataset', 'mmlu',
        '--base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        '--api_key', $ApiKey, '--model', $Model,
        '--n_agents', '5', '--n_rounds', '3', '--temperature', '0.3',
        '--max_tokens', '512', '--max_concurrency', [string]$MaxConcurrency,
        '--eval_concurrency', [string]$EvalConcurrency, '--topology', 'full',
        '--topology_density', '0.3', '--topology_seed', '0', '--n_attackers', '2',
        '--attack_temperature', '0.9', '--attack_max_tokens', '512',
        '--n_samples', [string]$NSamples, '--seed', [string]$Seed,
        '--top_k_conf', '10', '--out_file', $OutFile
    ) + $ThinkingArgs + $ExtraArgs
    & $PythonExe $Evaluate @Common
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: seed=$Seed $Name (exit code $LASTEXITCODE)" }
}

Write-Host "MMLU experiment plan: seed=$Seed, 11 settings, $NSamples samples." -ForegroundColor Yellow
Invoke-Experiment 'clean' @('--attack','none','--defense','none')
Invoke-Experiment 'overt' @('--attack','overt','--defense','none')
foreach ($AttackName in @('slow_drift','benign_wrapper','chaos_seeding')) {
    Invoke-Experiment $AttackName @('--attack',$AttackName,'--defense','none')
    Invoke-Experiment "${AttackName}_pruning_d04" @('--attack',$AttackName,'--defense','pruning','--prune_threshold','0.4')
    Invoke-Experiment "${AttackName}_downweight" @('--attack',$AttackName,'--defense','downweight')
}

$Calls = 11 * $NSamples * 5 * 3
Write-Host "`nAll MMLU experiments completed. Estimated API calls: $Calls" -ForegroundColor Green
Write-Host "Results: $ResultsDir" -ForegroundColor Green
