# Reproduce every piece of evidence in this folder.
#
# Usage (works with Windows PowerShell 5.1 and PowerShell 7):
#   powershell -ExecutionPolicy Bypass -File "C:\...\\99_repro\run_all_checks.ps1"
#
# The script is ASCII-only on purpose: Windows PowerShell 5.1 reads .ps1 files as ANSI,
# so non-ASCII literals would be mangled. All paths are derived from $PSScriptRoot.
# It only *runs* the pipeline and writes logs; nothing is fabricated.

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent $PSScriptRoot          # the evidence folder
$repo = Join-Path (Split-Path -Parent $root) "dynaguide"
if (-not (Test-Path $repo)) { throw "repo not found: $repo" }
Set-Location $repo
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$summary = @()

function Run-Logged {
    param([string]$Name, [string]$Command, [string]$LogPath)
    Write-Host ""
    Write-Host "### $Name" -ForegroundColor Cyan
    Write-Host "    $Command"
    Set-Content -Encoding utf8 $LogPath "# $stamp"
    Add-Content -Encoding utf8 $LogPath "# $Command"
    cmd /c "$Command >> `"$LogPath`" 2>&1"
    $code = $LASTEXITCODE
    Add-Content -Encoding utf8 $LogPath ""
    Add-Content -Encoding utf8 $LogPath "[exit code: $code]"
    Write-Host "    exit=$code  log=$LogPath"
    return $code
}

# ---------------------------------------------------------------- 0) environment
$ver = (python -c "import sys,torch;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())") -join "; "
Set-Content -Encoding utf8 (Join-Path $root "01_tests\_env.txt") "# environment / $stamp / $ver"

# ---------------------------------------------------------------- 1) test suites
foreach ($t in @("test_flows", "test_flow_dynaguide", "test_satellite_env", "test_robomimic_flow_policy")) {
    $c = Run-Logged $t "python tests\$t.py" (Join-Path $root "01_tests\$t.log")
    $summary += ("test {0,-28} exit={1}" -f $t, $c)
}

# ---------------------------------------------------------------- 2) guided evaluation
$eval = "python scripts\satellite\run_flow_dynaguide.py --policy checkpoints\satellite\flow_policy_payload.pt " +
        "--dynamics checkpoints\satellite\latent_payload.pt --tasks payload_box " +
        "--methods base latent physics both reject --episodes 8 --steps 4 --alpha 30 " +
        "--per_method_scale latent=1.0,physics=0.25,both=0.25 --out results\satellite\evidence_eval.json"
$c = Run-Logged "evaluation" $eval (Join-Path $root "02_eval\eval_stdout.log")
$summary += ("evaluation                     exit={0}" -f $c)

# ---------------------------------------------------------------- 3) surrogate-vs-sim diagnostic
$c = Run-Logged "energy alignment" "python scripts\satellite\check_energy_alignment.py --episodes 4" (Join-Path $root "04_diag\energy_alignment.log")
$summary += ("energy alignment               exit={0}" -f $c)

# ---------------------------------------------------------------- 4) end-to-end from scratch (small)
$tag = "e2e"
$c1 = Run-Logged "e2e collect demos" "python scripts\satellite\make_demos.py --tasks payload_box solar_panel battery_module --per_task 3 --noise 0.03 --tag $tag --seed 1" (Join-Path $root "03_train\e2e_demos.log")
$c2 = Run-Logged "e2e train policy" "python scripts\satellite\train_flow_policy.py --demos data\satellite\demos_$tag.npz --out checkpoints\satellite\${tag}_policy.pt --steps 300 --batch_size 32" (Join-Path $root "03_train\e2e_train_policy.log")
$c3 = Run-Logged "e2e train latent" "python scripts\satellite\train_latent_dynamics.py --demos data\satellite\demos_$tag.npz --out checkpoints\satellite\${tag}_latent.pt --steps 400" (Join-Path $root "03_train\e2e_train_latent.log")
$c4 = Run-Logged "e2e evaluate" "python scripts\satellite\run_flow_dynaguide.py --policy checkpoints\satellite\${tag}_policy.pt --dynamics checkpoints\satellite\${tag}_latent.pt --methods base latent --tasks payload_box --episodes 2 --steps 4 --out results\satellite\${tag}_eval.json" (Join-Path $root "03_train\e2e_eval.log")
$summary += ("e2e pipeline                   demos={0} policy={1} latent={2} eval={3}" -f $c1, $c2, $c3, $c4)

# ---------------------------------------------------------------- 5) refresh copies of result json/md
Copy-Item (Join-Path $repo "results\satellite\*.json") (Join-Path $root "02_eval") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $repo "results\satellite\*.md")   (Join-Path $root "02_eval") -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== summary ===" -ForegroundColor Yellow
$summary | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "logs: $root" -ForegroundColor Yellow
