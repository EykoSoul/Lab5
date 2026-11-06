$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "Эксперименты: $root\src\run_experiments.py"
& $python "$root\src\run_experiments.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Отчёт: $root\src\build_report.py"
& $python "$root\src\build_report.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Готово: см. $root\results и $root\output" -ForegroundColor Green