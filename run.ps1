# AURA-Face Launcher Script (inside project directory)
$env:PYTHONPATH = "src"
Write-Host "[+] PYTHONPATH set to 'src'" -ForegroundColor Cyan
Write-Host "[+] Launching AURA-Face Cockpit HUD..." -ForegroundColor Green

python -m aura_face.cli run $args
