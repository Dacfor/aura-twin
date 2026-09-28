@echo off
set PYTHONPATH=src
echo [+] Launching AURA-Face Cockpit HUD...
python -m aura_face.cli run %*
