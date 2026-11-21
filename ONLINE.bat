@echo off
REM HVAC Operations Assistant - ONLINE mode for the group (ITAI 2376 Group 2)
REM Starts the agent and a Cloudflare tunnel. Share the trycloudflare.com link + the access code.
REM Your computer must stay on with both windows open. Close the windows to go offline.
cd /d "%~dp0"
echo Checking the libraries...
python -m pip install -q pandas numpy scikit-learn anthropic

set CF=
where cloudflared >nul 2>nul && set CF=cloudflared
if not defined CF if exist cloudflared.exe set CF=cloudflared.exe
if not defined CF (
  echo Downloading cloudflared, Cloudflare's official tunnel program, about 60 MB, only the first time...
  curl -L -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
  set CF=cloudflared.exe
)

echo Starting the agent in a second window (the ACCESS CODE appears there)...
start "HVAC Agent - ACCESS CODE" cmd /k python server.py --online
timeout /t 10 /nobreak >nul

echo.
echo ================================================================
echo  Look below for a line like:  https://something.trycloudflare.com
echo  Send that link + the ACCESS CODE (other window) to your group.
echo ================================================================
echo.
%CF% tunnel --url http://localhost:8000
pause
