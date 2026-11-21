@echo off
REM HVAC Operations Assistant - ITAI 2376 Group 2 (14562)
REM Double-click to start the agent team and open the chat page in the browser.
cd /d "%~dp0"
echo Installing / checking the libraries (first time takes a few minutes)...
python -m pip install -q pandas numpy scikit-learn anthropic
echo.
echo Optional for the best manual search: python -m pip install sentence-transformers
echo.
python server.py
pause
