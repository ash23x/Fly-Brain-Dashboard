@echo off
rem Starts both fly-brain servers (minimised windows -- close them to stop).
rem   compass  -> http://localhost:8765/
rem   learning -> http://localhost:8766/
cd /d E:\fly-brain-dashboard
set PYTHONIOENCODING=utf-8
start "fly compass (8765)" /min .\.venv\Scripts\python.exe server\sim_server.py --mode compass --port 8765
start "fly learning (8766)" /min .\.venv\Scripts\python.exe server\sim_server.py --mode mushroom --port 8766
timeout /t 5 /nobreak >nul
start "" http://localhost:8765/
