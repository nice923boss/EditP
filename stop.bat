@echo off
rem ask EditP to quit (removes its tray icon), then make sure no process is left
curl -s -m 3 -X POST http://127.0.0.1:8793/api/shutdown 1>nul 2>nul
timeout /t 2 1>nul
taskkill /f /im pythonw_editp.exe 1>nul 2>nul
echo EditP stopped.
timeout /t 2 1>nul
