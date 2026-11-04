@echo off
cd /d "%~dp0"
if not exist "%~dp0pythonw_editp.exe" goto install
rem a copied folder may point at another PC's Python, or miss a newer package: check before launching
"%~dp0pythonw_editp.exe" "%~dp0launcher.py" --check
if errorlevel 1 goto install
goto launch
:install
call "%~dp0install.bat" nopause
if errorlevel 1 goto failed
:launch
start "" "%~dp0pythonw_editp.exe" "%~dp0launcher.py"
exit /b 0
:failed
pause
exit /b 1
