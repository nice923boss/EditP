@echo off
cd /d "%~dp0"
set "RC=1"
set "PY="
for /f "delims=" %%i in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%i"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%ProgramFiles%\Python312\python.exe" set "PY=%ProgramFiles%\Python312\python.exe"
if defined PY goto run
echo Python 3.12 not found. Installing it with winget...
winget install -e --id Python.Python.3.12 --scope user
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto run
echo.
echo Python 3.12 could not be installed automatically.
echo Install it from https://www.python.org/downloads/ then run install.bat again.
goto end
:run
"%PY%" "%~dp0install.py"
set "RC=%errorlevel%"
:end
if not "%~1"=="nopause" pause
exit /b %RC%
