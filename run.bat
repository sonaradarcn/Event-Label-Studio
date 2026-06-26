@echo off
setlocal
title Event Label Studio
cd /d "%~dp0"

set "BACKEND_APP=prototype\backend\app.py"
set "FRONTEND_DIR=prototype\frontend"

rem ---- Python interpreter -------------------------------------------------
rem Honour a user-provided %PYTHON_EXE% (e.g. a specific conda env's
rem python.exe); otherwise fall back to `python` / `py` on PATH.
rem Tip: activate your environment first, e.g.  conda activate eventcamera-blender
rem  or set a specific one:  set "PYTHON_EXE=C:\path\to\python.exe"
if defined PYTHON_EXE goto have_python
where python >nul 2>&1 && (set "PYTHON_EXE=python" & goto have_python)
where py >nul 2>&1 && (set "PYTHON_EXE=py" & goto have_python)
echo [ERROR] Python not found on PATH.
echo         Install Python 3.10 (or activate your conda env), or set PYTHON_EXE
echo         to a specific interpreter, e.g.:
echo             set "PYTHON_EXE=C:\path\to\python.exe"
pause
exit /b 1
:have_python

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found in PATH. Install Node.js first.
    pause
    exit /b 1
)

rem Stop a stale backend left over from a previous session, otherwise
rem the new backend cannot bind port 5050.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /c:"127.0.0.1:5050 " ^| findstr "LISTENING"') do (
    echo [setup] Stopping stale backend on port 5050 ^(PID %%p^)...
    taskkill /f /pid %%p >nul 2>&1
)

rem Check for the vite launcher itself, not just node_modules: a
rem node_modules tree installed on another OS exists but cannot run.
if not exist "%FRONTEND_DIR%\node_modules\.bin\vite.cmd" (
    echo [setup] First run: installing frontend dependencies...
    pushd "%FRONTEND_DIR%"
    call npm install
    popd
)

echo.
echo  ============================================
echo   Event Label Studio
echo   Backend  : http://127.0.0.1:5050
echo   Frontend : http://127.0.0.1:5173
echo.
echo   CLOSE THIS WINDOW TO STOP THE WHOLE APP
echo  ============================================
echo.

rem Backend runs in the background but stays attached to THIS console,
rem so closing the window terminates it together with the frontend.
start "" /b "%PYTHON_EXE%" "%BACKEND_APP%"

rem Open the browser once the dev server has had time to come up.
start "" /b cmd /c "ping -n 5 127.0.0.1 >nul & start http://127.0.0.1:5173"

rem Frontend runs in the foreground and keeps this window alive.
cd "%FRONTEND_DIR%"
call npm run dev

rem If the dev server exits on its own (q / Ctrl+C / crash),
rem also stop the backend instead of leaving it orphaned.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /c:"127.0.0.1:5050 " ^| findstr "LISTENING"') do (
    taskkill /f /pid %%p >nul 2>&1
)
endlocal
