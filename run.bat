@echo off
setlocal
title Event Label Studio
cd /d "%~dp0"

set "BACKEND_APP=prototype\backend\app.py"
set "FRONTEND_DIR=prototype\frontend"

rem ---- Python interpreter -------------------------------------------------
rem Resolution order: user-set %PYTHON_EXE%  ->  the project's conda env
rem (eventcamera-blender)  ->  python / py on PATH. The conda-env check fixes
rem the common case where PATH `python` is some other install missing the deps.
if not defined PYTHON_EXE if exist "%USERPROFILE%\.conda\envs\eventcamera-blender\python.exe" set "PYTHON_EXE=%USERPROFILE%\.conda\envs\eventcamera-blender\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\eventcamera-blender\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\eventcamera-blender\python.exe"
if not defined PYTHON_EXE (where python >nul 2>&1 && set "PYTHON_EXE=python")
if not defined PYTHON_EXE (where py >nul 2>&1 && set "PYTHON_EXE=py")
if not defined PYTHON_EXE (
    echo [ERROR] Python not found.
    echo         Create the conda env:  conda env create -f environment.yml
    echo         or set a specific interpreter:  set "PYTHON_EXE=C:\path\to\python.exe"
    pause
    exit /b 1
)
echo [setup] Using Python: %PYTHON_EXE%

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
