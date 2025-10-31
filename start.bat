@echo off
REM Start Unblur App with Python Backend
REM Windows Batch Script

setlocal enabledelayedexpansion

echo.
echo 🚀 Starting Unblur AI App...
echo.

REM Get project root directory
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

REM Check Python installation
echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed
    echo    Install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo ✅ Python found: %PYTHON_VERSION%

REM Check Node.js installation
echo 🔍 Checking Node.js installation...
where node >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed
    echo    Install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

for /f %%v in ('node --version') do set NODE_VERSION=%%v
echo ✅ Node.js found: %NODE_VERSION%

REM Check npm installation
where npm >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed
    echo    npm should be installed together with Node.js
    pause
    exit /b 1
)

for /f %%v in ('npm --version') do set NPM_VERSION=%%v
echo ✅ npm found: %NPM_VERSION%
echo.

REM Check Python virtual environment
echo 🔍 Checking Python virtual environment...
if not exist ".venv" (
    echo ⚠️  Python virtual environment (.venv) not found.
    echo    Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Could not create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔍 Activating virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ❌ Cannot activate virtual environment
    pause
    exit /b 1
)

REM Check Python dependencies
echo 🔍 Checking Python dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Python dependencies not found.
    echo    Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Could not install Python dependencies
        pause
        exit /b 1
    )
    echo ✅ Python dependencies installed
) else (
    echo ✅ Python dependencies found
)

REM Check Node.js dependencies
echo 🔍 Checking Node.js dependencies...
if not exist "node_modules" (
    echo ⚠️  Node.js dependencies not found.
    echo    Installing npm packages...
    call npm install
    if errorlevel 1 (
        echo ❌ Could not install Node.js dependencies
        pause
        exit /b 1
    )
    echo ✅ Node.js dependencies installed
) else (
    echo ✅ Node.js dependencies found
)

REM Check models directory
if not exist "models" (
    echo ⚠️  Models directory not found, creating...
    mkdir models
    echo ✅ Models directory created
)

echo.
echo ✅ All checks passed!
echo.

REM Start Python backend in new window
echo ✅ Starting Python backend on port 5000...
start "Unblur Backend" cmd /c ".venv\Scripts\python.exe unblur_server.py"
timeout /t 2 /nobreak >nul

REM Start Next.js frontend in new window
echo ✅ Starting Next.js frontend on port 3000...
start "Unblur Frontend" cmd /c "npm run dev"
timeout /t 2 /nobreak >nul

echo.
echo 🎉 Unblur AI App is running!
echo 📱 Frontend: http://localhost:3000
echo 🐍 Backend:  http://localhost:5000
echo.
echo The application is running in separate windows.
echo Close those windows to stop, or press any key here...
echo.

REM Wait for user to stop
pause >nul

REM Cleanup - kill processes by window title
echo.
echo 🛑 Stopping Unblur AI App...
taskkill /FI "WINDOWTITLE eq Unblur Backend*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Unblur Frontend*" /T /F >nul 2>&1
REM Also try to kill by process name as fallback
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *unblur*" >nul 2>&1
taskkill /F /IM node.exe /FI "WINDOWTITLE eq *npm*" >nul 2>&1
echo 👋 Bye!

endlocal
