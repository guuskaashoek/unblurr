# Start Unblur App with Python Backend
# Windows PowerShell Script

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🚀 Starting Unblur AI App..." -ForegroundColor Cyan
Write-Host ""

# Get project root directory
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PROJECT_ROOT

# Check Python installation
Write-Host "🔍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed" -ForegroundColor Red
    Write-Host "   Install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Node.js installation
Write-Host "🔍 Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    if ($LASTEXITCODE -ne 0) {
        throw "Node.js not found"
    }
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js is not installed" -ForegroundColor Red
    Write-Host "   Install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check npm installation
try {
    $npmVersion = npm --version
    if ($LASTEXITCODE -ne 0) {
        throw "npm not found"
    }
    Write-Host "✅ npm found: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ npm is not installed" -ForegroundColor Red
    Write-Host "   npm should be installed together with Node.js" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check Python virtual environment
Write-Host "🔍 Checking Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "⚠️  Python virtual environment (.venv) not found." -ForegroundColor Yellow
    Write-Host "   Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Could not create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔍 Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Cannot activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Python dependencies
Write-Host "🔍 Checking Python dependencies..." -ForegroundColor Yellow
try {
    python -c "import flask" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Flask not installed"
    }
    Write-Host "✅ Python dependencies found" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Python dependencies not found." -ForegroundColor Yellow
    Write-Host "   Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Could not install Python dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Python dependencies installed" -ForegroundColor Green
}

# Check Node.js dependencies
Write-Host "🔍 Checking Node.js dependencies..." -ForegroundColor Yellow
if (-not (Test-Path "node_modules")) {
    Write-Host "⚠️  Node.js dependencies not found." -ForegroundColor Yellow
    Write-Host "   Installing npm packages..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Could not install Node.js dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Node.js dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Node.js dependencies found" -ForegroundColor Green
}

# Check models directory
if (-not (Test-Path "models")) {
    Write-Host "⚠️  Models directory not found, creating..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "models" | Out-Null
    Write-Host "✅ Models directory created" -ForegroundColor Green
}

Write-Host ""
Write-Host "✅ All checks passed!" -ForegroundColor Green
Write-Host ""

# Function to cleanup on exit
function Cleanup {
    Write-Host ""
    Write-Host "🛑 Stopping Unblur AI App..." -ForegroundColor Yellow
    
    if ($pythonProcess -and !$pythonProcess.HasExited) {
        Stop-Process -Id $pythonProcess.Id -Force -ErrorAction SilentlyContinue
    }
    
    if ($nextProcess -and !$nextProcess.HasExited) {
        Stop-Process -Id $nextProcess.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Also kill any remaining processes
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*unblurr*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*next dev*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Host "👋 Bye!" -ForegroundColor Cyan
}

# Register cleanup on exit
Register-EngineEvent PowerShell.Exiting -Action { Cleanup } | Out-Null

# Start Python backend
Write-Host "✅ Starting Python backend on port 5000..." -ForegroundColor Green
$pythonExe = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}
$pythonProcess = Start-Process -FilePath $pythonExe -ArgumentList "unblur_server.py" -PassThru -NoNewWindow -WorkingDirectory $PROJECT_ROOT

Start-Sleep -Seconds 2

# Start Next.js frontend
Write-Host "✅ Starting Next.js frontend on port 3000..." -ForegroundColor Green
$nextProcess = Start-Process -FilePath "npm" -ArgumentList "run", "dev" -PassThru -NoNewWindow -WorkingDirectory $PROJECT_ROOT

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "🎉 Unblur AI App is running!" -ForegroundColor Green
Write-Host "📱 Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "🐍 Backend:  http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop..." -ForegroundColor Yellow
Write-Host ""

# Wait for user interrupt
try {
    while ($true) {
        if ($pythonProcess.HasExited -or $nextProcess.HasExited) {
            Write-Host "⚠️  One of the processes has stopped" -ForegroundColor Yellow
            break
        }
        Start-Sleep -Seconds 1
    }
} catch {
    # User pressed Ctrl+C
} finally {
    Cleanup
}
