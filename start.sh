#!/bin/bash

# Start Unblur App with Python Backend
# Cross-platform script for Mac and Linux

set -e

echo "🚀 Starting Unblur AI App..."
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE="Linux";;
    Darwin*)    OS_TYPE="Mac";;
    *)          OS_TYPE="Unknown";;
esac

echo "🖥️  Operating System: $OS_TYPE"
echo ""

# Check for libgl1 on Linux (required for OpenCV)
if [ "$OS_TYPE" = "Linux" ]; then
    echo "🔍 Checking for libgl1 (required for OpenCV)..."
    if ! ldconfig -p | grep -q libGL.so.1 2>/dev/null; then
        echo "⚠️  libgl1 not found. Attempting to install..."
        if command -v apt-get &> /dev/null; then
            if [ "$EUID" -eq 0 ]; then
                apt-get update && apt-get install -y libgl1
            else
                echo "   Running: sudo apt-get update && sudo apt-get install -y libgl1"
                sudo apt-get update && sudo apt-get install -y libgl1
            fi
            if [ $? -eq 0 ]; then
                echo "✅ libgl1 installed successfully"
            else
                echo "⚠️  Warning: Could not install libgl1 automatically"
                echo "   You may need to run manually: sudo apt-get update && sudo apt-get install -y libgl1"
            fi
        else
            echo "⚠️  Warning: apt-get not found. Please install libgl1 manually"
            echo "   For Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y libgl1"
        fi
    else
        echo "✅ libgl1 found"
    fi
    echo ""
fi

# Check Python installation
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed"
    echo "   Install Python 3.8+ from https://www.python.org/downloads/"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Python found: $PYTHON_VERSION"

# Check Python version (must be 3.8+)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Python 3.8+ is required, but you have $PYTHON_VERSION"
    exit 1
fi

# Check Node.js installation
echo "🔍 Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "   Install Node.js from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✅ Node.js found: $NODE_VERSION"

# Check npm installation
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed"
    echo "   npm should be installed together with Node.js"
    exit 1
fi

NPM_VERSION=$(npm --version)
echo "✅ npm found: $NPM_VERSION"
echo ""

# Check Python virtual environment
echo "🔍 Checking Python virtual environment..."
if [ ! -d ".venv" ]; then
    echo "⚠️  Python virtual environment (.venv) not found."
    echo "   Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔍 Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "❌ Cannot activate virtual environment"
    exit 1
fi

# Check if Python packages are installed
echo "🔍 Checking Python dependencies..."
if ! $PYTHON_CMD -c "import flask" 2>/dev/null; then
    echo "⚠️  Python dependencies not found."
    echo "   Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "✅ Python dependencies found"
fi

# Check Node.js packages
echo "🔍 Checking Node.js dependencies..."
if [ ! -d "node_modules" ]; then
    echo "⚠️  Node.js dependencies not found."
    echo "   Installing npm packages..."
    npm install
    echo "✅ Node.js dependencies installed"
else
    echo "✅ Node.js dependencies found"
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "⚠️  Models directory not found, creating..."
    mkdir -p models
    echo "✅ Models directory created"
fi

echo ""
echo "✅ All checks passed!"
echo ""

# Check if ports are available
check_port() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 || nc -z localhost $port 2>/dev/null; then
        echo "⚠️  Warning: Port $port ($name) appears to be in use"
        echo "   The application may not be able to start"
    fi
}

check_port 5000 "Backend"
check_port 3000 "Frontend"
echo ""

# Start Python backend
echo "✅ Starting Python backend on port 5000..."
$PYTHON_CMD unblur_server.py &
PYTHON_PID=$!

# Wait a bit so backend can start
sleep 2

# Check if backend started successfully
if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "❌ Python backend could not start"
    exit 1
fi

# Start Next.js frontend
echo "✅ Starting Next.js frontend on port 3000..."
npm run dev &
NEXT_PID=$!

# Wait a bit so frontend can start
sleep 2

# Check if frontend started successfully
if ! kill -0 $NEXT_PID 2>/dev/null; then
    echo "❌ Next.js frontend could not start"
    kill $PYTHON_PID 2>/dev/null || true
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping Unblur AI App..."
    
    # Stop Python backend
    if kill -0 $PYTHON_PID 2>/dev/null; then
        kill $PYTHON_PID 2>/dev/null || true
        wait $PYTHON_PID 2>/dev/null || true
    fi
    
    # Stop Next.js frontend
    if kill -0 $NEXT_PID 2>/dev/null; then
        kill $NEXT_PID 2>/dev/null || true
        wait $NEXT_PID 2>/dev/null || true
    fi
    
    # Deactivate virtual environment
    deactivate 2>/dev/null || true
    
    echo "👋 Bye!"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup INT TERM

echo ""
echo "🎉 Unblur AI App is running!"
echo "📱 Frontend: http://localhost:3000"
echo "🐍 Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

# Wait for both processes
wait $PYTHON_PID $NEXT_PID
