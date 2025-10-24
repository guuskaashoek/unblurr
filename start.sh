#!/bin/bash

# Start Unblur App with Python Backend

set -e

echo "🚀 Starting Unblur AI App..."
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Check if Python virtual environment is present
if [ ! -d ".venv" ]; then
    echo "❌ Python virtual environment (.venv) not found."
    echo "   Please run the setup instructions to create it before starting."
    exit 1
fi

# Check if Node is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    exit 1
fi

source .venv/bin/activate

echo "✅ Starting Python backend on port 5000..."
python unblur_server.py &
PYTHON_PID=$!

echo "✅ Starting Next.js frontend on port 3000..."
npm run dev &
NEXT_PID=$!

cleanup() {
    echo ""
    echo "🛑 Stopping Unblur AI App..."
    kill "$PYTHON_PID" "$NEXT_PID" 2>/dev/null || true
    wait "$PYTHON_PID" "$NEXT_PID" 2>/dev/null || true
    deactivate 2>/dev/null || true
    echo "👋 Bye!"
}

trap cleanup INT TERM

echo ""
echo "🎉 Unblur AI App is running!"
echo "📱 Frontend: http://localhost:3000"
echo "🐍 Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for both processes
wait "$PYTHON_PID" "$NEXT_PID"
