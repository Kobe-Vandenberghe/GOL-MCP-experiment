Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment..."
    py -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

Write-Host "Starting Game of Life MCP server on http://localhost:8000"
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
