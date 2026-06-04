@echo off
title Delaunay Art Studio Launcher
echo =================================================================
echo        Delaunay Art Studio - Computational Geometry Demo
echo =================================================================
echo.
echo [*] Starting Python HTTP Server on port 8000...
start /b python -m http.server 8000
timeout /t 2 >nul
echo [*] Launching Delaunay Art Studio in default browser...
start "" http://localhost:8000/index.html
echo.
echo [+] Server is running at http://localhost:8000/
echo [!] Keep this terminal window open to keep the server active.
echo [!] Close this window or press Ctrl+C inside it to stop.
echo.
pause
