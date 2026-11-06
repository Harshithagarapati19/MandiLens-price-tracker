@echo off
setlocal
title Push IntelliMandi to GitHub
echo ========================================================
echo   Push Intelligent Mandi Price Tracker to GitHub
echo ========================================================
echo.

set DEFAULT_REPO=https://github.com/Harshithagarapati19/MandiLens-price-tracker.git
echo Target Repository: %DEFAULT_REPO%
echo.
set /p REPO_URL="Enter GitHub Repository URL (Press ENTER to use target): "

if "%REPO_URL%"=="" (
    set REPO_URL=%DEFAULT_REPO%
)

echo.
echo Checking Git status...
git add .
git commit -m "feat: Intelligent Mandi Price Tracker with SARIMA Forecasting & Anomaly Detection" 2>nul
git branch -M main

echo Setting remote origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo Pushing code to GitHub (%REPO_URL%)...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   SUCCESS! Project has been pushed to GitHub.
    echo ========================================================
) else (
    echo.
    echo Push encountered an error. Please verify your GitHub credentials.
)

echo.
pause
