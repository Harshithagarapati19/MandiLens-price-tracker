@echo off
setlocal
title Push IntelliMandi to GitHub
echo ========================================================
echo   Push Intelligent Mandi Price Tracker to GitHub
echo ========================================================
echo.

set DEFAULT_REPO=https://github.com/Harshithagarapati19/Intelligent_Mandi_Price_Tracker.git
echo Default Repository: %DEFAULT_REPO%
set /p REPO_URL="Enter GitHub Repository URL (Press Enter to use default): "

if "%REPO_URL%"=="" (
    set REPO_URL=%DEFAULT_REPO%
)

echo.
echo Initializing / Checking Git repository...
if not exist .git (
    git init
    git config user.name "Harshitha Garapati"
    git config user.email "harshithagarapati19@gmail.com"
)

git add .
git commit -m "Initial commit: Intelligent Mandi Price Trends, Anomaly Detection & SARIMA Forecasting" 2>nul

echo Renaming branch to main...
git branch -M main

echo Configuring remote origin (%REPO_URL%)...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo Pushing codebase to GitHub...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   SUCCESS! Project has been pushed to GitHub.
    echo ========================================================
) else (
    echo.
    echo Push encountered an error. Please verify:
    echo 1. You have created the repository on GitHub at %REPO_URL%
    echo 2. You are logged into GitHub in your terminal/browser.
)

echo.
pause
