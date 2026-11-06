@echo off
setlocal enabledelayedexpansion
title Push IntelliMandi to GitHub
echo ======================================================================
echo   Push Intelligent Mandi Price Tracker to GitHub
echo   Target: https://github.com/Harshithagarapati19/MandiLens-price-tracker.git
echo ======================================================================
echo.

cd /d "%~dp0"

echo 1. Staging and committing all project files...
git add .
git commit -m "feat: Intelligent Mandi Price Tracker with SARIMA Forecasting, Anomaly Detection & Streamlit UI" 2>nul
git branch -M main

echo.
echo 2. Configuring remote origin...
git remote remove origin 2>nul
git remote add origin https://github.com/Harshithagarapati19/MandiLens-price-tracker.git

echo.
echo Choose authentication method:
echo   [1] Standard GitHub Login (Browser / Git Credential Manager) - Press ENTER
echo   [2] Push using GitHub Personal Access Token (PAT)
echo.
set /p AUTH_CHOICE="Select option (1 or 2, default is 1): "

if "%AUTH_CHOICE%"=="2" (
    echo.
    set /p GITHUB_PAT="Enter your GitHub Personal Access Token (ghp_...): "
    if "!GITHUB_PAT!"=="" (
        echo Error: No token provided. Exiting.
        pause
        exit /b 1
    )
    echo Pushing with Token...
    git push -u https://!GITHUB_PAT!@github.com/Harshithagarapati19/MandiLens-price-tracker.git main
) else (
    echo.
    echo Pushing with Git Credential Manager...
    echo (A browser window or sign-in prompt will appear if not already logged in)
    git push -u origin main
)

echo.
if %ERRORLEVEL% equ 0 (
    echo ======================================================================
    echo   SUCCESS! Project successfully pushed to GitHub!
    echo   Repository URL: https://github.com/Harshithagarapati19/MandiLens-price-tracker
    echo ======================================================================
    echo.
    echo Next Step - Deploy to Streamlit Cloud:
    echo 1. Open https://share.streamlit.io
    echo 2. Click 'New app'
    echo 3. Select repository: Harshithagarapati19/MandiLens-price-tracker
    echo 4. Set main file path: app.py
    echo 5. Click 'Deploy!'
) else (
    echo ======================================================================
    echo   Push encountered an error.
    echo   If you do not have Git Credential Manager, generate a token at:
    echo   https://github.com/settings/tokens (select 'repo' scope)
    echo   and rerun this script choosing option 2.
    echo ======================================================================
)

echo.
pause
