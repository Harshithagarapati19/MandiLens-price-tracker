@echo off
title IntelliMandi - Streamlit Dashboard
echo ========================================================
echo   Starting IntelliMandi Agricultural Intelligence Server
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
