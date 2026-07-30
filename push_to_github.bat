@echo off
REM Script to push CollegeBot to GitHub (Windows)
REM Run this after making changes

echo CollegeBot - Push to GitHub
echo ============================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
    git branch -M main
)

REM Check if remote exists
git remote | findstr "origin" >nul
if errorlevel 1 (
    echo Adding remote origin...
    git remote add origin https://github.com/Techy-A/collegebot-rag.git
)

echo Staging all changes...
git add .

echo.
set /p commit_msg="Enter commit message (or press Enter for default): "

if "%commit_msg%"=="" (
    set commit_msg=Update CollegeBot RAG system
)

echo Committing changes...
git commit -m "%commit_msg%"

echo Pushing to GitHub...
git push -u origin main

echo.
echo Done! Your changes are now on GitHub.
echo View at: https://github.com/Techy-A/collegebot-rag
pause
