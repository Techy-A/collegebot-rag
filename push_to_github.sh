#!/bin/bash
# Script to push CollegeBot to GitHub
# Run this after making changes

echo "CollegeBot - Push to GitHub"
echo "============================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
fi

# Check if remote exists
if ! git remote | grep -q "origin"; then
    echo "Adding remote origin..."
    git remote add origin https://github.com/Techy-A/collegebot-rag.git
fi

echo "Staging all changes..."
git add .

echo ""
echo "Enter commit message (or press Enter for default):"
read commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Update CollegeBot RAG system"
fi

echo "Committing changes..."
git commit -m "$commit_msg"

echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "Done! Your changes are now on GitHub."
echo "View at: https://github.com/Techy-A/collegebot-rag"
