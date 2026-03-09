#!/bin/bash
# One-time script to push to GitHub. DELETE THIS FILE after use.
# Your PAT was exposed - REVOKE IT at https://github.com/settings/tokens after pushing.

set -e
cd "$(dirname "$0")"

# Use PAT for authentication (replace with your token - or use GitHub CLI: gh auth login)
git remote remove origin 2>/dev/null || true
git remote add origin "https://Sid245439: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@github.com/Sid245439/cats_dogs_classification_mlops.git"

git add .
git status
git commit -m "Add Cats vs Dogs MLOps pipeline" || true
git branch -M main
git push -u origin main

echo "Done! Now REVOKE your PAT at https://github.com/settings/tokens and delete this script."
