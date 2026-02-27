# Publishing iDILI-Predict to GitHub

Step-by-step instructions to get this repository on GitHub.

## Prerequisites

### 1. Check if git is installed

```bash
git --version
```

If not installed, see https://git-scm.com/downloads

### 2. Create a GitHub account

If you don't have one, go to https://github.com/join and create a free account.

### 3. Set up GitHub authentication

The easiest way is GitHub CLI:

```bash
# Install GitHub CLI (on Ubuntu/WSL):
sudo apt install gh

# Or on Mac:
brew install gh

# Authenticate:
gh auth login
```

Follow the prompts — choose "GitHub.com", "HTTPS", and "Login with a web browser".

## Push to the Repository

Since the repository already exists at https://github.com/SextonLab/iDILI-Predict, connect and push:

### Step 1: Initialize git and push

```bash
cd /mnt/x/Active_Users_Data/Jonny/DILI_PatientScreen/Production/iDILI-Predict

# Initialize repo
git init
git add -A
git commit -m "Initial commit: iDILI-Predict pipeline"

# Connect to GitHub
git remote add origin https://github.com/SextonLab/iDILI-Predict.git

# Rename branch to 'main' if needed
git branch -M main

# Push
git push -u origin main
```

### Step 2: Verify

```bash
gh repo view --web
```

Or go to https://github.com/SextonLab/iDILI-Predict in your browser.

## Optional: Polish the GitHub Page

### Add topics (tags)

On your repo page, click the gear icon next to "About" and add topics:

```
dili, drug-induced-liver-injury, cell-painting, high-content-screening,
morphological-profiling, automl, autogluon, bioinformatics
```

### Add a description

Same gear icon — set the description to:
> Morphological profiling for drug-induced liver injury prediction using Cell Painting and AutoML

### Pin the repository

Go to your GitHub profile, click "Customize your pins", and add iDILI-Predict.

## Updating the Repo Later

After making local changes:

```bash
cd /mnt/x/Active_Users_Data/Jonny/DILI_PatientScreen/Production/iDILI-Predict

# Check what changed
git status

# Stage changes
git add -A

# Commit
git commit -m "description of what changed"

# Push to GitHub
git push
```

## Linking to Your Paper

In your manuscript, reference the repo as:

> Code and example data are available at https://github.com/SextonLab/iDILI-Predict
