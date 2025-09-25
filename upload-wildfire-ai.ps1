# Variables
$RepoName    = "wildfire-ai"
$GitHubUser  = "markl02us"
$RepoURL     = "https://github.com/$GitHubUser/$RepoName.git"
$LocalPath   = "C:\Users\markl\WildfireApplication"

# Navigate to project folder
Set-Location $LocalPath

# Create .gitignore if missing
if (!(Test-Path "$LocalPath\.gitignore")) {
    Set-Content -Path "$LocalPath\.gitignore" -Value @"
__pycache__/
*.pyc
*.engine
*.onnx
*.hef
*.pt
.env
.vscode/
.idea/
"@
}

# Initialize Git repo if not already
if (!(Test-Path "$LocalPath\.git")) {
    git init
    git branch -M main
    git remote add origin $RepoURL
}

# Stage and commit files
git add .
git commit -m "Initial commit - upload WildfireApplication code"

# Push to GitHub
git push -u origin main
