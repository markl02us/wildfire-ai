# Variables
$RepoName    = "wildfire-ai"
$GitHubUser  = "markl02us"
$RepoURL     = "https://github.com/$GitHubUser/$RepoName.git"
$LocalPath   = "$HOME\Documents\$RepoName"

# Create project folder
if (!(Test-Path $LocalPath)) {
    New-Item -ItemType Directory -Path $LocalPath -Force | Out-Null
}
Set-Location $LocalPath

# Create folder structure
$folders = @(
    "training",
    "exports",
    "runtime/common",
    "runtime/jetson",
    "runtime/rpi",
    "docs"
)
foreach ($f in $folders) {
    New-Item -ItemType Directory -Path (Join-Path $LocalPath $f) -Force | Out-Null
}

# Create baseline files
Set-Content -Path "$LocalPath\.gitignore" -Value @"
__pycache__/
*.pyc
*.engine
*.onnx
*.hef
*.pt
.env
"@

Set-Content -Path "$LocalPath\requirements.txt" -Value @"
numpy
opencv-python
ultralytics
onnx
onnxruntime
pycuda
"@

Set-Content -Path "$LocalPath\README.md" -Value @"
# Wildfire AI

Edge-deployable wildfire smoke/fire detection system for Raspberry Pi 5 + AI HAT and Jetson Orin Nano.

## Structure
- \`training/\` → scripts for model training
- \`exports/\` → exported model formats (.onnx, .engine, .hef)
- \`runtime/\` → inference pipelines (common utils, Jetson TensorRT, Pi HailoRT)
- \`docs/\` → setup guides

## Usage
1. Train YOLO model on workstation
2. Export to ONNX → TensorRT (.engine) for Jetson, HEF (.hef) for Pi + Hailo
3. Deploy runtime code to edge devices
"@

Set-Content -Path "$LocalPath\runtime\common\utils.py" -Value @"
import cv2

def draw_boxes(frame, detections, class_names):
    for x1,y1,x2,y2,conf,cls in detections:
        label = f\"{class_names[int(cls)]} {conf:.2f}\"
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return frame
"@

Set-Content -Path "$LocalPath\runtime\jetson\infer_trt.py" -Value @"
# TensorRT inference runner for Jetson Orin Nano
# Placeholder - insert TRTDetector logic here
def run():
    print(\"[Jetson] Running TensorRT inference...\")
"@

Set-Content -Path "$LocalPath\runtime\rpi\infer_hailo.py" -Value @"
# HailoRT inference runner for Raspberry Pi 5 + AI HAT
# Placeholder - insert HailoRT runtime logic here
def run():
    print(\"[Raspberry Pi] Running Hailo inference...\")
"@

Set-Content -Path "$LocalPath\runtime\demo.py" -Value @"
import platform

def main():
    if 'tegra' in platform.uname().release:
        from runtime.jetson.infer_trt import run
    elif 'raspberrypi' in platform.uname().node:
        from runtime.rpi.infer_hailo import run
    else:
        print(\"Unknown platform - defaulting to mock run\")
        def run(): print(\"[Mock] Running CPU/ONNX inference...\")
    run()

if __name__ == \"__main__\":
    main()
"@

# Initialize Git and push to GitHub
if (!(Test-Path "$LocalPath\.git")) {
    git init
    git branch -M main
    git remote add origin $RepoURL
}
git add .
git commit -m "Initial commit - wildfire-ai repo structure"
git push -u origin main
