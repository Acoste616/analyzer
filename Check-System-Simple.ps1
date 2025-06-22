# Jarvis EDU System Checker (Simple Version)
# ===========================================

Write-Host "Jarvis EDU System Checker" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

$errors = @()

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
$pythonOK = $false
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Python found: $pythonVersion" -ForegroundColor Green
        $pythonOK = $true
    }
} catch {
    try {
        $pyVersion = py --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] Python found via py launcher: $pyVersion" -ForegroundColor Green
            $pythonOK = $true
        }
    } catch {
        Write-Host "  [ERROR] Python not found" -ForegroundColor Red
        $errors += "Install Python from python.org or Microsoft Store"
    }
}

# Check FFmpeg
Write-Host ""
Write-Host "Checking FFmpeg..." -ForegroundColor Yellow
$ffmpegOK = $false
try {
    $ffmpegVersion = ffmpeg -version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] FFmpeg found in PATH" -ForegroundColor Green
        $ffmpegOK = $true
    }
} catch {
    # Check common locations
    $ffmpegPaths = @(
        "C:\ffmpeg\bin\ffmpeg.exe",
        "C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "$env:USERPROFILE\ffmpeg\bin\ffmpeg.exe"
    )
    
    foreach ($path in $ffmpegPaths) {
        if (Test-Path $path) {
            Write-Host "  [OK] FFmpeg found at: $path" -ForegroundColor Green
            $ffmpegOK = $true
            break
        }
    }
    
    if (-not $ffmpegOK) {
        Write-Host "  [ERROR] FFmpeg not found" -ForegroundColor Red
        $errors += "Install FFmpeg from https://ffmpeg.org/download.html"
    }
}

# Check LM Studio
Write-Host ""
Write-Host "Checking LM Studio..." -ForegroundColor Yellow
$lmStudioOK = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:1234/v1/models" -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        $data = $response.Content | ConvertFrom-Json
        $modelCount = $data.data.Count
        Write-Host "  [OK] LM Studio running with $modelCount models" -ForegroundColor Green
        $lmStudioOK = $true
        
        if ($modelCount -eq 0) {
            Write-Host "  [WARNING] No models loaded" -ForegroundColor Yellow
            $errors += "Load a model in LM Studio (My Models -> Load)"
        }
    }
} catch {
    Write-Host "  [ERROR] LM Studio not responding" -ForegroundColor Red
    $errors += "Start LM Studio and load a model, then start Local Server"
}

# Check project files
Write-Host ""
Write-Host "Checking project files..." -ForegroundColor Yellow
$projectOK = $true
$requiredFiles = @(
    "jarvis_edu\__init__.py",
    "jarvis_edu\extractors\enhanced_video_extractor.py",
    "jarvis_edu\processors\lm_studio.py",
    "start_jarvis.py"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Missing: $file" -ForegroundColor Red
        $projectOK = $false
    }
}

if (-not $projectOK) {
    $errors += "Some project files are missing - check installation"
}

# Check directories
Write-Host ""
Write-Host "Checking directories..." -ForegroundColor Yellow
$directories = @{
    "Input" = "C:\Users\barto\OneDrive\Pulpit\mediapobrane\pobrane_media"
    "Output" = "C:\Users\barto\JarvisEDU\output"
    "Backup" = "C:\Users\barto\JarvisEDU\backup"
}

foreach ($dir in $directories.GetEnumerator()) {
    if (Test-Path $dir.Value) {
        Write-Host "  [OK] $($dir.Key): $($dir.Value)" -ForegroundColor Green
    } else {
        Write-Host "  [INFO] Creating: $($dir.Value)" -ForegroundColor Yellow
        try {
            New-Item -ItemType Directory -Path $dir.Value -Force | Out-Null
            Write-Host "  [OK] Directory created" -ForegroundColor Green
        } catch {
            Write-Host "  [ERROR] Cannot create directory" -ForegroundColor Red
        }
    }
}

# Summary
Write-Host ""
Write-Host "SYSTEM STATUS SUMMARY" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host ""

$components = @{
    "Python" = $pythonOK
    "FFmpeg" = $ffmpegOK
    "LM Studio" = $lmStudioOK
    "Project Files" = $projectOK
}

$okCount = 0
foreach ($comp in $components.GetEnumerator()) {
    $status = if ($comp.Value) { "[OK]" } else { "[ERROR]" }
    $color = if ($comp.Value) { "Green" } else { "Red" }
    Write-Host "$($comp.Key): $status" -ForegroundColor $color
    if ($comp.Value) { $okCount++ }
}

Write-Host ""
Write-Host "Components OK: $okCount/$($components.Count)" -ForegroundColor $(if ($okCount -eq $components.Count) { "Green" } else { "Yellow" })

if ($errors.Count -gt 0) {
    Write-Host ""
    Write-Host "REQUIRED ACTIONS:" -ForegroundColor Red
    Write-Host "=================" -ForegroundColor Red
    for ($i = 0; $i -lt $errors.Count; $i++) {
        Write-Host "$($i + 1). $($errors[$i])" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "See SETUP_GUIDE.md for detailed instructions" -ForegroundColor Blue
} else {
    Write-Host ""
    Write-Host "SYSTEM READY!" -ForegroundColor Green
    Write-Host "=============" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Blue
    Write-Host "1. Run: python start_jarvis.py" -ForegroundColor Blue
    Write-Host "2. Drop files into watched folder" -ForegroundColor Blue
    Write-Host "3. Check results in output folder" -ForegroundColor Blue
}

Write-Host ""
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray 