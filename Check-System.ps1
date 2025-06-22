# Jarvis EDU System Checker
# ========================
# Sprawdza dostępność wszystkich komponentów systemowych

Write-Host "Jarvis EDU System Checker" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

$systemStatus = @{
    Python = $false
    FFmpeg = $false
    FFprobe = $false
    LMStudio = $false
    ProjectFiles = $false
}

$issues = @()

# Sprawdź Python
Write-Host "Sprawdzanie Python..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Python zainstalowany: $pythonVersion" -ForegroundColor Green
        $systemStatus.Python = $true
    } else {
        throw "Python nie znaleziony w PATH"
    }
} catch {
    try {
        $pyVersion = & py --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Python zainstalowany (py launcher): $pyVersion" -ForegroundColor Green
            $systemStatus.Python = $true
        } else {
            throw "Py launcher nie znaleziony"
        }
    } catch {
        Write-Host "[ERROR] Python nie zainstalowany lub nie dodany do PATH" -ForegroundColor Red
        $issues += "Zainstaluj Python z python.org lub Microsoft Store"
        $systemStatus.Python = $false
    }
}

# Sprawdź FFmpeg
Write-Host ""
Write-Host "🎬 Sprawdzanie FFmpeg..." -ForegroundColor Yellow

# Lista możliwych lokalizacji FFmpeg
$ffmpegPaths = @(
    "C:\ffmpeg\bin\ffmpeg.exe",
    "C:\ffmpeg\ffmpeg.exe",
    "C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    "C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    "C:\tools\ffmpeg\bin\ffmpeg.exe",
    "$env:USERPROFILE\ffmpeg\bin\ffmpeg.exe",
    "$env:USERPROFILE\Desktop\ffmpeg\bin\ffmpeg.exe",
    "$env:USERPROFILE\Downloads\ffmpeg\bin\ffmpeg.exe"
)

$ffmpegFound = $false
$ffmpegPath = ""

# Najpierw sprawdź PATH
try {
    $ffmpegVersion = & ffmpeg -version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $ffmpegFound = $true
        $ffmpegPath = "PATH"
        Write-Host "✅ FFmpeg dostępny w PATH" -ForegroundColor Green
        $systemStatus.FFmpeg = $true
    }
} catch {
    # Sprawdź konkretne lokalizacje
    foreach ($path in $ffmpegPaths) {
        if (Test-Path $path) {
            $ffmpegFound = $true
            $ffmpegPath = $path
            Write-Host "✅ FFmpeg znaleziony: $path" -ForegroundColor Green
            $systemStatus.FFmpeg = $true
            break
        }
    }
}

if (-not $ffmpegFound) {
    Write-Host "❌ FFmpeg nie znaleziony" -ForegroundColor Red
    $issues += "Zainstaluj FFmpeg z https://ffmpeg.org/download.html#build-windows"
    $systemStatus.FFmpeg = $false
}

# Sprawdź FFprobe
Write-Host ""
Write-Host "🔍 Sprawdzanie FFprobe..." -ForegroundColor Yellow

$ffprobePaths = $ffmpegPaths -replace "ffmpeg.exe", "ffprobe.exe"
$ffprobeFound = $false

try {
    $ffprobeVersion = & ffprobe -version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $ffprobeFound = $true
        Write-Host "✅ FFprobe dostępny w PATH" -ForegroundColor Green
        $systemStatus.FFprobe = $true
    }
} catch {
    foreach ($path in $ffprobePaths) {
        if (Test-Path $path) {
            $ffprobeFound = $true
            Write-Host "✅ FFprobe znaleziony: $path" -ForegroundColor Green
            $systemStatus.FFprobe = $true
            break
        }
    }
}

if (-not $ffprobeFound) {
    Write-Host "❌ FFprobe nie znaleziony" -ForegroundColor Red
    $issues += "FFprobe powinien być w tym samym folderze co FFmpeg"
    $systemStatus.FFprobe = $false
}

# Sprawdź LM Studio
Write-Host ""
Write-Host "🤖 Sprawdzanie LM Studio..." -ForegroundColor Yellow

try {
    $lmResponse = Invoke-WebRequest -Uri "http://localhost:1234/v1/models" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($lmResponse.StatusCode -eq 200) {
        $models = ($lmResponse.Content | ConvertFrom-Json).data
        Write-Host "✅ LM Studio działa na localhost:1234" -ForegroundColor Green
        Write-Host "   Załadowanych modeli: $($models.Count)" -ForegroundColor Blue
        
        if ($models.Count -gt 0) {
            Write-Host "   Modele:" -ForegroundColor Blue
            foreach ($model in $models[0..2]) {  # Pokaż pierwsze 3 modele
                Write-Host "   - $($model.id)" -ForegroundColor Blue
            }
            if ($models.Count -gt 3) {
                Write-Host "   - ... i $($models.Count - 3) więcej" -ForegroundColor Blue
            }
        } else {
            Write-Host "   ⚠️  Brak załadowanych modeli" -ForegroundColor Yellow
            $issues += "Załaduj model w LM Studio (My Models -> Load)"
        }
        $systemStatus.LMStudio = $true
    }
} catch {
    Write-Host "❌ LM Studio nie odpowiada lub nie działa" -ForegroundColor Red
    $issues += "Uruchom LM Studio i załaduj model, następnie uruchom Local Server"
    $systemStatus.LMStudio = $false
}

# Sprawdź pliki projektu
Write-Host ""
Write-Host "📁 Sprawdzanie plików projektu..." -ForegroundColor Yellow

$projectFiles = @(
    "jarvis_edu/__init__.py",
    "jarvis_edu/extractors/enhanced_video_extractor.py",
    "jarvis_edu/processors/lm_studio.py",
    "config/auto-mode.yaml",
    "start_jarvis.py"
)

$allFilesExist = $true
foreach ($file in $projectFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file - brak pliku" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if ($allFilesExist) {
    $systemStatus.ProjectFiles = $true
} else {
    $issues += "Niektóre pliki projektu są brakujące - sprawdź instalację"
    $systemStatus.ProjectFiles = $false
}

# Sprawdź katalogi
Write-Host ""
Write-Host "📂 Sprawdzanie katalogów..." -ForegroundColor Yellow

$directories = @{
    "Input" = "C:\Users\barto\OneDrive\Pulpit\mediapobrane\pobrane_media"
    "Output" = "C:\Users\barto\JarvisEDU\output"
    "Backup" = "C:\Users\barto\JarvisEDU\backup"
    "Logs" = "logs"
}

foreach ($dir in $directories.GetEnumerator()) {
    if (Test-Path $dir.Value) {
        Write-Host "✅ $($dir.Key): $($dir.Value)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  $($dir.Key): $($dir.Value) - katalog nie istnieje" -ForegroundColor Yellow
        try {
            New-Item -ItemType Directory -Path $dir.Value -Force | Out-Null
            Write-Host "   📁 Utworzono katalog" -ForegroundColor Blue
        } catch {
            Write-Host "   ❌ Nie można utworzyć katalogu" -ForegroundColor Red
        }
    }
}

# Podsumowanie
Write-Host ""
Write-Host "📊 PODSUMOWANIE SYSTEMU" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

$componentsOK = ($systemStatus.Values | Where-Object { $_ -eq $true }).Count
$totalComponents = $systemStatus.Count

Write-Host "Składniki systemowe: $componentsOK/$totalComponents OK" -ForegroundColor $(if ($componentsOK -eq $totalComponents) { "Green" } else { "Yellow" })
Write-Host ""

foreach ($component in $systemStatus.GetEnumerator()) {
    $status = if ($component.Value) { "[OK]" } else { "[ERROR]" }
    $color = if ($component.Value) { "Green" } else { "Red" }
    Write-Host "$($component.Key): $status" -ForegroundColor $color
}

# Instrukcje rozwiązania problemów
if ($issues.Count -gt 0) {
    Write-Host ""
    Write-Host "🔧 WYMAGANE DZIAŁANIA:" -ForegroundColor Red
    Write-Host "=====================" -ForegroundColor Red
    
    for ($i = 0; $i -lt $issues.Count; $i++) {
        Write-Host "$($i + 1). $($issues[$i])" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "📖 Szczegółowe instrukcje znajdziesz w SETUP_GUIDE.md" -ForegroundColor Blue
} else {
    Write-Host ""
    Write-Host "🎉 SYSTEM GOTOWY DO PRACY!" -ForegroundColor Green
    Write-Host "=========================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Następne kroki:" -ForegroundColor Blue
    Write-Host "1. Uruchom: python start_jarvis.py" -ForegroundColor Blue
    Write-Host "2. Wrzuć pliki do folderu obserwowanego" -ForegroundColor Blue
    Write-Host "3. Sprawdź wyniki w folderze wyjściowym" -ForegroundColor Blue
}

Write-Host ""
Write-Host "⏰ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray 