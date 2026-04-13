param([string]$Action = "start")
$ProjectDir = "C:\Users\pipe_render\stigmergic-reasoning"
$PidFile = Join-Path $ProjectDir "logs\experiment.pid"
$LogFile = Join-Path $ProjectDir "logs\experiment_progress.log"

switch ($Action) {
    "start" {
        if (Test-Path $PidFile) {
            $jobPid = [int](Get-Content $PidFile)
            if (Get-Process -Id $jobPid -ErrorAction SilentlyContinue) {
                Write-Host "Already running (PID: $jobPid). Use 'status' to check."
                return
            }
            Remove-Item $PidFile
        }
        New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "logs") | Out-Null
        $env:PYTHONIOENCODING = "utf-8"
        $env:PYTHONUNBUFFERED = "1"
        $proc = Start-Process -FilePath "python" `
            -ArgumentList "experiments/bottleneck_extended.py" `
            -WorkingDirectory $ProjectDir `
            -WindowStyle Hidden `
            -PassThru
        $proc.Id | Out-File $PidFile -Encoding ascii
        Write-Host "Started (PID: $($proc.Id))"
    }
    "status" {
        if (Test-Path $PidFile) {
            $jobPid = [int](Get-Content $PidFile)
            $proc = Get-Process -Id $jobPid -ErrorAction SilentlyContinue
            if ($proc) {
                $runtime = (Get-Date) - $proc.StartTime
                Write-Host "Running (PID: $jobPid, runtime: $($runtime.ToString('hh\:mm\:ss')))"
            } else {
                Write-Host "Completed (process exited)."
                Remove-Item $PidFile
            }
        } else {
            Write-Host "No job running."
        }
    }
    "log" {
        if (Test-Path $LogFile) {
            Get-Content $LogFile -Tail 20 -Wait
        } else {
            Write-Host "Log file not found: $LogFile"
        }
    }
    "stop" {
        if (Test-Path $PidFile) {
            $jobPid = [int](Get-Content $PidFile)
            Stop-Process -Id $jobPid -Force -ErrorAction SilentlyContinue
            Remove-Item $PidFile
            Write-Host "Stopped (PID: $jobPid)."
        } else {
            Write-Host "No job running."
        }
    }
}
