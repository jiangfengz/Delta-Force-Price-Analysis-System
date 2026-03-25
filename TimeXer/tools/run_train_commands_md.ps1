param(
  [Parameter(Mandatory = $true)][string]$MdPath,
  [int]$StartAt = 1
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $MdPath)) {
  throw ('MdPath not found: ' + $MdPath)
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $MdPath)
Set-Location -LiteralPath $repoRoot

$raw = Get-Content -LiteralPath $MdPath -Raw -Encoding UTF8
$m = [regex]::Match($raw, '```powershell\s*(?<code>[\s\S]*?)\s*```')
if (-not $m.Success) {
  throw ('No markdown powershell code block found in: ' + $MdPath)
}

$code = $m.Groups["code"].Value
$lines = $code -split "`r?`n"

$cmdStarts = New-Object System.Collections.Generic.List[int]
for ($i = 0; $i -lt $lines.Count; $i++) {
  if ($lines[$i].TrimStart().StartsWith("python run.py")) {
    $cmdStarts.Add($i)
  }
}

if ($cmdStarts.Count -eq 0) {
  throw ('No training commands found (python run.py ...) in: ' + $MdPath)
}

$startIndex = [Math]::Max(1, $StartAt)
if ($startIndex -gt $cmdStarts.Count) {
  throw ('StartAt out of range. StartAt=' + $StartAt + ', commands=' + $cmdStarts.Count)
}

$startLine = $cmdStarts[$startIndex - 1]
$exec = ($lines[$startLine..($lines.Count - 1)] -join "`n").Trim()

$tmp = Join-Path $env:TEMP ("train18_" + [guid]::NewGuid().ToString("N") + ".ps1")
Set-Content -LiteralPath $tmp -Value $exec -Encoding UTF8

& $tmp

