$outputFile = ".\results\cuda_results.csv"
$exePath = ".\build\parallel_reductions.exe"

if (!(Test-Path $exePath)) {
    Write-Host "Executable not found: $exePath"
    Write-Host "Build the CUDA version first in x64 Native Tools Command Prompt for VS 2022."
    exit 1
}

if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

$operations = @("sum", "min", "max", "scan")
$types = @("int64", "float", "double")
$sizes = @(1000000, 5000000, 10000000)

foreach ($operation in $operations) {
    foreach ($type in $types) {
        foreach ($size in $sizes) {
            Write-Host "Running CUDA $operation $type size=$size"

            & $exePath `
                --backend cuda `
                --operation $operation `
                --type $type `
                --size $size `
                --block-size 256 `
                --output $outputFile

            if ($LASTEXITCODE -ne 0) {
                Write-Host "Test failed: backend=cuda operation=$operation type=$type size=$size"
                exit $LASTEXITCODE
            }
        }
    }
}

Write-Host "CUDA benchmark finished."
Write-Host "Results saved to $outputFile"