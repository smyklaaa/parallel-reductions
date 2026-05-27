$outputFile = "results/sequential_results.csv"
$exePath = "build/parallel_reductions.exe"

if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

$operations = @("sum", "min", "max", "scan")
$types = @("int64", "float", "double")
$sizes = @(1000000, 5000000, 10000000)

foreach ($operation in $operations) {
    foreach ($type in $types) {
        foreach ($size in $sizes) {
            Write-Host "Running sequential $operation $type size=$size"

            & $exePath `
                --backend sequential `
                --operation $operation `
                --type $type `
                --size $size `
                --output $outputFile
        }
    }
}

Write-Host "Sequential benchmark finished."
Write-Host "Results saved to $outputFile"