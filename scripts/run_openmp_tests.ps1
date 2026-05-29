$env:OMP_PROC_BIND = "true"
$env:OMP_PLACES = "cores"

$outputFile = "results/openmp_raw_results.csv"
$exePath = "./build/parallel_reductions"

if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

$operations = @("sum", "min", "max", "scan")
$types = @("int64", "float", "double")
$sizes = @(1000000, 5000000, 10000000)
$threads = @(1, 2, 4, 6, 8, 12)
$repeats = @(1, 2, 3, 4, 5)

foreach ($operation in $operations) {
    foreach ($type in $types) {
        foreach ($size in $sizes) {
            foreach ($threadCount in $threads) {
                foreach ($repeat in $repeats) {
                    Write-Host "Running OpenMP operation=$operation type=$type size=$size threads=$threadCount repeat=$repeat"

                    & $exePath `
                        --backend openmp `
                        --operation $operation `
                        --type $type `
                        --size $size `
                        --threads $threadCount `
                        --verify true `
                        --output $outputFile

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "FAILED: operation=$operation type=$type size=$size threads=$threadCount repeat=$repeat exitCode=$LASTEXITCODE"
                    }
                }
            }
        }
    }
}

Write-Host "OpenMP raw benchmark finished."
Write-Host "Raw results saved to $outputFile"