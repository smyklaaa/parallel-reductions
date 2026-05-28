$outputFile = "results/mpi_raw_results.csv"
$exePath = "./build/parallel_reductions"

if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

$operations = @("sum", "min", "max", "scan")
$types = @("int64", "float", "double")
$sizes = @(1000000, 5000000, 10000000)
$processes = @(1, 2, 4, 6)
$repeats = @(1, 2, 3, 4, 5)

foreach ($operation in $operations) {
    foreach ($type in $types) {
        foreach ($size in $sizes) {
            foreach ($processCount in $processes) {
                foreach ($repeat in $repeats) {
                    Write-Host "Running MPI operation=$operation type=$type size=$size processes=$processCount repeat=$repeat"

                    & mpirun -np $processCount $exePath `
                        --backend mpi `
                        --operation $operation `
                        --type $type `
                        --size $size `
                        --processes $processCount `
                        --verify true `
                        --output $outputFile

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "FAILED: operation=$operation type=$type size=$size processes=$processCount repeat=$repeat exitCode=$LASTEXITCODE"
                    }
                }
            }
        }
    }
}

Write-Host "MPI raw benchmark finished."
Write-Host "Raw results saved to $outputFile"