#!/usr/bin/env bash

set +e

OUTPUT_FILE="results/mpi_raw_results.csv"
EXE_PATH="./build/parallel_reductions"

rm -f "$OUTPUT_FILE"

operations=("sum" "min" "max" "scan")
types=("int64" "float" "double")
sizes=(1000000 5000000 10000000)
processes=(1 2 4 6)
repeats=(1 2 3 4 5)

for operation in "${operations[@]}"; do
    for type in "${types[@]}"; do
        for size in "${sizes[@]}"; do
            for process_count in "${processes[@]}"; do
                for repeat in "${repeats[@]}"; do
                    echo "Running MPI operation=$operation type=$type size=$size processes=$process_count repeat=$repeat"

                    mpirun -np "$process_count" "$EXE_PATH" \
                        --backend mpi \
                        --operation "$operation" \
                        --type "$type" \
                        --size "$size" \
                        --processes "$process_count" \
                        --verify true \
                        --output "$OUTPUT_FILE"
                done
            done
        done
    done
done

echo "MPI raw benchmark finished."
echo "Raw results saved to $OUTPUT_FILE"