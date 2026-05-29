#!/usr/bin/env bash

set -e

export OMP_PROC_BIND=true
export OMP_PLACES=cores

OUTPUT_FILE="results/openmp_raw_results.csv"
EXE_PATH="./build/parallel_reductions"

rm -f "$OUTPUT_FILE"

operations=("sum" "min" "max" "scan")
types=("int64" "float" "double")
sizes=(1000000 5000000 10000000)
threads=(1 2 4 6 8 12)
repeats=(1 2 3 4 5)

for operation in "${operations[@]}"; do
    for type in "${types[@]}"; do
        for size in "${sizes[@]}"; do
            for thread_count in "${threads[@]}"; do
                for repeat in "${repeats[@]}"; do
                    echo "Running OpenMP operation=$operation type=$type size=$size threads=$thread_count repeat=$repeat"

                    "$EXE_PATH" \
                        --backend openmp \
                        --operation "$operation" \
                        --type "$type" \
                        --size "$size" \
                        --threads "$thread_count" \
                        --verify true \
                        --output "$OUTPUT_FILE"
                done
            done
        done
    done
done

echo "OpenMP raw benchmark finished."
echo "Raw results saved to $OUTPUT_FILE"