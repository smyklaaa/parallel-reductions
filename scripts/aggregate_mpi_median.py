import csv
import os
from collections import defaultdict
from statistics import median


INPUT_FILE = "results/mpi_raw_results.csv"
OUTPUT_FILE = "results/mpi_results.csv"


GROUP_COLUMNS = [
    "backend",
    "operation",
    "data_type",
    "size",
    "threads",
    "processes",
    "block_size",
]


NUMERIC_MEDIAN_COLUMNS = [
    "time_ms",
    "throughput_gbs",
    "absolute_error",
    "relative_error",
]


OUTPUT_COLUMNS = [
    "backend",
    "operation",
    "data_type",
    "size",
    "threads",
    "processes",
    "block_size",
    "time_ms",
    "throughput_gbs",
    "absolute_error",
    "relative_error",
    "result_summary",
]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def group_rows(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        grouped[key].append(row)

    return grouped


def median_row(rows):
    result = {}

    first = rows[0]

    for column in GROUP_COLUMNS:
        result[column] = first[column]

    for column in NUMERIC_MEDIAN_COLUMNS:
        values = [float(row[column]) for row in rows]
        result[column] = f"{median(values):.10g}"

    median_time = float(result["time_ms"])
    representative = min(rows, key=lambda row: abs(float(row["time_ms"]) - median_time))
    result["result_summary"] = representative["result_summary"]

    return result


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Nie znaleziono pliku: {INPUT_FILE}")
        print("Najpierw uruchom:")
        print("./scripts/run_mpi_tests.sh")
        return

    rows = read_rows(INPUT_FILE)

    if not rows:
        print(f"Plik {INPUT_FILE} jest pusty.")
        return

    grouped = group_rows(rows)

    aggregated = [
        median_row(group)
        for _, group in sorted(grouped.items())
    ]

    write_rows(OUTPUT_FILE, aggregated)

    print(f"Zapisano mediany do: {OUTPUT_FILE}")
    print(f"Liczba konfiguracji: {len(aggregated)}")


if __name__ == "__main__":
    main()