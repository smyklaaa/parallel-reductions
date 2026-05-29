import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


INPUT_FILE = "results/openmp_results.csv"
OUTPUT_DIR = "plots/openmp_plots"


def read_results(path):
    results = []

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            results.append({
                "backend": row["backend"],
                "operation": row["operation"],
                "data_type": row["data_type"],
                "size": int(row["size"]),
                "threads": int(row["threads"]),
                "processes": int(row["processes"]),
                "block_size": int(row["block_size"]),
                "time_ms": float(row["time_ms"]),
                "throughput_gbs": float(row["throughput_gbs"]),
                "absolute_error": float(row["absolute_error"]),
                "relative_error": float(row["relative_error"]),
                "result_summary": row["result_summary"],
            })

    return results


def group_by_operation_type_threads(results):
    grouped = defaultdict(list)

    for row in results:
        key = (row["operation"], row["data_type"], row["threads"])
        grouped[key].append(row)

    for key in grouped:
        grouped[key].sort(key=lambda item: item["size"])

    return grouped


def group_by_operation_type_size(results):
    grouped = defaultdict(list)

    for row in results:
        key = (row["operation"], row["data_type"], row["size"])
        grouped[key].append(row)

    for key in grouped:
        grouped[key].sort(key=lambda item: item["threads"])

    return grouped


def plot_throughput(grouped):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    operation_type_pairs = sorted(
        set((operation, data_type) for operation, data_type, _ in grouped.keys())
    )

    for operation, data_type in operation_type_pairs:
        plt.figure()

        for (current_operation, current_data_type, threads), rows in sorted(grouped.items()):
            if current_operation != operation or current_data_type != data_type:
                continue

            sizes = [row["size"] for row in rows]
            throughput = [row["throughput_gbs"] for row in rows]

            plt.plot(sizes, throughput, marker="o", label=f"{threads} wątków")

        plt.xlabel("Rozmiar tablicy")
        plt.ylabel("Przepustowość [GB/s]")
        plt.title(f"OpenMP przepustowość: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"openmp_throughput_{operation}_{data_type}.png"
        )
        plt.savefig(output_path)
        plt.close()


def plot_speedup(grouped):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    operation_type_pairs = sorted(
        set((operation, data_type) for operation, data_type, _ in grouped.keys())
    )

    for operation, data_type in operation_type_pairs:
        plt.figure()

        for (current_operation, current_data_type, size), rows in sorted(grouped.items()):
            if current_operation != operation or current_data_type != data_type:
                continue

            baseline = [row for row in rows if row["threads"] == 1]

            if not baseline:
                continue

            baseline_time = baseline[0]["time_ms"]

            threads = [row["threads"] for row in rows]
            speedup = [baseline_time / row["time_ms"] for row in rows]

            plt.plot(threads, speedup, marker="o", label=f"size={size}")

        max_threads = max(
            row["threads"]
            for (current_operation, current_data_type, _), rows in grouped.items()
            if current_operation == operation and current_data_type == data_type
            for row in rows
        )

        ideal_threads = sorted(set(
            row["threads"]
            for (current_operation, current_data_type, _), rows in grouped.items()
            if current_operation == operation and current_data_type == data_type
            for row in rows
        ))

        plt.plot(ideal_threads, ideal_threads, linestyle="--", label="Idealny speedup")

        plt.xlabel("Liczba wątków")
        plt.ylabel("Przyspieszenie")
        plt.title(f"OpenMP speedup: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"openmp_speedup_{operation}_{data_type}_all_sizes.png"
        )
        plt.savefig(output_path)
        plt.close()


def plot_efficiency(grouped):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    operation_type_pairs = sorted(
        set((operation, data_type) for operation, data_type, _ in grouped.keys())
    )

    for operation, data_type in operation_type_pairs:
        plt.figure()

        for (current_operation, current_data_type, size), rows in sorted(grouped.items()):
            if current_operation != operation or current_data_type != data_type:
                continue

            baseline = [row for row in rows if row["threads"] == 1]

            if not baseline:
                continue

            baseline_time = baseline[0]["time_ms"]

            threads = [row["threads"] for row in rows]
            efficiency = [
                (baseline_time / row["time_ms"]) / row["threads"]
                for row in rows
            ]

            plt.plot(threads, efficiency, marker="o", label=f"size={size}")

        plt.axhline(y=1.0, linestyle="--", label="Idealna efektywność")

        plt.xlabel("Liczba wątków")
        plt.ylabel("Efektywność")
        plt.title(f"OpenMP efektywność: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"openmp_efficiency_{operation}_{data_type}_all_sizes.png"
        )
        plt.savefig(output_path)
        plt.close()

def plot_time(grouped):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    operation_type_pairs = sorted(
        set((operation, data_type) for operation, data_type, _ in grouped.keys())
    )

    for operation, data_type in operation_type_pairs:
        plt.figure()

        for (current_operation, current_data_type, size), rows in sorted(grouped.items()):
            if current_operation != operation or current_data_type != data_type:
                continue

            threads = [row["threads"] for row in rows]
            times = [row["time_ms"] for row in rows]

            plt.plot(threads, times, marker="o", label=f"size={size}")

        plt.xlabel("Liczba wątków")
        plt.ylabel("Czas [ms]")
        plt.title(f"OpenMP czas wykonania: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"openmp_time_{operation}_{data_type}_all_sizes.png"
        )
        plt.savefig(output_path)
        plt.close()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Nie znaleziono pliku: {INPUT_FILE}")
        print("Najpierw uruchom:")
        print("./scripts/run_openmp_tests.sh")
        return

    results = read_results(INPUT_FILE)

    if not results:
        print(f"Plik {INPUT_FILE} jest pusty.")
        return

    grouped_by_threads = group_by_operation_type_threads(results)
    grouped_by_size = group_by_operation_type_size(results)

    plot_throughput(grouped_by_threads)
    plot_time(grouped_by_size)
    plot_speedup(grouped_by_size)
    plot_efficiency(grouped_by_size)

    print(f"Wykresy OpenMP zapisano w folderze: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()