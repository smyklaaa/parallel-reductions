import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


INPUT_FILE = "results/mpi_results.csv"
OUTPUT_DIR = "plots/mpi_plots"


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


def group_by_operation_type_processes(results):
    grouped = defaultdict(list)

    for row in results:
        key = (row["operation"], row["data_type"], row["processes"])
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
        grouped[key].sort(key=lambda item: item["processes"])

    return grouped


def plot_throughput(grouped):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    operation_type_pairs = sorted(
        set((operation, data_type) for operation, data_type, _ in grouped.keys())
    )

    for operation, data_type in operation_type_pairs:
        plt.figure()

        for (current_operation, current_data_type, processes), rows in sorted(grouped.items()):
            if current_operation != operation or current_data_type != data_type:
                continue

            sizes = [row["size"] / 1_000_000 for row in rows]
            throughput = [row["throughput_gbs"] for row in rows]

            plt.plot(sizes, throughput, marker="o", label=f"{processes} procesów")

        plt.xlabel("Rozmiar tablicy [mln elementów]")
        plt.ylabel("Przepustowość [GB/s]")
        plt.title(f"MPI przepustowość: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"mpi_throughput_{operation}_{data_type}.png"
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

            processes = [row["processes"] for row in rows]
            times = [row["time_ms"] for row in rows]

            plt.plot(processes, times, marker="o", label=f"size={size}")

        plt.xlabel("Liczba procesów")
        plt.ylabel("Czas [ms]")
        plt.title(f"MPI czas wykonania: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"mpi_time_{operation}_{data_type}_all_sizes.png"
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

            baseline = [row for row in rows if row["processes"] == 1]

            if not baseline:
                continue

            baseline_time = baseline[0]["time_ms"]

            processes = [row["processes"] for row in rows]
            speedup = [baseline_time / row["time_ms"] for row in rows]

            plt.plot(processes, speedup, marker="o", label=f"size={size}")

        all_processes = sorted(set(
            row["processes"]
            for (_, _, _), rows in grouped.items()
            for row in rows
        ))

        plt.plot(all_processes, all_processes, linestyle="--", label="Idealny speedup")

        plt.xlabel("Liczba procesów")
        plt.ylabel("Przyspieszenie")
        plt.title(f"MPI speedup: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"mpi_speedup_{operation}_{data_type}_all_sizes.png"
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

            baseline = [row for row in rows if row["processes"] == 1]

            if not baseline:
                continue

            baseline_time = baseline[0]["time_ms"]

            processes = [row["processes"] for row in rows]
            efficiency = [
                (baseline_time / row["time_ms"]) / row["processes"]
                for row in rows
            ]

            plt.plot(processes, efficiency, marker="o", label=f"size={size}")

        plt.axhline(y=1.0, linestyle="--", label="Idealna efektywność")

        plt.xlabel("Liczba procesów")
        plt.ylabel("Efektywność")
        plt.title(f"MPI efektywność: {operation}, {data_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"mpi_efficiency_{operation}_{data_type}_all_sizes.png"
        )
        plt.savefig(output_path)
        plt.close()


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Nie znaleziono pliku: {INPUT_FILE}")
        print("Najpierw uruchom:")
        print("./scripts/run_mpi_tests.sh")
        return

    results = read_results(INPUT_FILE)

    if not results:
        print(f"Plik {INPUT_FILE} jest pusty.")
        return

    grouped_by_processes = group_by_operation_type_processes(results)
    grouped_by_size = group_by_operation_type_size(results)

    plot_throughput(grouped_by_processes)
    plot_time(grouped_by_size)
    plot_speedup(grouped_by_size)
    plot_efficiency(grouped_by_size)

    print(f"Wykresy MPI zapisano w folderze: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()