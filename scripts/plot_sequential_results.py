import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


INPUT_FILE = "results/sequential_results.csv"
OUTPUT_DIR = "plots"


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
                "time_ms": float(row["time_ms"]),
                "throughput_gbs": float(row["throughput_gbs"]),
                "result_summary": row["result_summary"],
            })

    return results


def group_by_operation_and_type(results):
    grouped = defaultdict(list)

    for row in results:
        key = (row["operation"], row["data_type"])
        grouped[key].append(row)

    for key in grouped:
        grouped[key].sort(key=lambda item: item["size"])

    return grouped


def plot_time(grouped):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for (operation, data_type), rows in grouped.items():
        sizes = [row["size"] for row in rows]
        times = [row["time_ms"] for row in rows]

        plt.figure()
        plt.plot(sizes, times, marker="o")
        plt.xlabel("Rozmiar tablicy")
        plt.ylabel("Czas [ms]")
        plt.title(f"Czas wykonania: {operation}, {data_type}")
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, f"sequential_time_{operation}_{data_type}.png")
        plt.savefig(output_path)
        plt.close()


def plot_throughput(grouped):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for (operation, data_type), rows in grouped.items():
        sizes = [row["size"] for row in rows]
        throughput = [row["throughput_gbs"] for row in rows]

        plt.figure()
        plt.plot(sizes, throughput, marker="o")
        plt.xlabel("Rozmiar tablicy")
        plt.ylabel("Przepustowość [GB/s]")
        plt.title(f"Przepustowość: {operation}, {data_type}")
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, f"sequential_throughput_{operation}_{data_type}.png")
        plt.savefig(output_path)
        plt.close()


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Nie znaleziono pliku: {INPUT_FILE}")
        print("Najpierw uruchom skrypt benchmarkowy:")
        print(".\\scripts\\run_sequential_tests.ps1")
        return

    results = read_results(INPUT_FILE)
    grouped = group_by_operation_and_type(results)

    plot_time(grouped)
    plot_throughput(grouped)

    print(f"Wykresy zapisano w folderze: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()