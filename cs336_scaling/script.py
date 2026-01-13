import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


"""
chinchilla_isoflops
(a)
N = 25.793 * C ^ 0.4
compute budget is 10^23, model size is 5.0e10
compute budget is 10^24, model size is 1.27e11

(b)
D = 0.006 * C ^ 0.597
compute budget is 10^23, dataset size is 3.37e11
compute budget is 10^24, dataset size is 1.33e12
"""


def main():
    file_path = '../data/isoflops_curves.json'
    with open(file_path, 'r') as file:
        runs = json.load(file)

    from collections import defaultdict
    curves = defaultdict(list)
    for run in runs:
        C = run["compute_budget"]
        N = run["parameters"]
        L = run["final_loss"]
        curves[C].append((N, L))

    # For each compute budget, find the min‑loss point
    compute_budgets = []
    optimal_Ns = []
    optimal_Ds = []

    for C, points in curves.items():
        # sort by model size for plotting
        points_sorted = sorted(points, key=lambda x: x[0])
        Ns = [p[0] for p in points_sorted]
        Ls = [p[1] for p in points_sorted]
        Ds = [C / (6 * n) for n in Ns]

        # find min loss
        best_idx = np.argmin(Ls)
        optimal_N = Ns[best_idx]
        optimal_D = Ds[best_idx]
        compute_budgets.append(C)
        optimal_Ns.append(optimal_N)
        optimal_Ds.append(optimal_D)

        # Plot each IsoFLOPs curve (loss vs model size)
        plt.figure(1)
        plt.loglog(Ns, Ls, marker='o', linestyle='-', label=f"C={C:.1e}")

    # Decorate IsoFLOPs curves
    plt.figure(1)
    plt.xlabel("Model size (N params)")
    plt.ylabel("Final Loss")
    plt.title("IsoFLOPs Curves (Loss vs Model Size)")
    plt.legend()
    plt.grid(True)

    compute_budgets = np.array(compute_budgets)
    optimal_Ns = np.array(optimal_Ns)
    optimal_Ds = np.array(optimal_Ds)
    print(compute_budgets, optimal_Ns, optimal_Ds)

    # Define the power-law function to fit
    def power_law(C, k, a):
        return k * C**a

    # Fit the curve
    for fit_data, label in [(optimal_Ns, "Model"), (optimal_Ds, "Dataset")]:
        params, _ = curve_fit(power_law, compute_budgets, fit_data)
        k_fit, a_fit = params
        print(f"Fitted parameters: k = {k_fit:.3f}, a = {a_fit:.3f}")

        for tmp_c in [1e23, 1e24]:
            tmp_n = k_fit * tmp_c ** a_fit
            print(f"compute is {tmp_c}, {label} size is {tmp_n:.3e}")

        # Generate fitted curve for plotting
        C_fit = np.logspace(np.log10(min(compute_budgets)), np.log10(max(compute_budgets)), 100)
        N_fit = power_law(C_fit, k_fit, a_fit)

        # Plot original data and fitted curve
        plt.figure(figsize=(6,5))
        plt.loglog(compute_budgets, fit_data, 'o', label='Data')
        plt.loglog(C_fit, N_fit, '-', label=f'Fit: N = {k_fit:.2e} * C^{a_fit:.2f}')
        plt.xlabel("Compute budget (FLOPs)")
        plt.ylabel(f"Optimal {label} Size (N_opt)")
        plt.title(f"Compute-Optimal {label} Size vs Compute Budget")
        plt.legend()
        plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
