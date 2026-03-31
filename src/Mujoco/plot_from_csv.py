import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "tablet_force_log.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    plt.figure()
    plt.plot(df["time_s"], df["Fx_N"], label="Fx")
    plt.plot(df["time_s"], df["Fy_N"], label="Fy")
    plt.plot(df["time_s"], df["Fz_N"], label="Fz")
    plt.title("Tablet force components")
    plt.xlabel("time [s]")
    plt.ylabel("Force [N] (approx)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
