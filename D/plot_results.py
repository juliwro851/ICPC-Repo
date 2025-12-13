
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values(by=df.columns[0])
    plt.figure()
    for col in df.columns:
        if col != df.columns[0]:
            plt.plot(df[df.columns[0]], df[col], label=col)
    plt.legend()
    plt.xlabel(df.columns[0])
    plt.ylabel("value")
    plt.title("Training stats")
    plt.tight_layout()
    plt.show()

    #df = pd.DataFrame(results)  # wczytane logi RLlib

    plt.plot(df["training_iteration"], df["custom_metrics/episode_success_mean"])
    plt.xlabel("Iteracja")
    plt.ylabel("Odsetek sukcesów")
    plt.title("Skuteczność misji")
    plt.show()

    plt.plot(df["training_iteration"], df["custom_metrics/rescues_done_mean"], label="Uratowani")
    plt.plot(df["training_iteration"], df["custom_metrics/rubble_cleared_mean"], label="Usunięty gruz")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
