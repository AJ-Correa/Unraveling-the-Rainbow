import os
import glob
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# Get all matching folders
folders = glob.glob('JSSP/606/train_*_*x*')

dqn_benchmark = True

# Create empty lists for the handles and labels
handles = []
labels = []

dqn_smoothed = None
dqn_epochs = None

if dqn_benchmark:
    line_properties = {
        "DDQN": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'red'},
        "PER": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'gold'},
        "Dueling": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'purple'},
        "Noisy": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'cyan'},
        "Distributional": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'magenta'},
        "NStep": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'green'},
        "Rainbow": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'gray'}
    }

    label_order = ["DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"]

    for folder in folders:
        folder_name = os.path.basename(folder)
        name_part = folder_name.replace("train_", "")
        parts = name_part.split("_")
        label = "_".join(parts[:-1])

        if label == "DQN":
            file_path = os.path.join(folder, 'training_ave.xlsx')
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                if 'epochs' in df.columns and 'res' in df.columns:
                    dqn_smoothed = df['res'].rolling(window=30).mean()
                    dqn_epochs = df['epochs']
                else:
                    print(f"DQN file missing required columns: {file_path}")
            else:
                print(f"DQN training_ave.xlsx not found at {file_path}")
            break

    if dqn_smoothed is None:
        raise ValueError("DQN data could not be loaded. Cannot compute differences.")

    plt.figure(figsize=(18, 12))

    diff_data = []

    for folder in folders:
        folder_name = os.path.basename(folder)
        name_part = folder_name.replace("train_", "")
        parts = name_part.split("_")
        label = "_".join(parts[:-1])

        if label == "DQN" or label not in line_properties:
            continue

        file_path = os.path.join(folder, 'training_ave.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            if 'epochs' in df.columns and 'res' in df.columns:
                ext_smoothed = df['res'].rolling(window=30).mean()

                min_len = min(len(dqn_smoothed), len(ext_smoothed))
                diff = ext_smoothed.iloc[:min_len].values - dqn_smoothed.iloc[:min_len].values
                epochs = dqn_epochs.iloc[:min_len]

                diff_data.append((label, epochs, diff))
            else:
                print(f"Skipping {folder}: 'epochs' or 'res' column missing.")
        else:
            print(f"Skipping {folder}: training_ave.xlsx not found.")

    for label in label_order:
        for entry in diff_data:
            if entry[0] == label:
                _, epochs, diff = entry
                style = line_properties[label]
                plt.plot(
                    epochs,
                    diff,
                    label=label,
                    linewidth=style['linewidth'],
                    alpha=style['alpha'],
                    color=style['color']
                )
                break

    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.xlabel("Epochs", fontsize=40)
    plt.ylabel("JSSP - Δ Validation makespan (Extension − DQN)", fontsize=30)
    # plt.title("20 x 10", fontsize=40)
    plt.legend(loc='upper right', fontsize=24)
    plt.tick_params(axis='both', labelsize=40)
    plt.tight_layout()
    plt.savefig('Plots/JSSP/jssp_6x6_DQN.png', dpi=600, bbox_inches='tight')
else:
    line_properties = {
        "PPO": {'linewidth': 3.0, 'alpha': 1.0, 'color': 'blue'},
        "DQN": {'linewidth': 3.0, 'alpha': 1.0, 'color': 'green'},
        "DDQN": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'red'},
        "PER": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'gold'},
        "Dueling": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'purple'},
        "Noisy": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'cyan'},
        "Distributional": {'linewidth': 3.0, 'alpha': 1.0, 'color': 'magenta'},
        "NStep": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'crimson'},
        "Rainbow": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'gray'}
    }

    label_order = ["PPO", "DQN", "DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"]

    plt.figure(figsize=(18, 12))

    diff_data = []

    for folder in folders:
        folder_name = os.path.basename(folder)
        name_part = folder_name.replace("train_", "")
        parts = name_part.split("_")
        label = "_".join(parts[:-1])

        if label == "DQN" or label not in line_properties:
            continue

        file_path = os.path.join(folder, 'training_ave.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            if 'epochs' in df.columns and 'res' in df.columns:
                ext_smoothed = df['res'].rolling(window=30).mean()

                min_len = min(len(dqn_smoothed), len(ext_smoothed))
                diff = ext_smoothed.iloc[:min_len].values - dqn_smoothed.iloc[:min_len].values
                epochs = dqn_epochs.iloc[:min_len]

                diff_data.append((label, epochs, diff))
            else:
                print(f"Skipping {folder}: 'epochs' or 'res' column missing.")
        else:
            print(f"Skipping {folder}: training_ave.xlsx not found.")

    # Order the legend by the specified label_order
    ordered_handles = []
    ordered_labels = []
    for label in label_order:
        if label in labels:
            index = labels.index(label)
            ordered_handles.append(handles[index])
            ordered_labels.append(labels[index])

    # Add legend with the correct order
    plt.xlabel("Epochs", fontsize=40)
    # plt.ylabel("JSSP - Average validation makespan", fontsize=40)
    # plt.title("6 x 6", fontsize=40)
    plt.legend(handles=ordered_handles, labels=ordered_labels, loc='upper right', fontsize=24)
    plt.tick_params(axis='both', labelsize=40)
    # plt.gca().axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    # plt.grid(True)
    # plt.show()
    plt.savefig('Plots/JSSP/jssp_20x10_test.png', dpi=600, bbox_inches='tight')
