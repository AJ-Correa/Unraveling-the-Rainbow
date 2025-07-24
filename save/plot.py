import os
import glob
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats
import itertools
from scipy.stats import wilcoxon
import matplotlib.patches as patches
from collections import defaultdict


def plot_individual_value_and_avg_policy(folders, problem_type, num_jobs, num_machines):

    policy_labels = {"PPO", "A2C", "VMPO", "REINFORCE"}
    value_labels = {"DQN", "DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"}

    highlight_labels = {"DQN", "Distributional", "Policy (avg.)"}

    base_colors = {
        "DQN": '#1f77b4',  # blue
        "DDQN": '#ff7f0e',  # orange
        "PER": '#2ca02c',  # green
        "Dueling": '#d62728',  # red
        "Noisy": '#9467bd',  # purple
        "Distributional": '#8c564b',  # brown
        "NStep": '#e377c2',  # pink
        "Rainbow": '#7f7f7f',  # gray
        "Policy (avg.)": '#bcbd22'  # olive
    }

    line_properties = {}

    all_labels = policy_labels.union(value_labels).union({"Policy (avg.)"})

    for label in all_labels:
        if label in highlight_labels:
            line_properties[label] = {'linewidth': 3, 'alpha': 1.0, 'color': base_colors.get(label)}
        else:
            line_properties[label] = {'linewidth': 1, 'alpha': 1.0, 'color': base_colors.get(label)}

    value_runs = {label: [] for label in value_labels}
    policy_runs = []

    epochs = None

    # Load data
    for folder in folders:
        folder_name = os.path.basename(folder)
        name_part = folder_name.replace("train_", "")
        parts = name_part.split("_")
        label = "_".join(parts[:-1])

        file_path = os.path.join(folder, 'training_ave.xlsx')
        if not os.path.exists(file_path):
            continue

        df = pd.read_excel(file_path)
        if 'epochs' not in df.columns or 'res' not in df.columns:
            continue

        # Use rolling average
        res = df['res'].rolling(window=30, min_periods=1).mean().dropna().values
        if len(res) == 0:
            continue

        if epochs is None:
            epochs = df['epochs'].values[:len(res)]
        else:
            epochs = epochs[:len(res)]

        if label in value_labels:
            value_runs[label].append(res[:len(epochs)])
        elif label in policy_labels:
            policy_runs.append(res[:len(epochs)])

    # Prepare avg policy data
    if len(policy_runs) == 0:
        print("No value algorithm data found.")
        return
    policy_data = np.array(policy_runs)
    avg_policy = np.mean(policy_data, axis=0)

    plt.figure(figsize=(18, 12))

    # Plot individual value runs
    for label, runs in value_runs.items():
        if len(runs) == 0:
            continue
        runs_array = np.array(runs)
        avg_value = np.mean(runs_array, axis=0)
        plt.plot(epochs, avg_value, label=f'{label}', color=line_properties[label]["color"], linewidth=line_properties[label]["linewidth"], alpha=line_properties[label]["alpha"])

    # Plot average policy curve
    plt.plot(epochs, avg_policy, label='Policy (avg.)', color=line_properties["Policy (avg.)"]["color"], linewidth=line_properties["Policy (avg.)"]["linewidth"], alpha=line_properties["Policy (avg.)"]["alpha"])

    all_values = []
    for runs in value_runs.values():
        if len(runs) > 0:
            runs_array = np.array(runs)
            avg_value = np.mean(runs_array, axis=0)
            all_values.append(avg_value)

    all_values.append(avg_policy)
    all_values_flat = np.concatenate(all_values)

    y_min = 0.995 * np.min(all_values_flat)
    y_max = 1.005 * np.max(all_values_flat)
    plt.ylim(y_min, y_max)

    plt.xlabel("Epochs", fontsize=32)
    # plt.ylabel(f"{problem_type} - Average validation makespan", fontsize=32)
    # plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    # plt.title(f"{num_jobs} x {num_machines}", fontsize=45)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = list(base_colors.keys())
    ordered_labels = [label for label in ordered_labels if label in label_to_handle]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]
    plt.legend(ordered_handles, ordered_labels, fontsize=24)

    plt.tick_params(axis='both', labelsize=32)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
    # plt.show()
    plt.savefig(f'Plots/v2 plots/Figure 2/{problem_type}_{num_jobs}x{num_machines}.png', dpi=300, bbox_inches='tight')

def plot_individual_policy_and_avg_value(folders, problem_type, num_jobs, num_machines):
    policy_labels = {"PPO", "A2C", "VMPO", "REINFORCE"}
    value_labels = {"DQN", "DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"}

    highlight_labels = {"PPO", "A2C", "VMPO", "REINFORCE", "Value (avg.)"}

    base_colors = {
        "PPO": '#1f77b4',  # blue
        "A2C": '#ff7f0e',  # orange
        "VMPO": '#2ca02c',  # green
        "REINFORCE": '#d62728',  # red
        "Value (avg.)": '#bcbd22'  # olive
    }

    line_properties = {}

    all_labels = policy_labels.union(value_labels).union({"Value (avg.)"})

    for label in all_labels:
        if label in highlight_labels:
            line_properties[label] = {'linewidth': 3, 'alpha': 1.0, 'color': base_colors.get(label)}
        else:
            line_properties[label] = {'linewidth': 1, 'alpha': 1.0, 'color': base_colors.get(label)}

    policy_runs = {label: [] for label in policy_labels}
    value_runs = []

    epochs = None

    # Load data
    for folder in folders:
        folder_name = os.path.basename(folder)
        name_part = folder_name.replace("train_", "")
        parts = name_part.split("_")
        label = "_".join(parts[:-1])

        file_path = os.path.join(folder, 'training_ave.xlsx')
        if not os.path.exists(file_path):
            continue

        df = pd.read_excel(file_path)
        if 'epochs' not in df.columns or 'res' not in df.columns:
            continue

        # Use rolling average 30
        res = df['res'].rolling(window=30, min_periods=1).mean().dropna().values
        if len(res) == 0:
            continue

        if epochs is None:
            epochs = df['epochs'].values[:len(res)]
        else:
            epochs = epochs[:len(res)]

        if label in policy_labels:
            policy_runs[label].append(res[:len(epochs)])
        elif label in value_labels:
            value_runs.append(res[:len(epochs)])

    # Prepare avg value data
    if len(value_runs) == 0:
        print("No value algorithm data found.")
        return
    value_data = np.array(value_runs)
    avg_value = np.mean(value_data, axis=0)

    plt.figure(figsize=(18, 12))

    # Plot individual policy runs
    for label, runs in policy_runs.items():
        if len(runs) == 0:
            continue
        runs_array = np.array(runs)
        avg_policy = np.mean(runs_array, axis=0)
        plt.plot(epochs, avg_policy, label=f'{label}', color=line_properties[label]["color"], linewidth=line_properties[label]["linewidth"], alpha=line_properties[label]["alpha"])

    # Plot average value curve
    plt.plot(epochs, avg_value, label='Value (avg.)', color=line_properties["Value (avg.)"]["color"], linewidth=line_properties["Value (avg.)"]["linewidth"], alpha=line_properties["Value (avg.)"]["alpha"])

    all_values = []
    for runs in policy_runs.values():
        if len(runs) > 0:
            runs_array = np.array(runs)
            all_values.append(np.mean(runs_array, axis=0))

    all_values.append(avg_value)
    all_values_flat = np.concatenate(all_values)

    y_min = 0.995 * np.min(all_values_flat)
    y_max = 1.005 * np.max(all_values_flat)
    plt.ylim(y_min, y_max)

    plt.xlabel("Epochs", fontsize=32)
    # plt.ylabel(f"{problem_type} - Average validation makespan", fontsize=32)
    # plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    # plt.title(f"{num_jobs} x {num_machines}", fontsize=45)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = list(base_colors.keys())
    ordered_labels = [label for label in ordered_labels if label in label_to_handle]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]
    plt.legend(ordered_handles, ordered_labels, fontsize=24)

    plt.tick_params(axis='both', labelsize=32)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
    # plt.show()
    plt.savefig(f'Plots/v2 plots/Figure 3/{problem_type}_{num_jobs}x{num_machines}.png', dpi=300, bbox_inches='tight')


def plot_policy_vs_value_average(folders, problem_type, num_jobs, num_machines):
    policy_data = []
    value_data = []
    epochs = None

    value_labels = {"DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"}
    policy_labels = {"PPO", "A2C", "VMPO", "REINFORCE"}
    line_properties = {
        "Policy": {'linewidth': 2.0, 'alpha': 0.25, 'color': '#5DADE2'},
        "Value": {'linewidth': 2.0, 'alpha': 0.25, 'color': '#F39C12'}
    }

    for folder in folders:
        folder_name = os.path.basename(folder)
        name_part = folder_name.replace("train_", "")
        parts = name_part.split("_")
        label = "_".join(parts[:-1])

        file_path = os.path.join(folder, 'training_ave.xlsx')
        if not os.path.exists(file_path):
            print(f"Skipping {folder}: training_ave.xlsx not found.")
            continue

        df = pd.read_excel(file_path)
        if 'epochs' not in df.columns or 'res' not in df.columns:
            print(f"Skipping {folder}: 'epochs' or 'res' column missing.")
            continue

        res = df['res'].dropna().values
        if len(res) == 0:
            print(f"Skipping {folder}: all 'res' values are NaN.")
            continue

        if epochs is None:
            epochs = df['epochs'].dropna().values
        else:
            epochs = np.minimum(epochs[:len(res)], df['epochs'].dropna().values[:len(res)])

        if label in value_labels:
            value_data.append(res)
        elif label in policy_labels:
            policy_data.append(res)

    if not policy_data or not value_data:
        print("Not enough data to compute policy and value averages.")
        return

    min_len = min(min(map(len, policy_data)), min(map(len, value_data)))
    policy_data = [p[:min_len] for p in policy_data]
    value_data = [v[:min_len] for v in value_data]
    epochs = epochs[:min_len]

    avg_policy = np.mean(policy_data, axis=0)
    std_policy = np.std(policy_data, axis=0)

    avg_value = np.mean(value_data, axis=0)
    std_value = np.std(value_data, axis=0)

    y_min = min(np.min(np.mean(policy_data, axis=0)), np.min(np.mean(value_data, axis=0)))
    y_max = max(np.max(np.mean(policy_data, axis=0)), np.max(np.mean(value_data, axis=0)))

    plt.figure(figsize=(18, 12))

    plt.fill_between(epochs, avg_policy - std_policy, avg_policy + std_policy, color=line_properties["Policy"]["color"], alpha=line_properties["Policy"]["alpha"])
    plt.plot(epochs, avg_policy, label='Policy', color=line_properties["Policy"]["color"], linewidth=line_properties["Policy"]["linewidth"])

    plt.fill_between(epochs, avg_value - std_value, avg_value + std_value, color=line_properties["Value"]["color"], alpha=line_properties["Value"]["alpha"])
    plt.plot(epochs, avg_value, label='Value', color=line_properties["Value"]["color"], linewidth=line_properties["Value"]["linewidth"])

    plt.ylim(y_min * 0.98, y_max * 1.02)

    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.xlabel("Epochs", fontsize=32)
    # plt.ylabel(f"{problem_type} - Average validation makespan", fontsize=32)
    # plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    # plt.title(f"{num_jobs} x {num_machines}", fontsize=45)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = list(line_properties.keys())
    ordered_labels = [label for label in ordered_labels if label in label_to_handle]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]
    plt.legend(ordered_handles, ordered_labels, fontsize=24)

    plt.tick_params(axis='both', labelsize=32)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
    # plt.show()
    plt.savefig(f'Plots/v2 plots/Figure 4/{problem_type}_{num_jobs}x{num_machines}.png', dpi=300, bbox_inches='tight')

def plot_training_times():
    df = pd.DataFrame({
        'Algorithm': ['DQN', 'DDQN', 'PER', 'Dueling', 'NStep', 'Noisy', 'Distributional', 'Rainbow', 'PPO', 'A2C', 'REINFORCE', 'V-MPO'],
        'Training time (m)': [165.6, 197.9, 176.2, 174.1, 272.6, 190.7, 194.8, 390.2, 220.0, 188.5, 186.3, 192.8]
    })

    df = df.sort_values(by='Training time (m)', ascending=False).reset_index(drop=True)

    baseline = df['Training time (m)'].mean()
    df['Deviation (%)'] = ((df['Training time (m)'] - baseline) / baseline) * 100

    # Set up plot
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Create a unique color for each algorithm
    palette = sns.color_palette("husl", n_colors=len(df))

    # Plot with unique colors
    ax = sns.barplot(
        data=df,
        y='Algorithm',
        x='Training time (m)',
        palette=palette
    )

    # Annotate deviation
    for i, row in df.iterrows():
        bar = ax.patches[i]
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2

        # Training time (centered above bar)
        ax.text(
            width - 5, y_center,
            f"{row['Training time (m)']:.1f}m",
            ha='right', va='center',
            fontsize=11, color='white', weight='bold'
        )

        # % deviation (to the right of bar)
        ax.text(
            width + 5, y_center,
            f"{row['Deviation (%)']:+.1f}%",
            ha='left', va='center',
            fontsize=11, color='black'
        )

    ax.axvline(baseline, ls='--', color='gray', linewidth=1.5)
    ax.text(baseline - 18, -0.6, f"Average: {baseline:.1f}m", color='black', fontsize=11)

    # ax.set_title("Average Training Time per Algorithm", fontsize=16, weight='bold')
    ax.set_xlabel("Training Time (minutes)", fontsize=13)
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Plots/v2 plots/Figure 4/Figure 4.png', dpi=300, bbox_inches='tight')

def compute_pvalue_matrix(df, problem_type, test_func=wilcoxon):
    sub_df = df[df['Problem type'] == problem_type]
    algorithms = sub_df['Algorithm'].unique()
    pval_matrix = pd.DataFrame(index=algorithms, columns=algorithms, dtype=float)

    # Pivot to align by Instance index
    pivot = sub_df.pivot(index='Instance index', columns='Algorithm', values='Obj')
    win_counts = {alg: 0 for alg in algorithms}

    for alg1, alg2 in itertools.combinations(algorithms, 2):
        try:
            values1 = pivot[alg1]
            values2 = pivot[alg2]

            # Drop any rows with NaN (in case some algorithms failed)
            paired = pd.DataFrame({'a': values1, 'b': values2}).dropna()

            if len(paired) < 10:
                pval = np.nan  # Not enough data to test
            else:
                stat, pval = test_func(paired['a'], paired['b'])

                if pval <= 0.05:
                    mean1 = paired['a'].mean()
                    mean2 = paired['b'].mean()

                    if mean1 < mean2:
                        win_counts[alg1] += 1
                    elif mean2 < mean1:
                        win_counts[alg2] += 1
        except Exception:
            pval = np.nan

        pval_matrix.loc[alg1, alg2] = pval
        pval_matrix.loc[alg2, alg1] = pval

    # Diagonal should be zero
    for alg in algorithms:
        pval_matrix.loc[alg, alg] = 0.0

    return pval_matrix, win_counts

def plot_heatmap(filepath):
    xls = pd.ExcelFile(filepath)
    sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

    for problem_type in ['FJSP', 'JSSP']:
        total_win_counts = defaultdict(int)
        for instance_size, df in sheets.items():
            try:
                pval_matrix, win_counts = compute_pvalue_matrix(df, problem_type)
                for alg, count in win_counts.items():
                    total_win_counts[alg] += count

                annot = pval_matrix.copy().astype(object)

                for i in annot.index:
                    for j in annot.columns:
                        if i == j:
                            annot.loc[i, j] = "--"
                        else:
                            p = pval_matrix.loc[i, j]
                            annot.loc[i, j] = f"{p:.2g}"

                fig, ax = plt.subplots(figsize=(13, 13))
                ax = sns.heatmap(
                    pval_matrix.astype(float),
                    annot=annot,
                    fmt="",  # Disable automatic formatting
                    cmap=sns.color_palette("coolwarm_r", as_cmap=True),
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'p-value'},
                    cbar=True if instance_size == "40x10" else False,
                    annot_kws={"size": 14 if instance_size != "40x10" else 11},
                    ax=ax
                )

                cbar = ax.collections[0].colorbar
                if cbar is not None:
                    cbar.set_label('p-value', fontsize=28)
                    cbar.ax.tick_params(labelsize=28)

                # Highlight significant cells with white border (p <= 0.05)
                for i, alg1 in enumerate(pval_matrix.index):
                    for j, alg2 in enumerate(pval_matrix.columns):
                        p = pval_matrix.loc[alg1, alg2]
                        if i != j and p <= 0.05:
                            # Draw white rectangle with linewidth 2
                            rect = patches.Rectangle(
                                (j, i), 1, 1, fill=False, edgecolor='white', lw=3
                            )
                            ax.add_patch(rect)

                # if instance_size == "6x6":
                #     plt.ylabel('')  # Remove default ylabel if any
                #     plt.text(
                #         x=-3.0,  # slightly left of y-axis (adjust as needed)
                #         y=pval_matrix.shape[0] / 2,
                #         s=problem_type,
                #         rotation=90,
                #         verticalalignment='center',
                #         fontsize=50
                #     )

                # if problem_type == "FJSP":
                #     plt.title(f"{instance_size}", fontsize=50)

                if instance_size == "6x6":
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=28, rotation=0)
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')

                # X-axis: only for bottom row (JSSP)
                if problem_type == "JSSP":
                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=28, rotation=90)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel('')

                # plt.tight_layout()
                # plt.show()
                # plt.savefig(f"Plots/v2 plots/Figure 5/{problem_type}_{instance_size}.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Win counts ({problem_type}, {instance_size}):")
                for alg, count in sorted(win_counts.items(), key=lambda x: -x[1]):
                    print(f"  {alg}: {count}")
                # print(f"Done: {instance_size} - {problem_type}")
            except Exception as e:
                print(f"Skipped {instance_size} - {problem_type} due to error: {e}")

        print(f"\n=== Total win counts across all instance sizes on {problem_type} ===")
        for alg, count in sorted(total_win_counts.items(), key=lambda x: -x[1]):
            print(f"{alg}: {count}")
        print()

problem_type = "JSSP"
num_jobs = 20
num_machines = 10

# Get all matching folders
folders = glob.glob(f'{problem_type}/{num_jobs}{f"0{num_machines}" if num_machines < 10 else num_machines}/train_*_*x*')

# plot_individual_value_and_avg_policy(folders, problem_type, num_jobs, num_machines)
# plot_individual_policy_and_avg_value(folders, problem_type, num_jobs, num_machines)
# plot_policy_vs_value_average(folders, problem_type, num_jobs, num_machines)
# plot_training_times()
plot_heatmap("../results/testing_results.xlsx")



# dqn_benchmark = True
#
# # Create empty lists for the handles and labels
# handles = []
# labels = []
#
# dqn_smoothed = None
# dqn_epochs = None
#
# if dqn_benchmark:
#     line_properties = {
#         "DDQN": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'red'},
#         "PER": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'gold'},
#         "Dueling": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'purple'},
#         "Noisy": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'cyan'},
#         "Distributional": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'magenta'},
#         "NStep": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'green'},
#         "Rainbow": {'linewidth': 2.0, 'alpha': 0.8, 'color': 'gray'}
#     }
#
#     label_order = ["DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"]
#
#     for folder in folders:
#         folder_name = os.path.basename(folder)
#         name_part = folder_name.replace("train_", "")
#         parts = name_part.split("_")
#         label = "_".join(parts[:-1])
#
#         if label == "DQN":
#             file_path = os.path.join(folder, 'training_ave.xlsx')
#             if os.path.exists(file_path):
#                 df = pd.read_excel(file_path)
#                 if 'epochs' in df.columns and 'res' in df.columns:
#                     dqn_smoothed = df['res'].rolling(window=30).mean()
#                     dqn_epochs = df['epochs']
#                 else:
#                     print(f"DQN file missing required columns: {file_path}")
#             else:
#                 print(f"DQN training_ave.xlsx not found at {file_path}")
#             break
#
#     if dqn_smoothed is None:
#         raise ValueError("DQN data could not be loaded. Cannot compute differences.")
#
#     plt.figure(figsize=(18, 12))
#
#     diff_data = []
#
#     for folder in folders:
#         folder_name = os.path.basename(folder)
#         name_part = folder_name.replace("train_", "")
#         parts = name_part.split("_")
#         label = "_".join(parts[:-1])
#
#         if label == "DQN" or label not in line_properties:
#             continue
#
#         file_path = os.path.join(folder, 'training_ave.xlsx')
#         if os.path.exists(file_path):
#             df = pd.read_excel(file_path)
#             if 'epochs' in df.columns and 'res' in df.columns:
#                 ext_smoothed = df['res'].rolling(window=30).mean()
#
#                 min_len = min(len(dqn_smoothed), len(ext_smoothed))
#                 diff = ext_smoothed.iloc[:min_len].values - dqn_smoothed.iloc[:min_len].values
#                 epochs = dqn_epochs.iloc[:min_len]
#
#                 diff_data.append((label, epochs, diff))
#             else:
#                 print(f"Skipping {folder}: 'epochs' or 'res' column missing.")
#         else:
#             print(f"Skipping {folder}: training_ave.xlsx not found.")
#
#     for label in label_order:
#         for entry in diff_data:
#             if entry[0] == label:
#                 _, epochs, diff = entry
#                 style = line_properties[label]
#                 plt.plot(
#                     epochs,
#                     diff,
#                     label=label,
#                     linewidth=style['linewidth'],
#                     alpha=style['alpha'],
#                     color=style['color']
#                 )
#                 break
#
#     plt.axhline(0, color='black', linestyle='--', linewidth=2)
#     plt.xlabel("Epochs", fontsize=40)
#     plt.ylabel("JSSP - Δ Validation makespan (Extension − DQN)", fontsize=30)
#     # plt.title("20 x 10", fontsize=40)
#     plt.legend(loc='upper right', fontsize=24)
#     plt.tick_params(axis='both', labelsize=40)
#     plt.tight_layout()
#     plt.savefig('Plots/JSSP/jssp_6x6_DQN.png', dpi=600, bbox_inches='tight')
# else:
#     line_properties = {
#         "PPO": {'linewidth': 3.0, 'alpha': 1.0, 'color': 'blue'},
#         "DQN": {'linewidth': 3.0, 'alpha': 1.0, 'color': 'green'},
#         "DDQN": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'red'},
#         "PER": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'gold'},
#         "Dueling": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'purple'},
#         "Noisy": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'cyan'},
#         "Distributional": {'linewidth': 3.0, 'alpha': 1.0, 'color': 'magenta'},
#         "NStep": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'crimson'},
#         "Rainbow": {'linewidth': 1.0, 'alpha': 0.8, 'color': 'gray'}
#     }
#
#     label_order = ["PPO", "DQN", "DDQN", "PER", "Dueling", "Noisy", "Distributional", "NStep", "Rainbow"]
#
#     plt.figure(figsize=(18, 12))
#
#     diff_data = []
#
#     for folder in folders:
#         folder_name = os.path.basename(folder)
#         name_part = folder_name.replace("train_", "")
#         parts = name_part.split("_")
#         label = "_".join(parts[:-1])
#
#         if label == "DQN" or label not in line_properties:
#             continue
#
#         file_path = os.path.join(folder, 'training_ave.xlsx')
#         if os.path.exists(file_path):
#             df = pd.read_excel(file_path)
#             if 'epochs' in df.columns and 'res' in df.columns:
#                 ext_smoothed = df['res'].rolling(window=30).mean()
#
#                 min_len = min(len(dqn_smoothed), len(ext_smoothed))
#                 diff = ext_smoothed.iloc[:min_len].values - dqn_smoothed.iloc[:min_len].values
#                 epochs = dqn_epochs.iloc[:min_len]
#
#                 diff_data.append((label, epochs, diff))
#             else:
#                 print(f"Skipping {folder}: 'epochs' or 'res' column missing.")
#         else:
#             print(f"Skipping {folder}: training_ave.xlsx not found.")
#
#     # Order the legend by the specified label_order
#     ordered_handles = []
#     ordered_labels = []
#     for label in label_order:
#         if label in labels:
#             index = labels.index(label)
#             ordered_handles.append(handles[index])
#             ordered_labels.append(labels[index])
#
#     # Add legend with the correct order
#     plt.xlabel("Epochs", fontsize=40)
#     # plt.ylabel("JSSP - Average validation makespan", fontsize=40)
#     # plt.title("6 x 6", fontsize=40)
#     plt.legend(handles=ordered_handles, labels=ordered_labels, loc='upper right', fontsize=24)
#     plt.tick_params(axis='both', labelsize=40)
#     # plt.gca().axes.get_xaxis().set_visible(False)
#     plt.tight_layout()
#     # plt.grid(True)
#     # plt.show()
#     plt.savefig('Plots/JSSP/jssp_20x10_test.png', dpi=600, bbox_inches='tight')
