import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
from scipy import stats as scipy_stats

plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["axes.grid"] = True

MAX_ITER = 100
MAX_EPISODES = 500  

GLOBAL_REWARD_BONUS = 50
FONT_SIZE_TITLE = 20
FONT_SIZE_OTHERS = 18
DIMENTION_X = 12
DIMENTION_Y = 7

EXPERIMENT_CONFIG = {
    'A1': {'reward_type': 'local'},
    'A2': {'reward_type': 'global'},
    'B1': {'reward_type': 'global'},
    'B2': {'reward_type': 'global'},
    'C': {'reward_type': 'global'},
    'D': {'reward_type': 'global'},
}

# --- plot configuration ---
PLOT_CONFIG = {
    "Average Reward per Episode": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
        ("homo", "D", "D homogeneous"),
        ("hetero", "D", "D heterogeneous"),
    ],
    "Number of Rescued Victims": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
        ("homo", "D", "D homogeneous"),
        ("hetero", "D", "D heterogeneous"),
    ],
        "Number of Rescued Victims (first 100)": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
        ("homo", "D", "D homogeneous"),
        ("hetero", "D", "D heterogeneous"),
    ],
    "Average Episode Length": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
        ("homo", "D", "D homogeneous"),
        ("hetero", "D", "D heterogeneous"),
    ],
    "Normalized Reward Curve AUC": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
    ],
    "Average Success Rate": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
    ],
    "Density/Successful Clearing Actions": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
    ],
    "Density/Clearing Attempts": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
    ],
    "Density/Wall Collisions": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
    ],
    "Density/Action Withdrawals": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
    ],
     "Removed Rubble/Clearing Attempts": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
        ("homo", "D", "D homogeneous"),
        ("hetero", "D", "D heterogeneous"),
    ],
    "Rescued Victims/Removed Obstacles": [
        ("homo", "A1", "A1 homogeneous"),
        ("hetero", "A1", "A1 heterogeneous"),
        ("homo", "A2", "A2 homogeneous"),
        ("hetero", "A2", "A2 heterogeneous"),
        ("strazak", "B1", "B1 firefighters"),
        ("ratownicy", "B2", "B2 rescuers"),
        ("homo", "C", "C homogeneous"),
        ("hetero", "C", "C heterogeneous"),
        ("homo", "D", "D homogeneous"),
        ("hetero", "D", "D heterogeneous"),
    ],
}

EPISODE_REGEX = re.compile(
    r"(results|episode_stats)_(homo|hetero|ratownicy|strazak)_(\d+)_((A|B)\d|C|D)\.csv"
)

def load_episode_files(path):
    datasets = {}

    for fname in os.listdir(path):
        if not fname.lower().endswith('.csv'):
            continue

        match = EPISODE_REGEX.match(fname)
        if not match:
            print(f"Invalid filename format: {fname}")
            continue

        file = match.group(1).lower()
        variant = match.group(2).lower()
        iteration = int(match.group(3))
        experiment = match.group(4).upper()

        df = pd.read_csv(os.path.join(path, fname))
        
        if file == "episode_stats" and len(df) > MAX_EPISODES:
            print(f"Truncating {fname} from {len(df)} to {MAX_EPISODES} episodes")
            df = df.iloc[:MAX_EPISODES]

        datasets \
            .setdefault(file, {}) \
            .setdefault(variant, {}) \
            .setdefault(experiment, {}) \
            .setdefault(iteration, []) \
            .append(df)

        print(
            f"Loaded: {fname} -> "
            f"file: {file}, variant: {variant}, experiment: {experiment}, iteration: {iteration}, rows: {len(df)}"
        )
    return datasets


def average_series(series_list):
    if not series_list:
        return pd.Series(), pd.Series()
    
    max_len = max(len(s) for s in series_list)
    aligned = []
    for s in series_list:
        if len(s) < max_len:
            pad = pd.Series([np.nan] * (max_len - len(s)))
            s = pd.concat([s, pad], ignore_index=True)
        aligned.append(s.reset_index(drop=True))
    df = pd.DataFrame(aligned).T
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return mean, std


def aggregate_across_iterations(all_iteration_series):

    if not all_iteration_series:
        return pd.Series(), pd.Series(), pd.Series()
    
    max_len = max(len(s) for s in all_iteration_series)
    aligned = []
    for s in all_iteration_series:
        if len(s) < max_len:
            pad = pd.Series([np.nan] * (max_len - len(s)))
            s = pd.concat([s, pad], ignore_index=True)
        aligned.append(s.reset_index(drop=True))
    
    df = pd.DataFrame(aligned).T
    
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    sem = df.sem(axis=1) 
    
    return mean, std, sem


def calculate_statistics(mean_series, std_series, sem_series, label, n_iterations):
    clean_data = mean_series.dropna()
    
    if len(clean_data) == 0:
        return None
    
    stats = {
        'label': label,
        'n_iterations': n_iterations,
        'mean': clean_data.mean(),
        'std': std_series.dropna().mean(),
        'sem': sem_series.dropna().mean(),
        'min': clean_data.min(),
        'max': clean_data.max(),
        'median': clean_data.median(),
        'q25': clean_data.quantile(0.25),
        'q75': clean_data.quantile(0.75),
        'final_value': clean_data.iloc[-1] if len(clean_data) > 0 else np.nan,
        'final_std': std_series.dropna().iloc[-1] if len(std_series.dropna()) > 0 else np.nan,
        'first_value': clean_data.iloc[0] if len(clean_data) > 0 else np.nan,
        'improvement': clean_data.iloc[-1] - clean_data.iloc[0] if len(clean_data) > 0 else np.nan,
        'improvement_pct': ((clean_data.iloc[-1] - clean_data.iloc[0]) / abs(clean_data.iloc[0]) * 100) 
                          if len(clean_data) > 0 and clean_data.iloc[0] != 0 else np.nan,
        'count': len(clean_data)
    }
    
    return stats


def check_normality(data, label):
    clean_data = [x for x in data if not np.isnan(x)]
    
    if len(clean_data) < 3:
        return None, None, None
    
    try:
        statistic, p_value = scipy_stats.shapiro(clean_data)
        is_normal = p_value > 0.05
        return is_normal, p_value, statistic
    except:
        return None, None, None


def check_equal_variances(data1, data2):

    clean_data1 = [x for x in data1 if not np.isnan(x)]
    clean_data2 = [x for x in data2 if not np.isnan(x)]
    
    if len(clean_data1) < 2 or len(clean_data2) < 2:
        return None, None, None
    
    try:
        statistic, p_value = scipy_stats.levene(clean_data1, clean_data2)
        equal_var = p_value > 0.05
        return equal_var, p_value, statistic
    except:
        return None, None, None


def perform_statistical_tests(aggregated_data_dict, metric_name):
    test_results = []
    labels = list(aggregated_data_dict.keys())
    
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            data1 = aggregated_data_dict[label1]['all_means']
            data2 = aggregated_data_dict[label2]['all_means']
            
            if len(data1) < 2 or len(data2) < 2:
                continue
            
            norm1, p_norm1, _ = check_normality(data1, label1)
            norm2, p_norm2, _ = check_normality(data2, label2)
            equal_var, p_levene, _ = check_equal_variances(data1, data2)
            
            equal_var_param = equal_var if equal_var is not None else False
            t_stat, p_value = scipy_stats.ttest_ind(data1, data2, 
                                                     equal_var=equal_var_param,
                                                     nan_policy='omit')
            
            try:
                u_stat, p_value_mw = scipy_stats.mannwhitneyu(data1, data2, 
                                                               alternative='two-sided')
            except:
                u_stat, p_value_mw = np.nan, np.nan
            
            mean1, mean2 = np.nanmean(data1), np.nanmean(data2)
            std1, std2 = np.nanstd(data1, ddof=1), np.nanstd(data2, ddof=1)
            
            n1, n2 = len(data1), len(data2)
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan
            
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect_size = "negligible"
            elif abs_d < 0.5:
                effect_size = "small"
            elif abs_d < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"
            
            if norm1 and norm2:
                if equal_var:
                    recommended_test = "Student's t-test"
                else:
                    recommended_test = "Welch's t-test"
            else:
                recommended_test = "Mann-Whitney U"
            
            test_results.append({
                'comparison': f"{label1} vs {label2}",
                'mean_1': mean1,
                'std_1': std1,
                'n_1': n1,
                'mean_2': mean2,
                'std_2': std2,
                'n_2': n2,
                'diff': mean1 - mean2,
                'diff_pct': ((mean1 - mean2) / abs(mean2) * 100) if mean2 != 0 else np.nan,
                't_statistic': t_stat,
                'p_value_t': p_value,
                'u_statistic': u_stat,
                'p_value_MW': p_value_mw,
                'cohens_d': cohens_d,
                'effect_size': effect_size,
                'significance': significance,
                'normal_1': 'Yes' if norm1 else 'No' if norm1 is not None else 'N/A',
                'normal_2': 'Yes' if norm2 else 'No' if norm2 is not None else 'N/A',
                'equal_var': 'Yes' if equal_var else 'No' if equal_var is not None else 'N/A',
                'recommended_test': recommended_test,
                'p_normality_1': p_norm1,
                'p_normality_2': p_norm2,
                'p_levene': p_levene
            })
    
    return pd.DataFrame(test_results) if test_results else None


def plot_metrics(datasets, metric_name, metric_col_candidates, plot_type='line', output_dir='./statistics', xlabel = "OX", ylabel = "OY"):
    
    all_stats = []
    aggregated_data = {}
    has_data = False
    
    print(f"\n{'='*80}")
    print(f"CREATING PLOT: {metric_name}")
    print(f"{'='*80}")

    if metric_name not in PLOT_CONFIG:
        print(f"No configuration for: {metric_name}")
        return

    plt.figure(figsize=(DIMENTION_X, DIMENTION_Y))

    for variant, experiment, base_label in PLOT_CONFIG[metric_name]:

        print(f"\nProcessing: variant={variant}, experiment={experiment}")

        all_iteration_means = []
        all_iteration_series = []
        
        found_data = False
        
        for file_type in datasets:
            if variant not in datasets[file_type]:
                continue
                
            if experiment not in datasets[file_type][variant]:
                continue
            
            iterations = sorted(datasets[file_type][variant][experiment].keys())
            
            for iteration in iterations:
                dfs = datasets[file_type][variant][experiment][iteration]
                series_list = []
                
                for df in dfs:
                    for col in metric_col_candidates:
                        if col in df.columns:
                            series = df[col]
                            series_list.append(series)
                            break
                
                if series_list:
                    iter_mean, _ = average_series(series_list)
                    all_iteration_series.append(iter_mean)
                    all_iteration_means.append(iter_mean.mean())
                    found_data = True
        
        if not found_data or not all_iteration_series:
            print(f"No data found")
            continue
        
        print(f"Found {len(all_iteration_series)} iterations")
        
        mean, std, sem = aggregate_across_iterations(all_iteration_series)
        has_data = True
        
        aggregated_data[base_label] = {
            'mean': mean,
            'std': std,
            'sem': sem,
            'all_means': all_iteration_means,
            'n_iterations': len(all_iteration_series)
        }
        
        stats = calculate_statistics(mean, std, sem, base_label, len(all_iteration_series))
        if stats:
            all_stats.append(stats)

        if plot_type == 'line':
            max_plot_len = MAX_ITER if metric_name in ["Average Reward per Episode", "Average Episode Length"] else len(mean)

            if len(mean) < max_plot_len:
                pad_len = max_plot_len - len(mean)
                mean = pd.concat([mean, pd.Series([np.nan]*pad_len)], ignore_index=True)
                std = pd.concat([std, pd.Series([np.nan]*pad_len)], ignore_index=True)
                sem = pd.concat([sem, pd.Series([np.nan]*pad_len)], ignore_index=True)
            else:
                mean = mean[:max_plot_len]
                std = std[:max_plot_len]
                sem = sem[:max_plot_len]
            
            if metric_name == "Number of Rescued Victims (first 100)":
                x_axis = range(1, 60+1)
                mean = mean[:60]
                std = std[:60]
                sem = sem[:60]
            else:
                x_axis = range(1, max_plot_len + 1)
            plt.plot(x_axis, mean, label=f"{base_label}", linewidth=2.5)
            plt.fill_between(x_axis, mean - 1.96*sem, mean + 1.96*sem, alpha=0.2)

        elif plot_type == 'hist':
            all_values = []
            for iter_series in all_iteration_series:
                all_values.extend(iter_series.dropna().tolist())

            plt.hist(all_values, bins=30, alpha=0.6, label=f"{base_label}")

        if plot_type == 'scatter':
            color_map = {
                "A1 homogeneous": "tab:blue",
                "A1 heterogeneous": "tab:orange",
                "A2 homogeneous": "tab:green",
                "A2 heterogeneous": "tab:red",
                "B1 firefighters": "tab:purple",
                "B2 rescuers": "tab:brown",
                "C homogeneous": "tab:pink",
                "C heterogeneous": "tab:gray",
                "D homogeneous": "tab:olive",
                "D heterogeneous": "tab:cyan",
            }

            batch1_labels = [
                "A2 homogeneous", "A2 heterogeneous",
                "A1 homogeneous", "A1 heterogeneous"
            ]
            batch2_labels = [
                "B1 firefighters", "B2 rescuers",
                "C homogeneous", "C heterogeneous",
            ]
            batch3_labels = [
                "D homogeneous", "D heterogeneous"
            ]

            global_x_min, global_x_max = float('inf'), float('-inf')
            global_y_min, global_y_max = float('inf'), float('-inf')
            
            all_batches = [batch1_labels, batch2_labels, batch3_labels]
            
            for batch_labels in all_batches:
                for base_label in batch_labels:
                    found = False
                    for variant, experiment, label in PLOT_CONFIG[metric_name]:
                        if label == base_label:
                            found = True
                            break
                    if not found:
                        continue

                    for file_type in datasets:
                        if variant in datasets[file_type] and experiment in datasets[file_type][variant]:
                            iterations = datasets[file_type][variant][experiment]
                            for iteration, dfs in iterations.items():
                                for df in dfs:
                                    if metric_name == "Removed Rubble/Clearing Attempts":
                                        if all(col in df.columns for col in ["rubble_cleared", "clear_attempts"]):
                                            x = df["rubble_cleared"].dropna().tolist()
                                            y = df["clear_attempts"].dropna().tolist()
                                            if x and y:
                                                global_x_min = min(global_x_min, min(x))
                                                global_x_max = max(global_x_max, max(x))
                                                global_y_min = min(global_y_min, min(y))
                                                global_y_max = max(global_y_max, max(y))
                                    elif metric_name == "Rescued Victims/Removed Obstacles":
                                        if all(col in df.columns for col in ["rubble_cleared", "rescues_done"]):
                                            x = df["rubble_cleared"].dropna().tolist()
                                            y = df["rescues_done"].dropna().tolist()
                                            if x and y:
                                                global_x_min = min(global_x_min, min(x))
                                                global_x_max = max(global_x_max, max(x))
                                                global_y_min = min(global_y_min, min(y))
                                                global_y_max = max(global_y_max, max(y))

            x_margin = (global_x_max - global_x_min) * 0.05
            y_margin = (global_y_max - global_y_min) * 0.05
            global_x_min -= x_margin
            global_x_max += x_margin
            global_y_min -= y_margin
            global_y_max += y_margin

            for batch_idx, batch_labels in enumerate(all_batches, start=1):
                plt.figure(figsize=(DIMENTION_X, DIMENTION_Y))
                batch_has_data = False

                for base_label in batch_labels:
                    found = False
                    for variant, experiment, label in PLOT_CONFIG[metric_name]:
                        if label == base_label:
                            found = True
                            break
                    if not found:
                        print(f"Nie znaleziono wariantu dla {base_label}")
                        continue

                    all_x, all_y = [], []
                    for file_type in datasets:
                        if variant in datasets[file_type] and experiment in datasets[file_type][variant]:
                            iterations = datasets[file_type][variant][experiment]
                            for iteration, dfs in iterations.items():
                                for df in dfs:
                                    if metric_name == "Removed Rubble/Clearing Attempts":
                                        if all(col in df.columns for col in ["rubble_cleared", "clear_attempts"]):
                                            x = df["rubble_cleared"].dropna().tolist()
                                            y = df["clear_attempts"].dropna().tolist()
                                            min_len = min(len(x), len(y))
                                            all_x.extend(x[:min_len])
                                            all_y.extend(y[:min_len])
                                            batch_has_data = True
                                    elif metric_name == "Rescued Victims/Removed Obstacles":
                                        if all(col in df.columns for col in ["rubble_cleared", "rescues_done"]):
                                            x = df["rubble_cleared"].dropna().tolist()
                                            y = df["rescues_done"].dropna().tolist()
                                            min_len = min(len(x), len(y))
                                            all_x.extend(x[:min_len])
                                            all_y.extend(y[:min_len])
                                            batch_has_data = True

                    if all_x and all_y:
                        plt.scatter(all_x, all_y, label=base_label, marker='x', alpha=0.6, s=50, color=color_map.get(base_label))

                if batch_has_data:
                    plt.xlim(global_x_min, global_x_max)
                    plt.ylim(global_y_min, global_y_max)
                    
                    plt.xlabel(xlabel, fontsize=FONT_SIZE_OTHERS)
                    plt.ylabel(ylabel, fontsize=FONT_SIZE_OTHERS)
                    plt.title(f"{metric_name} (batch {batch_idx})", fontsize=FONT_SIZE_TITLE)
                    plt.legend(loc='best', fontsize=FONT_SIZE_OTHERS, framealpha=0.9)
                    plt.tight_layout()
                    safe_name = f"{metric_name.replace('/', '_').replace(' ', '_')}_batch_{batch_idx}"
                    plot_file = f"{output_dir}/plot_{safe_name}.pdf"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    print(f"\nSAVED SCATTER PLOT: {plot_file}")
                    plt.show()
                    plt.close()
            return

    if has_data:
        title = f"{metric_name} "
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.legend(loc='best', fontsize=FONT_SIZE_OTHERS, framealpha=0.9)
        plt.tight_layout()
        
        safe_name = metric_name.replace("/", "_").replace(" ", "_")
        plot_file = f"{output_dir}/plot_{safe_name}.pdf"
        
        try:
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"\nSAVED PLOT: {plot_file}")
        except Exception as e:
            print(f"\nSave error: {e}")
        
        try:
            plt.show()
            print(f"DISPLAYED PLOT")
        except Exception as e:
            print(f"Display error: {e}")
    else:
        print(f"\nNO DATA")
    
    plt.close()

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df = stats_df.sort_values(['label'])
        
        safe_name = metric_name.replace("/", "_").replace(" ", "_")
        stats_file = f"{output_dir}/stats_{safe_name}.csv"
        stats_df.to_csv(stats_file, index=False, float_format='%.4f')
        
        print(f"\n{'='*80}")
        print(f"STATISTICS: {metric_name}")
        print(f"{'='*80}")
        print(stats_df.to_string(index=False))
        print(f"\nSaved to: {stats_file}")
    
    if len(aggregated_data) >= 2:
        test_results = perform_statistical_tests(aggregated_data, metric_name)
        if test_results is not None and not test_results.empty:
            safe_name = metric_name.replace("/", "_").replace(" ", "_")
            test_file = f"{output_dir}/statistical_tests_{safe_name}.csv"
            test_results.to_csv(test_file, index=False, float_format='%.4f')
            
            print(f"\n{'='*80}")
            print(f"STATISTICAL TESTS: {metric_name}")
            print(f"{'='*80}")
            print(test_results[['comparison', 'mean_1', 'mean_2', 'diff', 'p_value_t', 
                               'p_value_MW', 'cohens_d', 'effect_size', 'significance',
                               'recommended_test']].to_string(index=False))
            print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
            print(f"Effect size: negligible |d|<0.2, small |d|<0.5, medium |d|<0.8, large |d|≥0.8")
            print(f"\n✓ Full results saved to: {test_file}\n")


def generate_summary_report(datasets, output_dir='./statistics'):
    """Generates summary report"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SUMMARY REPORT - RESULTS ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("AVAILABLE DATA:")
    report_lines.append("-"*80)
    
    for file_type in datasets:
        report_lines.append(f"\nFile type: {file_type}")
        for variant in datasets[file_type]:
            for experiment in datasets[file_type][variant]:
                iterations = sorted(datasets[file_type][variant][experiment].keys())
                num_files = sum(len(datasets[file_type][variant][experiment][it]) 
                              for it in iterations)
                report_lines.append(
                    f"  • {variant:12} - {experiment:4}: "
                    f"iterations {iterations} ({num_files} files)"
                )
    
    report_lines.append("="*80)
    
    report_file = f"{output_dir}/summary_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\n✓ Report saved to: {report_file}")


if __name__ == "__main__":
    folder = "./"
    output_dir = "./statistics"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    episode_data = load_episode_files(folder)
    
    generate_summary_report(episode_data, output_dir)
    
    print("\n" + "="*80)
    print("GENERATING PLOTS AND STATISTICS")
    print("="*80 + "\n")
    
    # Line plots
    plot_metrics(episode_data, "Average Reward per Episode", 
                ["reward_mean"], plot_type='line', output_dir=output_dir, xlabel="Training Iteration", ylabel="Average Reward per Episode")
    
    plot_metrics(episode_data, "Number of Rescued Victims", 
                ["rescues_done"], plot_type='line', output_dir=output_dir, xlabel="Training Iteration", ylabel="Number of Rescued Victims")
    
    plot_metrics(episode_data, "Number of Rescued Victims (first 100)", 
                ["rescues_done"], plot_type='line', output_dir=output_dir, xlabel="Training Iteration", ylabel="Number of Rescued Victims")
    
    plot_metrics(episode_data, "Average Episode Length", 
                ["episode_len_mean"], plot_type='line', output_dir=output_dir, xlabel="Training Iteration", ylabel="Average Episode Length")
    
    plot_metrics(episode_data, "Normalized Reward Curve AUC", 
                ["auc_reward"], plot_type='line', output_dir=output_dir, xlabel="Training Iteration", ylabel="Normalized Reward Curve AUC")
    
    # Histograms
    plot_metrics(episode_data, "Density/Successful Clearing Actions", 
                ["clear_success"], plot_type='hist', output_dir=output_dir, xlabel="Successful Clearing Actions", ylabel="Frequency")
    plot_metrics(episode_data, "Density/Clearing Attempts", 
                ["clear_attempts"], plot_type='hist', output_dir=output_dir, xlabel="Clearing Attempts", ylabel="Frequency")
    plot_metrics(episode_data, "Density/Wall Collisions", 
                ["wall_hits"], plot_type='hist', output_dir=output_dir, xlabel="Wall Collisions", ylabel="Frequency")
    plot_metrics(episode_data, "Density/Action Withdrawals", 
                ["noop"], plot_type='hist', output_dir=output_dir, xlabel="Action Withdrawals", ylabel="Frequency")

    # Scatter plots
    plot_metrics(episode_data, "Removed Rubble/Clearing Attempts", 
                ["rubble_cleared"], plot_type='scatter', output_dir=output_dir, xlabel="Removed Rubble", ylabel="Clearing Attempts")
    plot_metrics(episode_data, "Rescued Victims/Removed Obstacles", 
                ["rescues_done"], plot_type='scatter', output_dir=output_dir, xlabel="Removed Obstacles", ylabel="Rescued Victims")


    rescue_metric = "Number of Rescued Victims"
    print(f"\n{'='*80}")
    print(f"AVERAGE RESCUED VICTIMS ({rescue_metric})")
    print(f"{'='*80}")

    for variant, experiment, base_label in PLOT_CONFIG[rescue_metric]:
        all_iteration_means = []
        
        for file_type in episode_data:
            if variant not in episode_data[file_type]:
                continue
            if experiment not in episode_data[file_type][variant]:
                continue
            
            iterations = sorted(episode_data[file_type][variant][experiment].keys())
            for iteration in iterations:
                dfs = episode_data[file_type][variant][experiment][iteration]
                for df in dfs:
                    if "rescues_done" in df.columns:
                        series = df["rescues_done"]
                        all_iteration_means.append(series.mean())
        
        if all_iteration_means:
            mean_rescued = np.nanmean(all_iteration_means)
            std_rescued = np.nanstd(all_iteration_means, ddof=1)
            print(f"{base_label:25}: Mean rescued = {mean_rescued:.2f}, Std = {std_rescued:.2f}")

    obstacle_metric = "Removed Rubble/Clearing Attempts"
    print(f"\n{'='*80}")
    print(f"AVERAGE OBSTACLE REMOVAL ({obstacle_metric})")
    print(f"{'='*80}")

    for variant, experiment, base_label in PLOT_CONFIG["Removed Rubble/Clearing Attempts"]:
        all_rubble = []
        all_attempts = []
        
        for file_type in episode_data:
            if variant not in episode_data[file_type]:
                continue
            if experiment not in episode_data[file_type][variant]:
                continue
            
            iterations = sorted(episode_data[file_type][variant][experiment].keys())
            for iteration in iterations:
                dfs = episode_data[file_type][variant][experiment][iteration]
                for df in dfs:
                    if "rubble_cleared" in df.columns and "clear_attempts" in df.columns:
                        all_rubble.extend(df["rubble_cleared"].dropna().tolist())
                        all_attempts.extend(df["clear_attempts"].dropna().tolist())
        
        if all_rubble and all_attempts:
            mean_rubble = np.nanmean(all_rubble)
            mean_attempts = np.nanmean(all_attempts)
            std_rubble = np.nanstd(all_rubble, ddof=1)
            std_attempts = np.nanstd(all_attempts, ddof=1)
            print(f"{base_label:25}: Mean obstacles cleared = {mean_rubble:.0f} "
                f"(Std = {std_rubble:.0f}), Mean attempts = {mean_attempts:.0f} (Std = {std_attempts:.0f})")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)
    print(f"All plots, statistics and tests saved in: {output_dir}")
    print("="*80)