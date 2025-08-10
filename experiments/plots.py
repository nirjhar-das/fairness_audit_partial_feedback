import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "serif"
# plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 18
# Turn on grid lines
plt.rcParams['axes.grid'] = True

# Everywhere
#plt.figure(figsize=(8, 6))
# Do all plotting stuff
#plt.xlabel("Top-$k$ Neighbors", fontsize=15)
#plt.ylabel("% Neighbors with the Same Gender", fontsize=15)
#plt.legend(fontsize=15)


import sys
taus = [5, 10, 25, 50, 100, 200, 400, 500, 600, 750, 900, 1000]
def compute_mean_and_variation(values):
    mean = np.mean(values, axis=0)  
    variation = np.std(values, axis=0)  
    return mean, variation

def extract_values_from_csv(file_path):
    df = pd.read_csv(file_path)
    eo_vals = df['eo_val_1'].values  
    m_vals = df['m_val'].values 
    costs = df['cost'].values
    return eo_vals, m_vals, costs

#def plot_comparison_with_threshold(eo_vals_one, m_vals_one, cost_one, eo_vals_two, m_vals_two, cost_two, label_one, label_two, i):
def plot_comparison_with_threshold(eo_vals_one, m_vals_one, cost_one, label_one):
    label_one = label_one.replace(" ", "")
    algo, dataset_name, model_name, policy_type = label_one.split('-')
    mean_eo_one, var_eo_one = compute_mean_and_variation(eo_vals_one)
    mean_m_one, var_m_one = compute_mean_and_variation(m_vals_one)
    
    # mean_eo_two, var_eo_two = compute_mean_and_variation(eo_vals_two)
    # mean_m_two, var_m_two = compute_mean_and_variation(m_vals_two)
    
    mean_cost_one, var_cost_one = compute_mean_and_variation(cost_one)
    #mean_cost_two, var_cost_two = compute_mean_and_variation(cost_two)
    mean_eo_two = mean_eo_one[13:]
    mean_eo_one = mean_eo_one[:13]
    var_eo_two = var_eo_one[13:]
    var_eo_one = var_eo_one[:13]
    mean_m_two = mean_m_one[13:]
    mean_m_one = mean_m_one[:13]
    var_m_two = var_m_one[13:]
    var_m_one = var_m_one[:13]
    mean_cost_two = mean_cost_one[13:]
    mean_cost_one = mean_cost_one[:13]
    var_cost_two = var_cost_one[13:]
    var_cost_one = var_cost_one[:13]
    
    mean_eo_one_threshold = mean_eo_one[-1]
    mean_eo_two_threshold = mean_eo_two[-1]
    
    
    
    plt.figure(figsize=(10, 8))
    plt.fill_between(mean_m_one[:-1], mean_eo_one[:-1] - var_eo_one[:-1], mean_eo_one[:-1] + var_eo_one[:-1], alpha=0.1, color='red')
    plt.fill_between(mean_m_two[:-1], mean_eo_two[:-1] - var_eo_two[:-1], mean_eo_two[:-1] + var_eo_two[:-1], alpha=0.1, color='black')
    plt.plot(mean_m_one[:-1], mean_eo_one[:-1], label='Baseline', color='red', linewidth=3, linestyle='-.')
    plt.plot(mean_m_two[:-1], mean_eo_two[:-1], label='RS Audit', color='black', linewidth=3, linestyle='--')
    plt.axhline(y=mean_eo_two_threshold, color='green', linestyle='-', label = 'True EO Value')
    plt.xlabel('Mean Number of Samples', fontsize=18)
    plt.ylabel('Average EO Difference', fontsize=18)
    plt.legend(fontsize=18)
    if policy_type == 'random' or policy_type == 'protected':
        plt.savefig(f"../figures/eo_val_{dataset_name}_{policy_type}.png")
    else:
        plt.savefig(f"../figures/eo_val_{dataset_name}_{model_name}_{policy_type}.png")            
    
    
    
    plt.figure(figsize=(10, 8))
    error_eo_one = np.abs(mean_eo_one[:-1] - mean_eo_one_threshold)
    error_eo_two = np.abs(mean_eo_two[:-1] - mean_eo_two_threshold)
    plt.plot(mean_m_one[:-1], error_eo_one, color='red', linestyle='-.', linewidth=4, label='Err. - Baseline')
    plt.plot(mean_m_two[:-1], error_eo_two, color='black', linestyle='--', linewidth=4, label='Err. - RS Audit')
    plt.xlabel('Mean Number of Samples', fontsize=18)
    plt.ylabel('Absolute Error wrt EO value', fontsize=18)
    plt.legend(fontsize=18)
    if policy_type == 'random' or policy_type == 'protected':
        plt.savefig(f"../figures/errors_{dataset_name}_{policy_type}.png")
    else:
        plt.savefig(f"../figures/errors_{dataset_name}_{model_name}_{policy_type}.png")  
    
    
    
    plt.figure(figsize=(10, 8))
    bar_width = 0.4
    x = np.arange(len(taus))
    std_m_one = np.sqrt(var_m_one[:-1])
    std_m_two = np.sqrt(var_m_two[:-1])
    plt.bar(x - bar_width/2, mean_m_one[:-1], yerr=std_m_one, width=bar_width, label='Baseline',
        color='red', capsize=5, ecolor='black', hatch='//')
    plt.bar(x + bar_width/2, mean_m_two[:-1], yerr=std_m_two, width=bar_width, label='RS-Audit',
        color='black', capsize=5, ecolor='black', hatch='-')
    plt.xlabel('Tau Value', fontsize=18)
    plt.ylabel('Mean Number of Samples Required for Auditing', fontsize=18)
    plt.xticks(ticks=x, labels=taus, rotation=45)
    plt.legend(fontsize=18)
    plt.tight_layout()
    if policy_type == 'random' or policy_type == 'protected':
        plt.savefig(f"../figures/tau_{dataset_name}_{policy_type}.png")
    else:
        plt.savefig(f"../figures/tau_{dataset_name}_{model_name}_{policy_type}.png")  
    
    
    plt.figure(figsize=(10,8))
    plt.fill_between(taus, mean_cost_one[:-1] - var_cost_one[:-1], mean_cost_one[:-1] + var_cost_one[:-1], alpha=0.1, color='red')
    plt.fill_between(taus, mean_cost_two[:-1] - var_cost_two[:-1], mean_cost_two[:-1] + var_cost_two[:-1], alpha=0.1, color='black')
    plt.plot(taus, mean_cost_one[:-1], label='Baseline', color='red', linewidth=3, linestyle='-.')
    plt.plot(taus, mean_cost_two[:-1], label='RS-Audit', color='black', linewidth=3, linestyle='--')
    plt.xlabel('Tau (Stopping threshold)', fontsize=18)
    plt.ylabel('Average Cost', fontsize=18)
    plt.legend(fontsize=18)
    if policy_type == 'random' or policy_type == 'protected':
        plt.savefig(f"../figures/cost_{dataset_name}_{policy_type}.png")
    else:
        plt.savefig(f"../figures/cost_{dataset_name}_{model_name}_{policy_type}.png")            
    
root_dir = '../results' 
algorithm_options = ['algorithm_one']
domain_options = ['adult', 'law']
model_options = ['GNB', 'LR']
data_options = ['full_data', 'wo_proattr', 'fair']
try:
    os.mkdir('../figures')
except:
    pass

i = 0
for algorithm in algorithm_options:
    for domain in domain_options:
        for model in model_options:
            for data in data_options:
                dir_path = os.path.join(root_dir, algorithm, domain, model, data)
                
                csv_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]
                #csv_files = [os.path.join(dir_path, '20250731-094400.csv')] # Because we only want the latest file
                
                eo_vals = []
                m_vals = []
                costs = []
                for csv_file in csv_files:
                    eo, m, cost = extract_values_from_csv(csv_file)
                    eo_vals.append(eo)
                    m_vals.append(m)
                    costs.append(cost)
                eo_vals = np.array(eo_vals)
                m_vals = np.array(m_vals)
                costs = np.array(costs)

                # other_algorithm = 'algorithm_two' if algorithm == 'algorithm_one' else 'algorithm_one'
                # other_dir_path = os.path.join(root_dir, other_algorithm, domain, model, data)
                # other_csv_files = [os.path.join(other_dir_path, f) for f in os.listdir(other_dir_path) if f.endswith('.csv')]
                
                # other_eo_vals = []
                # other_m_vals = []
                # other_costs = []
                
                # for csv_file in other_csv_files:
                #     eo, m, cost = extract_values_from_csv(csv_file)
                #     other_eo_vals.append(eo)
                #     other_m_vals.append(m)
                #     other_costs.append(cost)
                    
                # other_eo_vals = np.array(other_eo_vals)
                # other_m_vals = np.array(other_m_vals)
                # other_costs = np.array(other_costs)

                # plot_comparison_with_threshold(eo_vals, m_vals, costs, other_eo_vals, other_m_vals, other_costs,
                #                             f'{algorithm} - {domain} - {model} - {data}', 
                #                             f'{other_algorithm} - {domain} - {model} - {data}', i)
                plot_comparison_with_threshold(eo_vals, m_vals, costs,
                                            f'{algorithm} - {domain} - {model} - {data}')
                i += 1
    break

model_options = ['LR']
data_options = [ 'random', 'protected']
i = 0
for algorithm in algorithm_options:
    for domain in domain_options:
        for model in model_options:
            for data in data_options:
                dir_path = os.path.join(root_dir, algorithm, domain, model, data)
                
                csv_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]
                
                eo_vals = []
                m_vals = []
                costs = []
                
                for csv_file in csv_files:
                    eo, m, cost = extract_values_from_csv(csv_file)
                    eo_vals.append(eo)
                    m_vals.append(m)
                    costs.append(cost)
                eo_vals = np.array(eo_vals)
                m_vals = np.array(m_vals)
                costs = np.array(costs)

                # other_algorithm = 'algorithm_two' if algorithm == 'algorithm_one' else 'algorithm_one'
                # other_dir_path = os.path.join(root_dir, other_algorithm, domain, model, data)
                # other_csv_files = [os.path.join(other_dir_path, f) for f in os.listdir(other_dir_path) if f.endswith('.csv')]
                
                # other_eo_vals = []
                # other_m_vals = []
                # other_costs = []
                
                # for csv_file in other_csv_files:
                #     eo, m, cost = extract_values_from_csv(csv_file)
                #     other_eo_vals.append(eo)
                #     other_m_vals.append(m)
                #     other_costs.append(cost)
                
                # other_eo_vals = np.array(other_eo_vals)
                # other_m_vals = np.array(other_m_vals)
                # other_costs = np.array(other_costs)

                # plot_comparison_with_threshold(eo_vals, m_vals, costs, other_eo_vals, other_m_vals, other_costs,
                #                             f'{algorithm} - {domain} - {model} - {data}', 
                #                             f'{other_algorithm} - {domain} - {model} - {data}', i)
                plot_comparison_with_threshold(eo_vals, m_vals, costs,
                                            f'{algorithm} - {domain} - {model} - {data}')
                i += 1
    break