import pandas as pd
import numpy as np
import random 
import time
import pickle
from fairlearn.metrics import equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
#from ucimlrepo import fetch_ucirepo 
import sklearn 
import sklearn.cluster
from collections import Counter
import argparse
import os
import sys
import csv
import warnings
from folktables import ACSDataSource, ACSEmployment
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from scipy.special import lambertw
from imblearn.over_sampling import SMOTE
import time
# Seeds used for different runs:
# 1029, 42, 13, 729, 333, 7, 222, 86, 1500, 17
rng_classifier = np.random.default_rng(seed=1) # To be kept constant


def preprocess_data(df, target_col, categorical_cols, scale_numerical=False, from_NB = False, dataset = None):
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df.drop(columns=[target_col])
    if from_NB:
        y = df['f_0']
    else:
        y = df[target_col]
    if scale_numerical:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, y, label_encoders


def naive_bayes_pipeline(df, target_col, categorical_cols, dataset, from_NB = True, fair=False, protected_attribute = None):
    X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=from_NB)
    # Because we dont care how f is trained, we dont care about stratified splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if not fair:
        model = GaussianNB()
        model.fit(X_train, y_train)
    else:
        A = X_train[protected_attribute]
        gnb = GaussianNB()
        model = ExponentiatedGradient(
            estimator=gnb,
            constraints=EqualizedOdds()
        )
        model.fit(X_train, y_train, sensitive_features=A)
    return model


def clf_pipeline(df, target_col, categorical_cols, dataset, from_NB = True, fair=False, protected_attribute = None, random_forest = False, resample=False):
    if resample:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                pd.Series(y_resampled, name=target_col)], axis=1)
    X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=from_NB, scale_numerical=True)
    # Because we dont care how f is trained, we dont care about stratified splits
    try:
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, df[protected_attribute], test_size=0.2, random_state=42, stratify=y)
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)        
    if not fair:
        if random_forest:
            model = RandomForestClassifier()        
        else:
            if resample:
                w = {0:99, 1:1}
                model = LogisticRegression(solver='liblinear', class_weight=w)
            else:
                model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # s = model.predict_proba(X_test)[:,1]
        # bins = np.linspace(min(s), max(s), 10)
        # print(sklearn.metrics.confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        # # print(classification_report(y_test, model.predict(X_test)))
        # counts, bin_edges = np.histogram(s, bins=bins)
        # for i in range(len(counts)):
        #     bin_range = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        #     bar = '#' * (counts[i] // 100)  # adjust the divider to control bar width
        #     print(f"{bin_range}: {bar} ({counts[i]})")
        
        
    else:
        try:
            A = X_train[protected_attribute]
        except:
            A = A_train
        lr = LogisticRegression()
        model = ExponentiatedGradient(
            estimator=lr,
            constraints=EqualizedOdds()
        )
        model.fit(X_train, y_train, sensitive_features=A)
    return model


def get_f_0(policy, dataset, df, protected_attribute, train, model, target_col = None, categorical_cols = None):
    if policy == 'full_data':
        if train == 'True':
            if model == 'GNB':
                print("Training GNB..")
                indices = rng_classifier.integers(0, len(df), 100)
                df_sub = df.iloc[indices]
                model_ = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False)
            elif model == 'LR':
                print("Training LR..")
                if dataset == 'law':
                    indices = rng_classifier.integers(0, len(df), 5000)
                    df_sub = df.iloc[indices]
                    model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False, resample=True)
                else:
                    indices = rng_classifier.integers(0, len(df), 100)
                    df_sub = df.iloc[indices]
                    model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False)                            
        # df = df.drop(indices)
        X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
        df['f_0'] = model_.predict(X)
    elif policy == 'wo_proattr':
        if dataset == 'adult':
            categorical_cols.remove(protected_attribute)
        if train == 'True':
            if model == 'GNB':
                print("Training GNB..")
                indices = rng_classifier.integers(0, len(df), 100)
                df_sub = df.iloc[indices]
                df_sub = df_sub.drop(protected_attribute, axis=1)
                model_ = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False)
            elif model == 'LR':
                print("Training LR..")
                if dataset == 'law':
                    indices = rng_classifier.integers(0, len(df), 5000)
                    df_sub = df.iloc[indices]
                    df_sub = df_sub.drop(protected_attribute, axis=1)
                    model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False, resample=True)
                else:
                    indices = rng_classifier.integers(0, len(df), 100)
                    df_sub = df.iloc[indices]
                    df_sub = df_sub.drop(protected_attribute, axis=1)
                    model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False)                            
        # df = df.drop(indices)
        df_sub = df.drop(protected_attribute, axis=1)
        X, y, _ = preprocess_data(df_sub, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
        df['f_0'] = model_.predict(X)
    elif policy == 'random':
        df['f_0'] = rng_classifier.integers(0, 2, len(df))
    elif policy == 'fair':
        if train == 'True':
            if model == 'GNB':
                print("Training GNB..")
                indices = rng_classifier.integers(0, len(df), 100)
                df_sub = df.iloc[indices]
                model_ = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False, fair=True, protected_attribute=protected_attribute)
            elif model == 'LR':
                print("Training LR..")
                if dataset == 'law':
                    indices = rng_classifier.integers(0, len(df), 5000)
                    df_sub = df.iloc[indices]
                    model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False, resample=True)
                else:
                    indices = rng_classifier.integers(0, len(df), 100)
                    df_sub = df.iloc[indices]
                    model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False)                            
        # df = df.drop(indices)
        X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
        df['f_0'] = model_.predict(X)    
    else:
        df['f_0'] = df[protected_attribute]
    df = df.reset_index(drop=True)
    return df

def main(args):
    # Set the args
    dataset = args.dataset
    policy = args.policy
    alpha = float(args.alpha)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    algorithm = args.algorithm
    train = args.train
    model = args.model
    # Load Datasets
    if dataset == 'adult':
        columns = [ "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]        
        df = pd.read_csv("../data/adult/adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)
        df.dropna(inplace=True)  
        target_col = "income"
        categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
        A = 'sex' # protected attribute (other option 'Race')
        df[target_col] = np.where(df[target_col] == '>50K', 1, 0)
        df[A] = np.where(df[A] == 'Male', 1, 0)
            
    elif dataset == 'law':
        df = pd.read_csv('../data/law_school/law_school_clean.csv')
        df.male = df.male.astype(int)
        df.pass_bar = df.pass_bar.astype(int)
        target_col = "pass_bar"
        categorical_cols = ["fam_inc", "tier", "race"]
        A = 'male' # protected attribute
    
    try:
        os.makedirs(f"../results/algorithm_{algorithm}/{dataset}/{model}/{policy}") 
        os.makedirs(f"../models") 
    except:
        pass

    file_exists = os.path.exists(f"../results/algorithm_{algorithm}/{dataset}/{model}/{policy}/{timestr}.csv")
    with open(f"../results/algorithm_{algorithm}/{dataset}/{model}/{policy}/{timestr}_{args.seed}.csv", mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists or os.path.getsize(f"../results/algorithm_{algorithm}/{dataset}/{model}/{policy}/{timestr}.csv") == 0:
                writer.writerow(['dataset', 'algorithm', 'f_0',  'm_val', 'cost', 'eo_val_1'])
            df = get_f_0(policy, dataset, df, A, train, model, target_col=target_col, categorical_cols=categorical_cols)
            if algorithm == 'one':
                p_00 = df[(df['f_0'] == 1) & (df[target_col] == 0) & (df[A] == 0)].shape[0] / len(df)
                p_01 = df[(df['f_0'] == 1) & (df[target_col] == 0) & (df[A] == 1)].shape[0] / len(df)    
                p_10 = df[(df['f_0'] == 1) & (df[target_col] == 1) & (df[A] == 0)].shape[0] / len(df)
                p_11 = df[(df['f_0'] == 1) & (df[target_col] == 1) & (df[A] == 1)].shape[0] / len(df) 
                costs = []
                for tau in [5, 10, 25, 50, 100, 200, 400, 500, 600, 750, 900, 1000]: # Random Choices of Tau (could have instead used tau(eps, delta))
                    cost = 0
                    q_ya = []
                    for y in [0,1]: # Why not y=0, we should do that too
                        concatenated_samples_a = []
                        for a in [0,1]:
                            concatenated_samples = 0
                            m = tau
                            ##### Implementation of MultiEst, will return m/len(concat_samples)
                            while(m > 0):
                                index = rng.integers(0,len(df), 1) # Must be Sampling without replacement, should be outside the loop
                                sampled_row = df.iloc[index]
                                concatenated_samples += 1   
                                if (y == sampled_row[target_col].values[0]) and (a == sampled_row[A].values[0]):
                                    m -= 1
                                if sampled_row[target_col].values[0] == 0 and sampled_row['f_0'].values[0]==0:
                                    cost += 1
                            #####
                            # Since we have already calculate q_hat = tau/len(concatenated_samples), we can use that to calculate the eo_diff, all we need is p_hat
                            q_ya.append(tau/concatenated_samples)
                            concatenated_samples_a.append(concatenated_samples)
                    q_00, q_01 = q_ya[0], q_ya[1]
                    q_10, q_11 = q_ya[2], q_ya[3]
                    total_samples = sum(concatenated_samples_a)
                    fin_val = max(np.abs((p_00/q_00) - (p_01/q_01)), np.abs((p_10/q_10) - (p_11/q_11)))
                    writer.writerow([dataset, 'Algorithm One', policy, total_samples, cost, fin_val])
                    costs.append(cost)
                q_10 = df[(df[target_col] == 1) & (df[A] == 0)].shape[0] / len(df)
                q_11 = df[(df[target_col] == 1) & (df[A] == 1)].shape[0] / len(df)
                q_00 = df[(df[target_col] == 0) & (df[A] == 0)].shape[0] / len(df)
                q_01 = df[(df[target_col] == 0) & (df[A] == 1)].shape[0] / len(df)
                fin_val = max(np.abs((p_00/q_00) - (p_01/q_01)), np.abs((p_10/q_10) - (p_11/q_11)))
                writer.writerow([dataset, 'Algorithm One Final Value', policy, len(df), np.sum(costs), fin_val])
                    
                df_y0 = df[df[A]==0]
                df_y1 = df[df[A]==1]
                p_00 = df_y0[(df_y0['f_0'] == 1) & (df_y0[target_col] == 0)].shape[0] / len(df_y0)
                p_01 = df_y1[(df_y1['f_0'] == 1) & (df_y1[target_col] == 0)].shape[0] / len(df_y1)    
                p_10 = df_y0[(df_y0['f_0'] == 1) & (df_y0[target_col] == 1)].shape[0] / len(df_y0)
                p_11 = df_y1[(df_y1['f_0'] == 1) & (df_y1[target_col] == 1)].shape[0] / len(df_y1)    
                costs = []
                for tau in [5, 10, 25, 50, 100, 200, 400, 500, 600, 750, 900, 1000]:
                    s = 0
                    cost = 0
                    q_ya = []
                    for y in [0,1]:
                        concatenated_samples_a = []
                        for a in [0,1]:
                            df_ya = df.loc[df[A]==a]
                            df_ya = df_ya.reset_index(drop=True)    
                            m = tau
                            concatenated_samples = 0
                            while(m > 0):
                                index = rng.integers(0,len(df_ya), 1)
                                sampled_row = df_ya.iloc[index]
                                concatenated_samples += 1
                                if y == sampled_row[target_col].values[0]:  
                                    m -= 1
                                if sampled_row[target_col].values[0] == 0 and sampled_row['f_0'].values[0]==0:
                                    cost += 1
                            q_ya.append(tau/concatenated_samples)
                            concatenated_samples_a.append(concatenated_samples)
                    q_00, q_01 = q_ya[0], q_ya[1]
                    q_10, q_11 = q_ya[2], q_ya[3]
                    s += sum(concatenated_samples_a)
                    eo_diff_1 = max(np.abs((p_00/q_00) - (p_01/q_01)), np.abs((p_10/q_10) - (p_11/q_11)))
                    writer.writerow([dataset, 'Algorithm Two', policy, s, cost, eo_diff_1])
                    costs.append(cost)
                q_00 = df_y0[(df_y0[target_col] == 0)].shape[0] / len(df_y0)
                q_01 = df_y1[(df_y1[target_col] == 0)].shape[0] / len(df_y1)
                q_10 = df_y0[(df_y0[target_col] == 1)].shape[0] / len(df_y0)
                q_11 = df_y1[(df_y1[target_col] == 1)].shape[0] / len(df_y1)
                eo_diff_1 = max(np.abs((p_00/q_00) - (p_01/q_01)), np.abs((p_10/q_10) - (p_11/q_11)))
                writer.writerow([dataset, 'Algorithm Two Final Value', policy, len(df), np.sum(costs), eo_diff_1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Testing Fairness.')
    parser.add_argument('--dataset', type=str, help="Name of dataset: adult, law, newadult", default='adult')
    parser.add_argument('--policy', type=str, help="Name of audit policy: full_data, wo_proattr, random, protected, fair", default='full_data')
    parser.add_argument('--model', type=str, help="Whether to train LR or GNB", default='LR')
    parser.add_argument('--algorithm', type=str, help="Name of algorithm: one", default='one')
    parser.add_argument('--alpha', type=str, help="Alternate Hypothesis Constant, alpha", default='0.10')
    parser.add_argument('--train', type=str, help="Whether to train or load saved models", default='True')
    parser.add_argument('--seed', type=int, help="Random seed for reproducibility", default=1029)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed=args.seed)
    main(args)