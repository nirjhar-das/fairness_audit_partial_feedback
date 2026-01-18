# debug changes
# - 2 samples
# - 250 iters in cp.SCS

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch

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
import sys
import csv
import warnings
from folktables import ACSDataSource, ACSEmployment
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from scipy.special import lambertw
#from imblearn.over_sampling import SMOTE
import time
import copy
# Seeds used for different runs:
# 1029, 42, 13, 729, 333, 7, 222, 86, 1500, 17
rng = np.random.default_rng(seed=1029)
rng_classifier = np.random.default_rng(seed=1) # To be kept constant

# Set global seed
np.random.seed(42)
sklearn.utils.check_random_state(42)

def preprocess_data(df, target_col, scale_numerical=True, from_NB = False, scaler=None):
    X = df.drop(columns=[target_col])
    if from_NB:
        y = df['f_0']
    else:
        y = df[target_col]
    if scale_numerical:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
    return X, y, scaler

def preprocess_real_data(df, target_col, categorical_cols, sensitive_column, scale_numerical=False, from_NB = False, dataset = None):
    # label_encoders = {}
    # for col in categorical_cols:
    #     le = OneHotEncoder(handle_unknown='ignore')
    #     df[col] = le.fit_transform(df[col])
    #     label_encoders[col] = le
    # X = df.drop(columns=[target_col])
    # if from_NB:
    #     y = df['f_0']
    # else:
    #     y = df[target_col]
    # if scale_numerical:
    #     scaler = StandardScaler()
    #     X = scaler.fit_transform(X)
    # One hot encode categorical variables
    numerical_cols = list(set(list(df.columns)) - set(categorical_cols) - set([target_col]))
    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), list(set(categorical_cols) - set([sensitive_column])))
        ])
    X = preprocessor.fit_transform(df.drop(columns=[target_col, sensitive_column]))
    if from_NB:
        y = df['f_0']
    else:
        y = df[target_col]
    label_encoders = preprocessor

    # Convert sparse matrix to dense array first, then to DataFrame
    if hasattr(X, 'toarray'):  # Check if it's a sparse matrix
        X = X.toarray()
    
    if isinstance(X, np.ndarray):
        # Get feature names from the preprocessor
        feature_names = preprocessor.get_feature_names_out()
        X = pd.DataFrame(X, columns=feature_names)
        # match the label index
        X.index = df.index
        # Add sensitive column back
        X[sensitive_column] = df[sensitive_column]

    return X, y, label_encoders

def naive_bayes_pipeline(df, target_col, dataset, from_NB = True, fair=False, protected_attribute = None):
    X, y, scaler = preprocess_data(df, target_col=target_col, dataset=dataset, from_NB=from_NB)
    # Because we dont care how f is trained, we dont care about stratified splits
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X
    y_train = y
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

def clf_pipeline(df, target_col, from_NB = True, fair=False, protected_attribute = None, random_forest = False, resample=False, test_df=None):
    if resample:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                pd.Series(y_resampled, name=target_col)], axis=1)
    #X, y, scaler = preprocess_data(df, target_col=target_col, from_NB=from_NB, scale_numerical=True)
    if test_df is not None:
        #X_test, y_test, _ = preprocess_data(test_df, target_col=target_col, from_NB=from_NB, scale_numerical=True, scaler=scaler)
        a_test = test_df[protected_attribute]
    # Because we dont care how f is trained, we dont care about stratified splits
    # try:
    #     X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, df[protected_attribute], test_size=0.2, random_state=42, stratify=y)
    # except:
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]
    A_train = df[protected_attribute]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    a_test = test_df[protected_attribute]
    if not fair:
        if random_forest:
            model = RandomForestClassifier()        
        else:
            if resample:
                w = {0:99, 1:1}
                model = LogisticRegression(solver='liblinear', class_weight=w)
            else:
                model = LogisticRegression()
        model.fit(X_train, y_train)
        
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
    # Check positivity rates
    preds = model.predict(X_test)
    for a in [0,1]:
        idx_a = (a_test == a)
        idx_y = (y_test == 0)
        pr = np.sum(preds[idx_a & idx_y]) / len(preds[idx_a & idx_y])
        true_pr = np.sum(y_test[idx_a]) / len(y_test[idx_a])
        print(f'Positivity rate for group {a}: {pr}, True positivity rate: {true_pr}')
    return model

def get_unfairness(df, A='sensitive', target_col='label'):
    df_y0 = df.loc[df[A]==0]
    df_y1 = df.loc[df[A]==1]
    
    p_00 = df_y0[(df_y0['f_0'] == 1) & (df_y0[target_col] == 0)].shape[0] / len(df_y0)
    p_01 = df_y1[(df_y1['f_0'] == 1) & (df_y1[target_col] == 0)].shape[0] / len(df_y1)
    q_00 = df_y0[(df_y0[target_col] == 0)].shape[0] / len(df_y0)
    q_01 = df_y1[(df_y1[target_col] == 0)].shape[0] / len(df_y1)
    
    p_10 = df_y0[(df_y0['f_0'] == 1) & (df_y0[target_col] == 1)].shape[0] / len(df_y0)
    p_11 = df_y1[(df_y1['f_0'] == 1) & (df_y1[target_col] == 1)].shape[0] / len(df_y1)
    q_10 = df_y0[(df_y0[target_col] == 1)].shape[0] / len(df_y0)
    q_11 = df_y1[(df_y1[target_col] == 1)].shape[0] / len(df_y1)
    
    return max(np.abs((p_00/q_00) - (p_01/q_01)), np.abs((p_10/q_10) - (p_11/q_11))) 

def get_f_0(policy, df, protected_attribute, train, model, target_col = None, test_df=None, df_f_0=None, given_f_0=None):
    if policy == 'full_data':
        if train == 'True':
            if model == 'GNB':
                print("Training GNB..")
                indices = rng_classifier.integers(0, len(df), 1000)
                df_sub = df.iloc[indices]
                model_ = naive_bayes_pipeline(df_sub, target_col, from_NB = False)
            elif model == 'LR':
                print("Training LR..")
                indices = rng_classifier.integers(0, len(df), 1000)
                #df_sub = df_f_0
                df_sub = df.iloc[indices]
                model_ = clf_pipeline(df_sub, target_col, from_NB=False, protected_attribute=protected_attribute, test_df=test_df)
        else:
            model_ = given_f_0
        # df = df.drop(indices)
        # Partial feedback model uses a subset of the training data
        X, y, scaler = preprocess_data(df, target_col=target_col, from_NB=False)
        X_test, y_test, _ = preprocess_data(test_df, target_col=target_col, from_NB=False, scaler=scaler)
        #if given_f_0 is None:
        df['f_0'] = model_.predict(X)
        test_df['f_0'] = model_.predict(X_test)
        #unfairness = equalized_odds_difference(y_test, test_df['f_0'], sensitive_features=test_df[protected_attribute])
        unfairness = get_unfairness(test_df, A=protected_attribute, target_col=target_col)
        # else:
        #     df['f_0'] = given_f_0.predict(X)
        #     test_df['f_0'] = given_f_0.predict(X_test)
        #     unfairness = get_unfairness(test_df, A=protected_attribute, target_col=target_col)
    elif policy == 'random':
        df['f_0'] = rng_classifier.integers(0, 2, len(df))
        test_df['f_0'] = rng_classifier.integers(0, 2, len(test_df))
        #unfairness = equalized_odds_difference(y_test, test_df['f_0'], sensitive_features=test_df[protected_attribute])
        unfairness = get_unfairness(test_df, A=protected_attribute, target_col=target_col)
    # elif policy == 'fair':
    #     if train == 'True':
    #         if model == 'GNB':
    #             print("Training GNB..")
    #             indices = rng_classifier.integers(0, len(df), 100)
    #             df_sub = df.iloc[indices]
    #             model_ = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False, fair=True, protected_attribute=protected_attribute)
    #         elif model == 'LR':
    #             print("Training LR..")
    #             if dataset == 'law':
    #                 indices = rng_classifier.integers(0, len(df), 5000)
    #                 df_sub = df.iloc[indices]
    #                 model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False, resample=True)
    #             else:
    #                 indices = rng_classifier.integers(0, len(df), 100)
    #                 df_sub = df.iloc[indices]
    #                 model_ = clf_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB=False)                            
    #     # df = df.drop(indices)
    #     X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
    #     df['f_0'] = model_.predict(X)    
    else:
        df['f_0'] = df[protected_attribute]
        test_df['f_0'] = test_df[protected_attribute]
        #unfairness = equalized_odds_difference(y_test, test_df['f_0'], sensitive_features=test_df[protected_attribute])
        unfairness = get_unfairness(test_df, A=protected_attribute, target_col=target_col)
    # Reset and drop previous index
    df = df.reset_index(drop=True)
    # Get the positive rate of f_0 for every subgroup
    alpha = {}
    for y in [0,1]:
        for a in [0,1]:
            alpha[f'{y}{a}'] = df[(df['f_0'] == 1) & (df['sensitive'] == a) & (df['label'] == y)].shape[0] / df[(df['sensitive'] == a) & (df['label'] == y)].shape[0]
    return df, model_, alpha, unfairness

# Zeng et al. synthetic data
def generate_examples(num_examples=1000, dim=100, var=1, means=None, a_1_prob=0.7, y_1_a_1_prob=0.7, y_1_a_0_prob=0.4):
    num_a_1 = 0
    num_a_0 = 0
    for i in range(num_examples):
        if np.random.uniform(0,1) < a_1_prob:
            num_a_1 += 1
        else:
            num_a_0 += 1
    num_y_1_a_1 = 0
    num_y_0_a_1 = 0
    for i in range(num_a_1):
        if np.random.uniform(0,1) < y_1_a_1_prob:
            num_y_1_a_1 += 1
        else:
            num_y_0_a_1 += 1
    num_y_1_a_0 = 0
    num_y_0_a_0 = 0
    for i in range(num_a_0):
        if np.random.uniform(0,1) < y_1_a_0_prob:
            num_y_1_a_0 += 1
        else:
            num_y_0_a_0 += 1
    covar = var*np.eye(dim) # Spherical gaussian for now
    if means == None:
        mu_0_0 = [np.random.uniform(-1,1) for i in range(dim)]
        # Ensure separation between mu_0_0 and mu_1_0
        def generate_sep(var, eps=0.001):
            candidate = [np.random.uniform(-1,1) for i in range(dim)]
            candidate = candidate/np.linalg.norm(candidate)
            return var*(np.sqrt(0.5*np.log(50/eps)))*candidate
        #mu_1_0 = [np.random.uniform(0,1) for i in range(dim)]
        mu_1_0 = generate_sep(var) + mu_0_0

        mu_0_1 = [np.random.uniform(-1,1) for i in range(dim)]
        #mu_1_1 = [np.random.uniform(0,1) for i in range(dim)]
        mu_1_1 = generate_sep(var) + mu_0_1
    else:
        mu_0_0 = means[0]
        mu_0_1 = means[1]
        mu_1_0 = means[2]
        mu_1_1 = means[3]
    # Samples
    samples_0_0 = np.random.multivariate_normal(mu_0_0, covar, num_y_0_a_0)
    samples_0_1 = np.random.multivariate_normal(mu_0_1, covar, num_y_0_a_1)
    samples_1_0 = np.random.multivariate_normal(mu_1_0, covar, num_y_1_a_0)
    samples_1_1 = np.random.multivariate_normal(mu_1_1, covar, num_y_1_a_1)
    sensitive = np.asarray([0]*len(samples_0_0) + [1]*len(samples_0_1) + [0]*len(samples_1_0) + [1]*len(samples_1_1))
    label = np.asarray([0]*len(samples_0_0) + [0]*len(samples_0_1) + [1]*len(samples_1_0) + [1]*len(samples_1_1))
    return [mu_0_0, mu_0_1, mu_1_0, mu_1_1], np.vstack((samples_0_0, samples_0_1, samples_1_0, samples_1_1)), sensitive, label

# Gaussianized representations of real data
def generate_real_data_and_model(dataset='adult', seed=42):
    # Load Datasets
    if dataset == 'adult':
        columns = [ "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]        
        df = pd.read_csv("../data/adult/adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)
        df_test = pd.read_csv("../data/adult/adult.test", header=None, names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)
        # Concat train and test for preprocessing
        df = pd.concat([df, df_test], ignore_index=True)
        df.dropna(inplace=True)  
        target_col = "income"
        categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
        A = 'sex' # protected attribute (other option 'Race')
        # Drop other senstitive attributes
        df[target_col] = np.where(df[target_col] == '>50K', 1, 0)
        df[A] = np.where(df[A] == 'Male', 1, 0)
        # Print subgroup stats
        subgroups = {
            '00': (df[A] == 0) & (df[target_col] == 0),
            '01': (df[A] == 1) & (df[target_col] == 0),
            '10': (df[A] == 0) & (df[target_col] == 1),
            '11': (df[A] == 1) & (df[target_col] == 1),
        }
        print("Subgroup distribution:", {key: df[subgroups[key]].shape[0] for key in subgroups})
            
    elif dataset == 'law':
        df = pd.read_csv('../data/law_school/law_school_clean.csv')
        df.male = df.male.astype(int)
        df.pass_bar = df.pass_bar.astype(int)
        target_col = "pass_bar"
        categorical_cols = ["fam_inc", "tier", "race"]
        A = 'male' # protected attribute
    
    X, y, _ = preprocess_real_data(df, target_col=target_col, categorical_cols=categorical_cols, sensitive_column=A, dataset=dataset, from_NB=False)
    
    from tabpfn_extensions import TabPFNClassifier
    from tabpfn_extensions.embedding import TabPFNEmbedding
    from sklearn.model_selection import train_test_split

    from fairlearn.metrics import equalized_odds_difference, true_positive_rate_difference
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    # subgroups = {
    #     '00': (df[A] == 0) & (df[target_col] == 0),
    #     '01': (df[A] == 1) & (df[target_col] == 0),
    #     '10': (df[A] == 0) & (df[target_col] == 1),
    #     '11': (df[A] == 1) & (df[target_col] == 1),
    # }
    # # Shuffle all subgroups
    # subgroups['00'] = df[subgroups['00']].sample(frac=1, random_state=seed).index
    # subgroups['01'] = df[subgroups['01']].sample(frac=1, random_state=seed).index
    # subgroups['10'] = df[subgroups['10']].sample(frac=1, random_state=seed).index
    # subgroups['11'] = df[subgroups['11']].sample(frac=1, random_state=seed).index
    # train_indices = []
    # test_indices = []
    # for key in subgroups:
    #     train_indices.extend(subgroups[key][:int(0.7*len(subgroups[key]))])
    #     test_indices.extend(subgroups[key][int(0.7*len(subgroups[key])):])
    # X_train = pd.DataFrame(X).iloc[train_indices]
    # X_test = pd.DataFrame(X).iloc[test_indices]
    # y_train = pd.Series(y).iloc[train_indices]
    # y_test = pd.Series(y).iloc[test_indices]

    #y_train = X_train.pop(target_col)
    a_train = X_train.pop(A)

    #y_test = X_test.pop(target_col)
    a_test = X_test.pop(A)

    # Convert everything to numpy
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    a_train = a_train.values
    a_test = a_test.values

    #baseline_model = LogisticRegression().fit(np.concatenate((X_train, a_train.reshape(-1,1)), axis=1), y_train)
    #preds = baseline_model.predict(np.concatenate((X_test, a_test.reshape(-1,1)), axis=1))

    # group blind
    baseline_model = LogisticRegression().fit(X_train, y_train)
    preds = baseline_model.predict(X_test)

    #unf = equalized_odds_difference(y_test, preds, sensitive_features=a_test)
    unf = true_positive_rate_difference(y_test, preds, sensitive_features=a_test)
    print("Baseline Model Performance:", accuracy_score(y_test, preds), "Unfairness:", unf)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, preds))
    # # Positivity rate for each group
    # for a in [0,1]:
    #     idx = (a_test == a)
    #     pr = np.sum(preds[idx]) / len(preds[idx])
    #     true_pr = np.sum(y_test[idx]) / len(y_test[idx])
    #     print(f'Positivity rate for group {a}: {pr}, True positivity rate: {true_pr}')

    # Train and evaluate TabPFN embeddings (vanilla)
    if not os.path.exists(f'{dataset}_{seed}_embeddings.pt'):
        print('Generating TabPFN embeddings...')
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="train",
        )[0]

        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )[0]
        torch.save((train_embeddings, test_embeddings), f'{dataset}_{seed}_embeddings.pt')
    else:
        print('Loading precomputed TabPFN embeddings...')
        train_embeddings, test_embeddings = torch.load(f'{dataset}_{seed}_embeddings.pt', weights_only=False)

    print('Obtained embeddings of shape:', train_embeddings.shape, test_embeddings.shape)

    emb_model = LogisticRegression().fit(np.concatenate((train_embeddings, a_train.reshape(-1,1)), axis=1), y_train)
    emb_preds = emb_model.predict(np.concatenate((test_embeddings, a_test.reshape(-1,1)), axis=1))
    emb_unf = equalized_odds_difference(y_test, emb_preds, sensitive_features=a_test)
    #emb_unf = true_positive_rate_difference(y_test, emb_preds, sensitive_features=a_test)
    print("TabPFN Embedding Model Performance:", accuracy_score(y_test, emb_preds), "Unfairness:", emb_unf)
    # Positivity rate for each group
    for a in [0,1]:
        idx = (a_test == a)
        idx_y = (y_test == 0)
        pr = np.sum(emb_preds[idx & idx_y]) / np.sum(idx & idx_y)
        true_pr = np.sum(y_test[idx]) / len(y_test[idx])
        print(f'Positivity rate at source for group {a}: {pr}, True positivity rate: {true_pr}')
    print(classification_report(y_test, emb_preds))
    
    #return [mu_0_0, mu_0_1, mu_1_0, mu_1_1], np.vstack((samples_0_0, samples_0_1, samples_1_0, samples_1_1)), sensitive, label
    # Split data into subgroups
    features_00 = train_embeddings[(a_train == 0) & (y_train == 0)]
    features_01 = train_embeddings[(a_train == 1) & (y_train == 0)]
    features_10 = train_embeddings[(a_train == 0) & (y_train == 1)]
    features_11 = train_embeddings[(a_train == 1) & (y_train == 1)]

    mu_0_0 = np.mean(features_00, axis=0)
    mu_0_1 = np.mean(features_01, axis=0)
    mu_1_0 = np.mean(features_10, axis=0)
    mu_1_1 = np.mean(features_11, axis=0)

    # Estimate diagonal covariance
    var_0_0 = np.cov(features_00, rowvar=False)
    var_0_1 = np.cov(features_01, rowvar=False)
    var_1_0 = np.cov(features_10, rowvar=False)
    var_1_1 = np.cov(features_11, rowvar=False)

    # pf_model = LogisticRegression() # Only x dependent oracle, can experiment with other oracles later
    # # Fit pf_model on random data
    # #rng = np.random.default_rng(seed=4)
    # #random_x = np.random.rand(num_samples, train_embeddings.shape[1])  # Random data for fitting
    # random_x = rng.random((num_samples, train_embeddings.shape[1]))
    # #random_y = np.random.randint(0, 2, num_samples)  # Random labels for fitting
    # random_y = rng.integers(0, 2, num_samples)
    # pf_model.fit(random_x, random_y)
    # # Sanity check: this should give random accuracy
    # print('Sanity Check accuracy: ', accuracy_score(y_train, pf_model.predict(train_embeddings)))

    # # Generate predictions on train data
    # filteration = pf_model.predict(train_embeddings)
    # # Filter train data based on filteration
    # x_filtered = train_embeddings[filteration == 1]
    # z_filtered = a_train[filteration == 1]
    # y_filtered = y_train[filteration == 1]

    # print('Truncation Stats: ', x_filtered.shape, y_filtered.shape, train_embeddings.shape, y_train.shape)

    # return x_filtered, z_filtered, y_filtered,test_embeddings, a_test, y_test, pf_model, [(mu_0_0, var_0_0), (mu_0_1, var_0_1), (mu_1_0, var_1_0), (mu_1_1, var_1_1)]\
    return [(mu_0_0, var_0_0), (mu_0_1, var_0_1), (mu_1_0, var_1_0), (mu_1_1, var_1_1)], train_embeddings, a_train, y_train, test_embeddings, a_test, y_test, emb_model

def generate_data_and_model(dim=10, var=3, train_samples=5000, test_samples=2000, priors=None, means=None):
    """
    Generate a synthetic dataset and a black-box model.
    """
    # Unroll priors
    p_a_1, p_y_1_a_1, p_y_1_a_0 = priors if priors else (0.7, 0.7, 0.4)
    means, x, z, y = generate_examples(dim=dim, var=var, num_examples=train_samples, 
                                       a_1_prob=p_a_1, 
                                       y_1_a_1_prob=p_y_1_a_1, 
                                       y_1_a_0_prob=p_y_1_a_0)
    _, x_test, z_test, y_test = generate_examples(dim=dim, var=var, num_examples=test_samples)

    # Concat x and z
    #x_z_train = np.hstack((x, z.reshape(-1, 1)))
    #x_z_test = np.hstack((x_test, z_test.reshape(-1, 1)))

    pf_model = LogisticRegression() # Only x dependent oracle, can experiment with other oracles later
    # Fit pf_model on random data
    random_x = np.random.rand(train_samples, dim)  # Random data for fitting
    random_y = np.random.randint(0, 2, train_samples)  # Random labels for fitting
    pf_model.fit(random_x, random_y)
    # Sanity check: this should give random accuracy
    print('Sanity Check accuracy: ', accuracy_score(y, pf_model.predict(x)))

    # Generate predictions on train data
    filteration = pf_model.predict(x)
    # Filter train data based on filteration
    x_filtered = x[filteration == 1]
    z_filtered = z[filteration == 1]
    y_filtered = y[filteration == 1]

    print('Truncation Stats: ', x_filtered.shape, y_filtered.shape, x.shape, y.shape)

    return x_filtered, z_filtered, y_filtered, x_test, z_test, y_test, pf_model, means

# Manolis specific steps
def mle_moment_estimation(sample):
    """
    Maximum Likelihood Estimation (MLE) for mean and covariance from the sample.
    """
    mean_est = np.mean(sample, axis=0)
    cov_est = np.cov(sample, rowvar=False)
    return {'mean': mean_est, 'cov': cov_est}

import cvxpy as cp
from scipy.linalg import fractional_matrix_power

def suff_statistic(x, dist='gaussian'):
    """
    Calculate sufficient statistics for the given distribution.
    """
    if dist == 'gaussian':
        xxt = (-1/2)*np.outer(x, x)
        # flatten the matrix in the shape of x and return
        return np.hstack((xxt.reshape(-1,), x))
    else:
        raise ValueError("Unsupported distribution type")

def sample_gradient(w, x, oracle, dim, dist='gaussian', group=None, max_iter=5):
    # Extract matrix and vector from w
    T = w[0:dim*dim].reshape(dim, dim)
    v = w[dim*dim:].reshape(dim, 1)
    T_inv = np.linalg.inv(T)  # Inverse of the matrix T
    mu_est = T_inv @ v  # Mean estimate
    cov_est = T_inv
    counter = 0
    while True:
        # Sample from a gaussian distribution using parameters
        if dist == 'gaussian':
            y = np.random.multivariate_normal(mu_est.reshape(-1,), cov_est).reshape(1,-1)
            # Add group info as last column
            y_aware = np.hstack((y, np.array([[group]]))) if group is not None else 0
        if oracle.predict(y_aware):
            y = y.reshape(-1,)
            return suff_statistic(y, dist=dist) - suff_statistic(x[:-1], dist=dist)
        counter += 1
        if counter >= max_iter:
            #print('Max iter reached')
            return None

def project2(r, dim, c, w0):
    # Program to find v
    v_prime = cp.Variable((dim,))
    T_prime = cp.Variable((dim, dim), PSD=True)

    # Add small regularization for numerical stability
    epsilon = 1e-6
    
    # Check if scaling is actually needed
    r_norm = np.linalg.norm(r)
    w0_norm = np.linalg.norm(w0)
    
    # Skip scaling if norms are already reasonable
    if 0.1 <= r_norm <= 10 and 0.1 <= w0_norm <= 10:
        r_scaled = r
        w0_scaled = w0
        scaling_factor = 1.0
    else:
        # Only scale when necessary
        scaling_factor = max(r_norm, w0_norm, 1.0)
        r_scaled = r / scaling_factor if r_norm > 1e-10 else r
        w0_scaled = w0 / scaling_factor if w0_norm > 1e-10 else w0
    
    c_scaled = c / scaling_factor
    
    # Set up L2 minimization problem
    objective = cp.Minimize(
        cp.sum_squares(v_prime - r_scaled[dim*dim:]) + 
        cp.sum_squares(T_prime - r_scaled[0:dim*dim].reshape(dim, dim))
    )
    
    # Constraints
    constraints = [
        cp.sum_squares(v_prime - w0_scaled[dim*dim:]) + 
        cp.sum_squares(T_prime - w0_scaled[0:dim*dim].reshape(dim, dim)) <= 2*(c_scaled**2),
        T_prime >> epsilon * np.eye(dim),
        cp.norm(v_prime, 2) <= 5,
        cp.norm(T_prime, 'fro') <= 5
    ]
    
    prob = cp.Problem(objective, constraints)
    
    # Use SCS with relaxed tolerances for speed
    # make it 250 for now for debugging
    try:
        prob.solve(solver=cp.SCS, eps=1e-3, max_iters=2500, verbose=False, acceleration_lookback=0)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback to default solver only if SCS fails
            prob.solve(verbose=False)
    except:
        prob.solve(verbose=False)
    
    if prob.status in ["optimal", "optimal_inaccurate"]:
        # Scale back only if we scaled
        if scaling_factor != 1.0:
            v_opt = v_prime.value * scaling_factor
            T_opt = T_prime.value * scaling_factor
        else:
            v_opt = v_prime.value
            T_opt = T_prime.value
        
        # Ensure PSD by eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(T_opt)
        eigvals = np.maximum(eigvals, epsilon)
        T_opt = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        return np.hstack((T_opt.reshape(-1,), v_opt.reshape(-1,)))
    else:
        return None

import copy
import matplotlib.pyplot as plt
def subgroup_manolis(data, pf_model, alpha, var=2, dim=5, means=None, dataset='gaussian'):
    print(f'Estimating params using Manolis algorithm')
    # Extract f_0=1 samples from data
    df = data[data['f_0'] == 1]
    # Pop out sensitive and label
    z_filtered = df['sensitive'].values
    y_filtered = df['label'].values
    x_filtered = df.drop(columns=['label', 'f_0']).values # The model is group aware, hence we need to retain sensitive attributes
    # Manolis algorithm
    unique_a = np.unique(z_filtered)
    #print('Unique values in sensitive attribute:', unique_a)
    unique_y = np.unique(y_filtered)
    #print('Unique values in label:', unique_y)
    subgroups_params = {}
    lam = 1000
    for a in unique_a:
        for y in unique_y:
            rad = 500*((np.log(1/alpha[f'{y}{a}'])) / (alpha[f'{y}{a}']**2))
            # Filter data based on sensitive attribute and label
            # stack the columns of a identity matrix
            filtered_data = x_filtered[(z_filtered == a) & (y_filtered == y)]
            # Split data into two parts
            num_samples = min(int(0.5*filtered_data.shape[0]), 10000) # 2 samples for only testing code
            filtered_data_split = filtered_data[:num_samples]
            filtered_data = filtered_data[num_samples:num_samples*2]
            print(f'Number of samples for a={a}, y={y}: {filtered_data.shape[0]}')
            #if filtered_data.shape[0] > 0:
            # Estimate parameters for the filtered data
            params = mle_moment_estimation(filtered_data_split[:,:-1]) # Parameters only need x, we will send group info to sample_gradient so that oracle can use group information

            w = np.hstack((np.linalg.pinv(params['cov']).reshape(-1,), (np.linalg.pinv(params['cov']) @ params['mean']).reshape(-1)))
            all_params = {}
            all_params[0] = copy.deepcopy(w)
            no_result_gradient = 0
            no_result_project = 0
            for k in tqdm(range(1,filtered_data.shape[0]+1)):
                eta_i = 1/(lam)
                v = sample_gradient(w, filtered_data[k-1], pf_model, dim=dim, group=a)
                # Check if v is a vector or None
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    r = w - (eta_i * v)
                    w_new = project2(r, dim, rad, all_params[0])
                    if isinstance(w_new, np.ndarray) and w_new.ndim == 1:
                        w = w_new
                    else:
                        no_result_project += 1
                else:
                    no_result_gradient += 1
                all_params[k] = w
            print(f'% of samples with no result in gradient for subgroup a={a}, y={y} (counter): {no_result_gradient/filtered_data.shape[0]}')
            print(f'% of samples with no result in project for subgroup a={a}, y={y} (infeasible): {no_result_project/filtered_data.shape[0]}')
            subgroups_params[f'{y}{a}'] = all_params
            if y == 0 and a == 0:
                true_mean = means[0]
            elif y == 0 and a == 1:
                true_mean = means[1]
            elif y == 1 and a == 0:
                true_mean = means[2]
            elif y == 1 and a == 1:
                true_mean = means[3]
            
            if dataset == 'gaussian':
                true_var = var*np.eye(dim)
            else:
                if y == 0 and a == 0:
                    true_var = var[0]
                elif y == 0 and a == 1:
                    true_var = var[1]
                elif y == 1 and a == 0:
                    true_var = var[2]
                elif y == 1 and a == 1:
                    true_var = var[3]
            true_mean = np.array(true_mean).reshape(-1,)
            true_param_var = np.linalg.pinv(true_var)
            true_param_mean = true_param_var @ true_mean
            true_vectors = np.hstack((true_param_var.reshape(-1,), true_param_mean.reshape(-1,)))
            errors = {}
            running_average_vector = None
            for k in subgroups_params[f'{y}{a}']:
                if k == 0:
                    running_average_vector = subgroups_params[f'{y}{a}'][k]
                else:
                    running_average_vector += subgroups_params[f'{y}{a}'][k]
                final_est_vector = (1/(k+1))*running_average_vector
                error = np.linalg.norm(final_est_vector - true_vectors)
                errors[k] = error
            subgroups_params[f'{y}{a}'] = final_est_vector
            plt.figure()
            plt.title(f'Subgroup: a={a}, y={y}')
            plt.plot(errors.keys(), errors.values(), label='Error')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.legend()
            if dataset == 'gaussian':
                plt.savefig(f'../results_gaussian/manolis_errors_a_{a}_y_{y}.png')
            else:
                plt.savefig(f'../results_gaussian/{dataset}_manolis_errors_a_{a}_y_{y}.png')
            plt.close()
            # print(final_est_vector[0 : dim*dim].reshape(dim, dim))
            # print(final_est_vector[dim*dim:].reshape(dim, 1))
            # print(true_vectors[0 : dim*dim].reshape(dim, dim))
            # print(true_vectors[dim*dim:].reshape(dim, 1))
    return subgroups_params

def map_oracle(sampled_row, q, theta, dim):
    inv_var_0 = theta[0][0:dim*dim].reshape(dim, dim)
    mean_0 = np.linalg.inv(inv_var_0).dot(theta[0][dim*dim:].reshape(dim,))
    inv_var_1 = theta[1][0:dim*dim].reshape(dim, dim)
    mean_1 = np.linalg.inv(inv_var_1).dot(theta[1][dim*dim:].reshape(dim,))
    # Calculate gaussian density of incoming point
    eta_0 = (np.sqrt(np.linalg.det(inv_var_0))*np.exp(-0.5 * np.dot((sampled_row - mean_0).T, np.dot(inv_var_0, (sampled_row - mean_0)))))/((2*np.pi)**(dim/2))
    eta_1 = (np.sqrt(np.linalg.det(inv_var_1))*np.exp(-0.5 * np.dot((sampled_row - mean_1).T, np.dot(inv_var_1, (sampled_row - mean_1)))))/((2*np.pi)**(dim/2))
    map_0 = q[0] * eta_0
    map_1 = q[1] * eta_1
    if map_0 > map_1:
        return 0
    else:
        return 1

def generate_online_sample(means, var, dim, priors, group, model):
    """
    priors = {
        'p_a_1': p_a_1,
        'p_y_1_a_1': p_y_1_a_1,
        'p_y_1_a_0': p_y_1_a_0,
        'p_y_1': p_y_1
    }
    """
    if group == 0:
        p_y_1_a_a = priors['p_y_1_a_0']
    else:
        p_y_1_a_a = priors['p_y_1_a_1']
    true_label = np.random.binomial(1, p_y_1_a_a)
    if group == 0 and true_label == 0:
        sampled_row = np.random.multivariate_normal(means[0], var*np.eye(dim))
    elif group == 1 and true_label == 0:
        sampled_row = np.random.multivariate_normal(means[1], var*np.eye(dim))
    elif group == 0 and true_label == 1:
        sampled_row = np.random.multivariate_normal(means[2], var*np.eye(dim))
    elif group == 1 and true_label == 1:
        sampled_row = np.random.multivariate_normal(means[3], var*np.eye(dim))
    # Add group info as last column
    sampled_row_new = np.hstack((sampled_row, np.array([group])))
    decision = model.predict(sampled_row_new.reshape(1, -1)).reshape(-1,)
    return sampled_row, true_label, decision

def generate_real_online_sample(df, group, model):
    # df_group = df[df['sensitive'] == group]
    # sampled_row = df_group.sample(n=1).drop(columns=['label', 'sensitive']).values.reshape(-1,)
    # true_label = df_group.sample(n=1)['label'].values[0]
    # # Add group info as last column
    # sampled_row_new = np.hstack((sampled_row, np.array([group])))
    # decision = model.predict(sampled_row_new.reshape(1, -1)).reshape(-1,)
    # return sampled_row, true_label, decision
    df_group = df[df['sensitive'] == group]
    sampled_row = df_group.sample(n=1)
    true_label = sampled_row['label'].values[0]
    sampled_row.drop(columns=['label', 'f_0'], inplace=True)
    decision = model.predict(sampled_row.values).reshape(-1,)
    sampled_row_new  = sampled_row.drop(columns=['sensitive'])
    return sampled_row_new.values.reshape(-1,), true_label, decision

def main(args):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dataset = args.dataset
    policy = args.policy
    algorithm = args.algorithm
    train = args.train
    model = args.model
    score_algo = args.score
    dim = args.dim
    var = args.var
    train_samples = args.train_samples
    test_samples = args.test_samples
    # Load Dataset
    
    if args.dataset == 'gaussian':
        # Priors
        p_a_1 = 0.7
        p_y_1_a_1 = 0.7
        p_y_1_a_0 = 0.4
        p_y_1 = (p_a_1 * p_y_1_a_1) + ((1 - p_a_1) * p_y_1_a_0) # 0.28 + 0.12 = 0.4
        priors = {
            'p_a_1': p_a_1,
            'p_y_1_a_1': p_y_1_a_1,
            'p_y_1_a_0': p_y_1_a_0,
            'p_y_1': p_y_1
        }

        #generate_real_data_and_model()

        means, x, z, y = generate_examples(dim=dim, var=var, num_examples=train_samples, 
                                        a_1_prob=p_a_1, 
                                        y_1_a_1_prob=p_y_1_a_1, 
                                        y_1_a_0_prob=p_y_1_a_0)
        _, x_test, z_test, y_test = generate_examples(dim=dim, var=var, num_examples=train_samples,
                                                    means=means, a_1_prob=p_a_1,
                                                        y_1_a_1_prob=p_y_1_a_1,
                                                        y_1_a_0_prob=p_y_1_a_0)
        _, x_val, z_val, y_val = generate_examples(dim=dim, var=var, num_examples=10000,
                                                means=means, a_1_prob=p_a_1,
                                                y_1_a_1_prob=p_y_1_a_1,
                                                y_1_a_0_prob=p_y_1_a_0)
        print(f'Data generated and priors set: {x.shape}, {x_test.shape}')
        # Convert to dataframe
        df = pd.DataFrame(x, columns=[f'x_{i}' for i in range(dim)])
        df['sensitive'] = z
        df['label'] = y
        test_df = pd.DataFrame(x_test, columns=[f'x_{i}' for i in range(dim)])
        test_df['sensitive'] = z_test
        test_df['label'] = y_test
        # val_df = pd.DataFrame(x_val, columns=[f'x_{i}' for i in range(dim)])
        # val_df['sensitive'] = z_val
        # val_df['label'] = y_val
        A = 'sensitive'
        target_col = 'label'
    else:
        # return [(mu_0_0, var_0_0), (mu_0_1, var_0_1), (mu_1_0, var_1_0), (mu_1_1, var_1_1)], train_embeddings, a_train, y_train, test_embeddings, a_test, y_test, emb_model
        moments, x_all, z_all, y_all, x_test_all, z_test_all, y_test_all, emb_model = generate_real_data_and_model(dataset=dataset)
        print(f'Real data loaded: {x_all.shape}, {x_test_all.shape}')
        means = [moments[0][0], moments[1][0], moments[2][0], moments[3][0]]
        var = [moments[0][1], moments[1][1], moments[2][1], moments[3][1]]
        # Convert to dataframe
        df_all = pd.DataFrame(x_all, columns=[f'x_{i}' for i in range(x_all.shape[1])])
        df_all['sensitive'] = z_all
        df_all['label'] = y_all
        test_df_all = pd.DataFrame(x_test_all, columns=[f'x_{i}' for i in range(x_test_all.shape[1])])
        test_df_all['sensitive'] = z_test_all
        test_df_all['label'] = y_test_all
        A = 'sensitive'
        target_col = 'label'
    
    try:
        if args.dataset == 'gaussian':
            os.makedirs(f"../results_gaussian/algorithm_exp/{dataset}/{model}/{policy}")
            os.makedirs(f"../results_gaussian/algorithm_two/{dataset}/{model}/{policy}") 
            os.makedirs(f"../models_gaussian")
        else:
            os.makedirs(f"../results_gaussian/algorithm_exp/{dataset}_manolis/{model}/{policy}")
            os.makedirs(f"../results_gaussian/algorithm_two/{dataset}_manolis/{model}/{policy}") 
            os.makedirs(f"../models_{dataset}_gaussian")
    except:
        pass

    if args.dataset == 'gaussian':
        file_exists_exp = os.path.exists(f"../results_gaussian/algorithm_exp/{dataset}/{model}/{policy}/{timestr}.csv")
        file_exists_two = os.path.exists(f"../results_gaussian/algorithm_two/{dataset}/{model}/{policy}/{timestr}.csv")
        writer_path_exp = f"../results_gaussian/algorithm_exp/{dataset}/{model}/{policy}/{timestr}.csv"
        writer_path_two = f"../results_gaussian/algorithm_two/{dataset}/{model}/{policy}/{timestr}.csv"
    else:
        file_exists_exp = os.path.exists(f"../results_gaussian/algorithm_exp/{dataset}_manolis/{model}/{policy}/{timestr}.csv")
        file_exists_two = os.path.exists(f"../results_gaussian/algorithm_two/{dataset}_manolis/{model}/{policy}/{timestr}.csv")
        writer_path_exp = f"../results_gaussian/algorithm_exp/{dataset}_manolis/{model}/{policy}/{timestr}.csv"
        writer_path_two = f"../results_gaussian/algorithm_two/{dataset}_manolis/{model}/{policy}/{timestr}.csv"

    with open(writer_path_exp, mode='w', newline='', encoding='utf-8') as file_exp, open(writer_path_two, mode='w', newline='', encoding='utf-8') as file_two:
        writer_exp = csv.writer(file_exp)
        writer_two = csv.writer(file_two)
        if not file_exists_exp or os.path.getsize(writer_path_exp) == 0:
            writer_exp.writerow(['run', 'algo_name', 'sample_cost_fixed', 'label_cost_fixed', 'eps', 'tau_prime', 'f_0', 'policy', 'sample_cost', 'label_cost', 'eo_diff_est', 'eo_diff_true', 'num_samples', 'true_fair', 'pred_fair'])
        if not file_exists_two or os.path.getsize(writer_path_two) == 0:
            # Check this later
            writer_two.writerow(['run', 'algo_name', 'sample_cost_fixed', 'label_cost_fixed', 'eps', 'tau_prime', 'f_0', 'policy', 'sample_cost', 'label_cost', 'eo_diff_est', 'eo_diff_true', 'num_samples', 'true_fair', 'pred_fair'])
        # Get filtered data and model (Same for both algorithms)
        if dataset == 'gaussian':
            _ , model, alpha, dist_fair = get_f_0(policy, df, A, train, model, target_col=target_col, test_df=test_df)
        else:
            _ , model, alpha, dist_fair = get_f_0(policy, df_all, A, train, model, target_col=target_col, test_df=test_df_all)
        print(f'Positivity rate of f_0 for every subgroup: {alpha}')
        for run in range(5):
            # Generate new samples
            if dataset == 'gaussian':
                _, x, z, y = generate_examples(dim=dim, var=var, num_examples=train_samples, means=means, a_1_prob=p_a_1,
                                                y_1_a_1_prob=p_y_1_a_1,
                                                y_1_a_0_prob=p_y_1_a_0)
                _, x_test, z_test, y_test = generate_examples(dim=dim, var=var, num_examples=train_samples,
                                                    means=means, a_1_prob=p_a_1,
                                                        y_1_a_1_prob=p_y_1_a_1,
                                                        y_1_a_0_prob=p_y_1_a_0)
            else:
                train_indices = np.random.choice(x_all.shape[0], train_samples, replace=True)
                test_indices = np.random.choice(x_test_all.shape[0], test_samples, replace=True)
                x = x_all[train_indices]
                z = z_all[train_indices]
                y = y_all[train_indices]
                x_test = x_test_all[test_indices]
                z_test = z_test_all[test_indices]
                y_test = y_test_all[test_indices]

            # Convert to dataframe
            df = pd.DataFrame(x, columns=[f'x_{i}' for i in range(x.shape[1])])
            df['sensitive'] = z
            df['label'] = y
            test_df = pd.DataFrame(x_test, columns=[f'x_{i}' for i in range(x_test.shape[1])])
            test_df['sensitive'] = z_test
            test_df['label'] = y_test
            df , _, _, dist_fair = get_f_0(policy, df, A, False, model, target_col=target_col, test_df=test_df, given_f_0=model)
            print(f'Run {run+1}, Dist fairness: {dist_fair}')
            # Estimate using filtered data (Treat f_0 as truncation)
            subgroup_params = subgroup_manolis(data=df, pf_model=model, alpha=alpha, var=var, dim=x.shape[1], means=means, dataset=dataset)
            costs = []
            eps_prime = 1/5
            delta = 0.01
            for eps in [0.8, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]:
                for costs in [(0.5,0.25), (0.5, 0.5), (0.5, 1), (0.5, 3)]:
                    tau_prime = (0.5*np.log(24/delta))/(eps_prime**2)
                    tau = min((2*np.log(24/delta))/(eps**2), 1000) # Where will this be used, the paper uses tau to define p, whereas we use sample ratios
                    print(f'Tau prime: {tau_prime}, Tau: {tau}, Epsilon: {eps}, Delta: {delta}, Costs: {costs}')
                    sample_cost = costs[0]
                    label_cost = costs[1]
                        
                    df_a_0 = df[df[A]==0]
                    df_a_1 = df[df[A]==1]
                    p_00 = df_a_0[(df_a_0['f_0'] == 1) & (df_a_0[target_col] == 0)].shape[0] / len(df_a_0)
                    p_01 = df_a_1[(df_a_1['f_0'] == 1) & (df_a_1[target_col] == 0)].shape[0] / len(df_a_1)    
                    p_10 = df_a_0[(df_a_0['f_0'] == 1) & (df_a_0[target_col] == 1)].shape[0] / len(df_a_0)
                    p_11 = df_a_1[(df_a_1['f_0'] == 1) & (df_a_1[target_col] == 1)].shape[0] / len(df_a_1)  

                    s = 0
                    s_2 = 0
                    curr_sample_cost = 0
                    curr_label_cost = 0
                    curr_sample_cost_2 = 0
                    curr_label_cost_2 = 0
                    q_ya = []
                    q_ya_2 = []
                    for y in [0,1]:
                        concatenated_samples_a = [] # This should be outside the loop
                        concatenated_samples_a_2 = []
                        for a in [0,1]:
                            df_a = df.loc[df[A]==a]
                            df_a = df_a.reset_index(drop=True)   
                            m_prime = tau_prime
                            concatenated_samples_prime = 0
                            while(m_prime > 0):
                                #index = rng.integers(0,len(df_a), 1)
                                #sampled_row = df_a.iloc[index]
                                # Get sample for group A=a
                                if dataset == 'gaussian':
                                    sampled_row, true_label, decision = generate_online_sample(means, var, dim, priors, a, model)
                                else:
                                    sampled_row, true_label, decision = generate_real_online_sample(df_all, a, model)
                                concatenated_samples_prime += 1
                                #if sampled_row['f_0'].values[0] == 0:
                                if decision == 0:
                                    curr_sample_cost += sample_cost
                                #if y == sampled_row[target_col].values[0]: 
                                if y == true_label:
                                    m_prime -= 1
                                #if sampled_row[target_col].values[0] == 0 and sampled_row['f_0'].values[0]==0:
                                if true_label == 0 and decision == 0:
                                    curr_label_cost += label_cost
                            m = tau
                            concatenated_samples = 0 
                            while (m > 0):
                                # index = rng.integers(0,len(df_a), 1)
                                # sampled_row = df_a.iloc[index]
                                # concatenated_samples += 1
                                # Get sample for group A=a
                                if dataset == 'gaussian':
                                    sampled_row, true_label, decision = generate_online_sample(means, var, dim, priors, a, model)
                                else:
                                    sampled_row, true_label, decision = generate_real_online_sample(df_all, a, model)
                                concatenated_samples += 1
                                #if sampled_row['f_0'].values[0] == 0:
                                if decision == 0:
                                    curr_sample_cost_2 += sample_cost
                                #if y == sampled_row[target_col].values[0]: 
                                if y == true_label:
                                    m -= 1
                                #if sampled_row[target_col].values[0] == 0 and sampled_row['f_0'].values[0]==0:
                                if true_label == 0 and decision == 0:
                                    curr_label_cost_2 += label_cost
                            q_ya.append(tau_prime/concatenated_samples_prime)
                            q_ya_2.append(tau/concatenated_samples)
                            concatenated_samples_a.append(concatenated_samples)
                            concatenated_samples_a_2.append(concatenated_samples)
                    q_00, q_01 = q_ya[0], q_ya[1]
                    q_10, q_11 = q_ya[2], q_ya[3]

                    q_00_2, q_01_2 = q_ya_2[0], q_ya_2[1]
                    q_10_2, q_11_2 = q_ya_2[2], q_ya_2[3]

                    s += sum(concatenated_samples_a)
                    s_2 += sum(concatenated_samples_a_2)
                    # Do more stuff here
                    N_a = {}
                    for a in [0,1]:
                        df_a = df.loc[df[A]==a]
                        df_a = df_a.reset_index(drop=True)  
                        if a == 0:
                            q_min = min(q_00, q_10)
                            q_oracle = (q_00, q_10)
                            theta_oracle = subgroup_params['00'], subgroup_params['10']
                        else:
                            q_min = min(q_01, q_11)
                            q_oracle = (q_01, q_11)
                            theta_oracle = subgroup_params['01'], subgroup_params['11']
                        m = min(int((500*np.log(8/delta))/(q_min*eps**2)), 20000)
                        concat_samples_2 = 0
                        curr_sample_cost_2 = 0
                        N_0 = 0
                        N_1 = 0
                        N = copy.deepcopy(m)
                        while(m > 0):
                            # index = rng.integers(0,len(df_a), 1)
                            # sampled_row = df_a.iloc[index]
                            # Get sample for group A=a
                            if dataset == 'gaussian':
                                sampled_row, true_label, decision = generate_online_sample(means, var, dim, priors, a, model)
                            else:
                                sampled_row, true_label, decision = generate_real_online_sample(df_all, a, model)
                            concat_samples_2 += 1
                            #if sampled_row['f_0'].values[0] == 0:
                            if decision == 0:
                                curr_sample_cost_2 += sample_cost
                            #new_sampled_row = sampled_row.drop(columns=['f_0', 'sensitive', 'label']).values.reshape(-1,)
                            new_sampled_row = sampled_row
                            if map_oracle(new_sampled_row, q_oracle, theta_oracle, dim=sampled_row.shape[0]) == 0:
                                N_0 += 1
                            else:
                                N_1 += 1
                            m -= 1
                        N_a[a] = (N_0, N_1, N)
                    q_hat_00 = N_a[0][0]/N_a[0][2]
                    q_hat_10 = N_a[0][1]/N_a[0][2]
                    q_hat_01 = N_a[1][0]/N_a[1][2]
                    q_hat_11 = N_a[1][1]/N_a[1][2]
                    # End more stuff here
                    eo_diff_exp = max(np.abs((p_00/q_hat_00) - (p_01/q_hat_01)), np.abs((p_10/q_hat_10) - (p_11/q_hat_11)))
                    eo_diff_2 = max(np.abs((p_00/q_00_2) - (p_01/q_01_2)), np.abs((p_10/q_10_2) - (p_11/q_11_2)))
                    # What all to track: writer.writerow(['algo_name', 'sample_cost_fixed', 'label_cost_fixed', 'eps', 'tau_prime', 'policy', 'sample_cost', 'label_cost', 'eo_diff_est', 'eo_diff_true', 'num_samples', 'true_fair', 'pred_fair'])
                    writer_exp.writerow([run+1,'exp', sample_cost, label_cost, eps, tau_prime, args.model, policy, curr_sample_cost, curr_label_cost, eo_diff_exp, dist_fair, s, 1 - (dist_fair > eps) , 1 - (eo_diff_exp > eps/2)])
                    writer_two.writerow([run+1, 'two', sample_cost, label_cost, eps, tau, args.model, policy, curr_sample_cost_2, curr_label_cost_2, eo_diff_2, dist_fair, s_2, 1 - (dist_fair > eps), 1 - (eo_diff_2 > eps/2)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Testing Fairness.')
    parser.add_argument('--dataset', type=str, help="Name of dataset: adult, law, newadult, gaussian", default='adult')
    parser.add_argument('--policy', type=str, help="Name of audit policy: full_data, wo_proattr, random, protected, fair", default='full_data')
    parser.add_argument('--model', type=str, help="Whether to train LR or GNB", default='LR')
    parser.add_argument('--algorithm', type=str, help="Name of algorithm: one, three", default='three')
    parser.add_argument('--alpha', type=str, help="Alternate Hypothesis Constant, alpha", default='0.10')
    parser.add_argument('--train', type=str, help="Whether to train or load saved models", default='True')
    parser.add_argument('--score', type=str, help="For score-based formulation", default='True')
    parser.add_argument('--var', type=int, help="Variance", default=3)
    parser.add_argument('--dim', type=int, help="Dimension", default=5)
    parser.add_argument('--train_samples', type=int, help="Number of training samples", default=20000)
    parser.add_argument('--test_samples', type=int, help="Number of test samples", default=2000)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)