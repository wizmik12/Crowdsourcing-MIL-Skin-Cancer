"""
Utility to merge all the resulting csv from every experiment into one single file.
"""
import os
import pandas as pd
import numpy as np
import argparse


def join(path):
    files = os.listdir(path)
    results = pd.DataFrame([], columns=["lengthscale", "variance", "num_inducing_points", "learning_rate", "F1 Score", "F1 std", "Accuracy", "Accuracy std", "ROC_AUC", "ROC_AUC std", "Precision", "Precision std", "Recall", "Recall std"])
    for name in files:
        lengthscale, variance, learning_rate, num_inducing_points = name[:-4].split('-')
        result_df = pd.read_csv(path + name)
        cols = list(result_df.columns)
        cols[0] = 'idx'
        result_df.columns = cols
        result_df = result_df.set_index('idx')
        mean = pd.DataFrame(result_df.loc['mean']).transpose().to_numpy()
        std = pd.DataFrame(result_df.loc['std']).transpose().to_numpy()
        values = [float(lengthscale), float(variance), float(num_inducing_points), float(learning_rate), *np.vstack([mean,std]).transpose().reshape((1,-1)).tolist()[0]]
        values = pd.DataFrame(values).transpose()
        values.columns = results.columns
        results = pd.concat([results, values], axis=0, ignore_index=True)
    results = results.set_index(["lengthscale", "variance", "num_inducing_points", "learning_rate"]).sort_index()
    results.to_csv(path + 'aggregated_results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to results (csv) folder.')   
    args = parser.parse_args()
    join(args.path)