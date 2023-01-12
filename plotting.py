import argparse
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()


def make_overlapping_scatterplot(df):
    new_df = {'ev_0': [], 'ev_1': [], 'Set': []}
    for _, row in df.iterrows():
        if row['dds']:
            new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('DDS')
        if row['cdds']:
            new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('CDUS')
        if row['pf']:
            new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('PF')
        if row['ch']:
            new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('CH')

    sns.scatterplot(data=new_df, x="ev_0", y="ev_1", hue='Set', style="Set", palette='deep', s=150, markers=['o', 'X', '^', '*'])
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.show()


def extend_df(df):
    def get_smallest(row):
        if row['ch']:
            row['Smallest Subset'] = 'CH'
        elif row['pf']:
            row['Smallest Subset'] = 'PF'
        elif row['cdds']:
            row['Smallest Subset'] = 'CDUS'
        else:
            row['Smallest Subset'] = 'DDS'
        return row

    df = df.apply(get_smallest, axis=1)
    return df


def make_scatterplot(df):
    df = extend_df(df)
    sns.scatterplot(data=df, x="ev_0", y="ev_1", hue="Smallest Subset")
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs', help='The directory to save the logs.')
    parser.add_argument("--seed", type=int, nargs='+', default=[1],
                        help="The seed for random number generation.")
    parser.add_argument("--env", type=str, nargs='+', default=["small"],
                        help="The environments to run experiments on.")
    parser.add_argument("--alg", type=str, nargs='+', default=['DIMOQ'], help="The algorithm to use.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    for env_name in args.env:
        for seed in args.seed:
            for alg in args.alg:
                dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
                df = pd.read_csv(os.path.join(dists_dir, 'results.csv'))
                make_overlapping_scatterplot(df)
