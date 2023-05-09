import argparse
import os
import numpy as np
from collections import defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set_style('white', rc={'xtick.bottom': False,
                           'ytick.left': False})
sns.set_context("paper", rc={'text.usetex': True,
                            'lines.linewidth' : 2.2,
                            'font.size': 15,
                            'figure.autolayout': True,
                            'xtick.labelsize': 12,
                            'ytick.labelsize': 12,
                            'axes.titlesize' : 12,
                            'axes.labelsize' : 15,
                            'lines.markersize' : 12,
                            'legend.fontsize': 14})


def make_overlapping_scatterplot(df):
    new_df = {'ev_0': [], 'ev_1': [], 'Set': []}
    ch = []
    pf = []
    for _, row in df.iterrows():
        if row['dds']:
            new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('DUS')
        if row['cdds'] and row['dds']:
            new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('CDUS')
        if row['pf'] and row['dds']:
            """new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('PF')"""
            pf.append((row['ev_0'], row['ev_1']))
        if row['ch'] and row['dds']:
            """new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('CH')"""
            ch.append((row['ev_0'], row['ev_1']))

    sns.scatterplot(data=new_df, x="ev_0", y="ev_1", hue='Set', style="Set", palette=['#a31034', '#5CB5FF'], markers=['o', 'X'], size='Set', sizes=(40, 120), alpha=0.8)
    pf = sorted(pf)
    sns.lineplot(x=[x[0] for x in pf], y=[x[1] for x in pf], label='PF', markers='-', color='black', linewidth=1.5)
    ch = sorted(ch)
    sns.lineplot(x=[x[0] for x in ch], y=[x[1] for x in ch], label='CH', color='grey', linewidth=1)
    plt.grid(alpha=0.25)
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.savefig('visualisation.pdf')


def extend_df(df):
    def get_smallest(row):
        if row['ch']:
            row['Smallest Subset'] = 'CH'
        elif row['pf']:
            row['Smallest Subset'] = 'PF'
        elif row['cdds']:
            row['Smallest Subset'] = 'CDUS'
        else:
            row['Smallest Subset'] = 'DUS'
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
    parser.add_argument("--seed", type=int, nargs='+', default=[4],
                        help="The seed for random number generation.")
    parser.add_argument("--env", type=str, nargs='+', default=["small"],
                        help="The environments to run experiments on.")
    parser.add_argument("--alg", type=str, nargs='+', default=['DIMOQ'], help="The algorithm to use.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    for env_name in args.env:
        for alg in args.alg:
            for seed in args.seed:
                dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
                df = pd.read_csv(os.path.join(dists_dir, 'pruning_results.csv'))
                make_overlapping_scatterplot(df)
