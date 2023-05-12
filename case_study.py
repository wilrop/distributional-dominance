import os
import argparse

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from distrib.multivariate_categorical_distribution import MCD
from utils.data import load_dists

sns.set_style('white', rc={'xtick.bottom': False,
                           'ytick.left': False})
sns.set_context("paper", rc={'text.usetex': True,
                             'lines.linewidth': 2.2,
                             'font.size': 15,
                             'figure.autolayout': True,
                             'xtick.labelsize': 12,
                             'ytick.labelsize': 12,
                             'axes.titlesize': 12,
                             'axes.labelsize': 15,
                             'lines.markersize': 12,
                             'legend.fontsize': 14})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs', help='The directory of the saved logs.')
    parser.add_argument("--seed", type=int, default=1, help="The seed of the experiment.")
    parser.add_argument("--env", type=str, default="small", help="The environment of the experiment.")
    parser.add_argument("--alg", type=str, default='DIMOQ', help="The algorithm that was used.")
    args = parser.parse_args()
    return args


def u_func1(coords):
    return np.prod(coords, axis=-1)


def u_func2(coords):
    return np.min(coords, axis=-1)


def make_overlapping_scatterplot(df,
                                 coords_pf=None,
                                 offset_pf=None,
                                 u_pf=None,
                                 coords_all=None,
                                 offset_all=None,
                                 u_all=None,
                                 annotate=False,
                                 num=0):
    new_df = {'ev_0': [], 'ev_1': [], 'Set': []}
    ch = []
    pf = []
    for _, row in df.iterrows():
        if row['dds']:
            new_df['ev_0'].append(row['ev_0'])
            new_df['ev_1'].append(row['ev_1'])
            new_df['Set'].append('(C)DUS')
        if row['pf'] and row['dds']:
            pf.append((row['ev_0'], row['ev_1']))
        if row['ch'] and row['dds']:
            ch.append((row['ev_0'], row['ev_1']))
    sns.scatterplot(data=new_df, x="ev_0", y="ev_1", hue='Set', style="Set", palette=['#a31034', ], markers=['o'],
                    size='Set', sizes=(120, 40), alpha=0.8)
    pf = sorted(pf)
    sns.lineplot(x=[x[0] for x in pf], y=[x[1] for x in pf], label='PF', markers='-', color='black', linewidth=1.5)
    ch = sorted(ch)
    sns.lineplot(x=[x[0] for x in ch], y=[x[1] for x in ch], label='CH', color='grey', linewidth=1)

    sns.scatterplot(x=[coords_pf[0]], y=[coords_pf[1]], marker='X', s=40, color='gold')
    sns.scatterplot(x=[coords_all[0]], y=[coords_all[1]], marker='X', s=40, color='gold')

    if annotate:
        plt.annotate(f'{u_pf:.1f}', coords_pf - offset_pf, fontsize=10)
        plt.annotate(f'{u_all:.1f}', coords_all - offset_all, fontsize=10)

    plt.grid(alpha=0.25)
    plt.xlabel('Expected value of objective 1')
    plt.ylabel('Expected value of objective 2')
    plt.savefig(f'case_study_{num}.pdf')
    plt.clf()


def case_study(df, u_func, dists, offset_pf, offset_all, num):
    u_lst = [dist.expected_utility(u_func) for dist in dists]
    argmax_u = np.argmax(u_lst)
    u_pf_lst = []
    for i, u in enumerate(u_lst):
        if df.iloc[i]['pf'] == 1:
            u_pf_lst.append(u)
        else:
            u_pf_lst.append(-np.inf)
    argmax_u_pf = np.argmax(u_pf_lst)

    coords_pf = df.iloc[argmax_u_pf][['ev_0', 'ev_1']].to_numpy()
    coords_all = df.iloc[argmax_u][['ev_0', 'ev_1']].to_numpy()
    u_pf = u_pf_lst[argmax_u_pf]
    u_all = u_lst[argmax_u]

    print(f'Max utility in PF: {u_pf}')
    print(f'Max utility: {u_all}')
    print(f'-----------------')
    make_overlapping_scatterplot(df,
                                 coords_pf=coords_pf,
                                 offset_pf=offset_pf,
                                 u_pf=u_pf,
                                 coords_all=coords_all,
                                 offset_all=offset_all,
                                 u_all=u_all,
                                 num=num)


if __name__ == '__main__':
    args = parse_args()
    dists_dir = os.path.join(args.log_dir, args.env, str(args.seed), args.alg)
    dists = load_dists(dists_dir, MCD)
    df = pd.read_csv(os.path.join(dists_dir, 'pruning_results.csv'))

    offset_pf1 = np.array([0.16, 0.11])
    offset_all1 = np.array([0.16, 0.11])
    case_study(df, u_func1, dists, offset_pf1, offset_all1, num=1)

    offset_pf2 = np.array([0.16, 0.11])
    offset_all2 = np.array([0.13, 0.11])
    case_study(df, u_func2, dists, offset_pf2, offset_all2, num=2)
