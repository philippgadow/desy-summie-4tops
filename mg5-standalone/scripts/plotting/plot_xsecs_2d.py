import os
import argparse
import pandas as pd
import matplotlib
# configure matplotlib for use without X server (e.g. in batch mode)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import atlas_mpl_style as ampl
import seaborn as sns


def getArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', help='Input file in CSV format with data.')
    parser.add_argument('--process', choices=['ttv1', 'tjv1', 'tWv1'], help='Name of process.')
    parser.add_argument('--ratio', action='store_true', help='Compute ratio to ttv1')
    return parser


def main():
    # get arguments from command line
    args = getArgumentParser().parse_args()

    # load data sets and convert cross-section to fb
    data = pd.read_csv(args.inputFile, delim_whitespace=True)
    data['cross'] = data['cross'] * 1000.
    data = data.pivot(index='v1params#1', columns='mass#6000055', values='cross')

    if args.ratio:
        data_ratio = pd.read_csv(os.path.join('data', 'proc_bsm4top-13TeV_ttv1_mv1-scan_wv1-auto_t1-1_theta1-scan-scanresults.txt'), delim_whitespace=True)
        data_ratio['cross'] = data_ratio['cross'] * 1000.
        data_ratio = data_ratio.pivot(index='v1params#1', columns='mass#6000055', values='cross')
        data = data / data_ratio

    # plot heatmaps
    plt.style.use('print')

    f, ax = plt.subplots(figsize=(9, 6))
    label = 'cross-section (LO) [fb]'
    if args.ratio: label = 'cross-section ratio to ttv1'
    sns.heatmap(data, annot=True, linewidths=.5, ax=ax, cbar_kws={'label': label})
    plt.title(args.process)
    plt.xlabel('m(v1) [GeV]')
    plt.ylabel('$\\tan \\theta_{1}$')
    plt.gca().invert_yaxis()

    ratio = '_ratio_to_ttv1' if args.ratio else ''
    f.savefig(f'xsec_{args.process}{ratio}.png')
    plt.close()


if __name__ == '__main__':
    main()
