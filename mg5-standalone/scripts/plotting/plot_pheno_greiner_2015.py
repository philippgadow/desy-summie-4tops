import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
# configure matplotlib for use without X server (e.g. in batch mode)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import atlas_mpl_style as ampl


def getArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', help="MadGraph output file with cross-sections")
    parser.add_argument('process', choices=['ttv1', 'tjv1', 'tWv1'], help="Type of process (3 choices: ttv1, tjv1, tWv1)")
    return parser


def prepareData(inputFile):
    df = pd.read_csv(inputFile, delim_whitespace=True)
    # convert cross-section and width from pb in fb
    df['cross_fb'] = df['cross'] * 1000.
    # rename columns
    df.rename(columns={'mass#6000055': 'mass', 'v1params#1': 'theta1', 'width#6000055': 'width'}, inplace=True)
    return df


def makePlot(df, process):
    ampl.use_atlas_style(usetex=False)
    fig, ax = plt.subplots(figsize=(9,8))

    # create contour plot
    X, Y = np.meshgrid(np.sort(df.mass.unique()), np.sort(df.theta1.unique()))
    Z = df.pivot(index='theta1', columns='mass', values='cross_fb').values

    levels = None
    colors = None
    linestyles = None
    if process == 'tjv1':
        levels = [0.5, 3.0, 10, 30, 60]
        colors = ['purple', 'blue', 'green', 'red', 'black']
        linestyles = ['solid', 'dashdot', 'dashed', 'dotted', 'solid']
    if process == 'tWv1':
        levels = [1, 3, 10, 20, 40]
        colors = ['purple', 'blue', 'green', 'red', 'black']
        linestyles = ['solid', 'dashdot', 'dashed', 'dotted', 'solid']
    if process == 'ttv1':
        levels = [5, 10, 20, 60, 150]
        colors = ['purple', 'blue', 'green', 'red', 'black']
        linestyles = ['solid', 'dashdot', 'dashed', 'dotted', 'solid']

    cp = ax.contour(X, Y, Z, levels=levels, colors=colors, linestyles=linestyles)
    plt.clabel(cp, inline=True, fontsize=8)

    # label plot
    plt.xlabel('$m_{V1}$ [GeV]')
    plt.ylabel('$c_{t}$')
    plt.text(0.8, 0.9, '$\\sigma$ [fb]', transform=ax.transAxes)
    plt.text(0.3, 0.05, f'p p > {process}, v1 > tt, $\\sqrt{{s}}$ = 14 TeV, $c_{{t}}$ = 1', transform=ax.transAxes)

    # style plot
    ax.set_xlim(375, 1000)
    ax.set_ylim(0, 3.2)

    # save figure
    fig.savefig(f'plot_greiner2015_{process}.png')



def main():
    args = getArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO)

    data = prepareData(args.inputFile)
    logging.info(data)

    makePlot(data, args.process)


if __name__ == "__main__":
    main()
