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
    return parser


def prepareData(inputFile):
    df = pd.read_csv(inputFile, delim_whitespace=True)
    # convert cross-section and width from pb in fb
    df['cross_fb'] = df['cross'] * 1000.
    # rename columns
    df.rename(columns={'mass#6000055': 'mass', 'v0params#1': 'ct', 'width#6000055': 'width'}, inplace=True)
    return df


def makePlot(df):
    ampl.use_atlas_style(usetex=False)
    fig, ax = plt.subplots(figsize=(9,8))

    # create contour plot
    X, Y = np.meshgrid(np.sort(df.mass.unique()), np.sort(df.ct.unique()))
    Z = df.pivot(index='ct', columns='mass', values='cross_fb').values
    levels = [2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    cp = ax.contour(X, Y, Z, levels)
    plt.clabel(cp, inline=True, fontsize=8)

    # label plot
    plt.xlabel('$m_{V1}$ [GeV]')
    plt.ylabel('$c_{t}$')
    plt.text(0.7, 0.2, '$\\sigma$ [fb]', transform=ax.transAxes)
    plt.text(0.7, 0.1, 'p p > ttv1, v1 > tt', transform=ax.transAxes)
    plt.text(0.7, 0.05, '$\\sqrt{s}$ = 14 TeV', transform=ax.transAxes)

    # style plot
    ax.set_xlim(990, 2010)
    ax.set_ylim(0, 5.1)

    # save figure
    fig.savefig('plot_1604p07421_fig3.png')



def main():
    args = getArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO)

    data = prepareData(args.inputFile)
    logging.info(data)

    makePlot(data)


if __name__ == "__main__":
    main()
