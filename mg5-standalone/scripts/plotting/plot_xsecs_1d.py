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
    # convert cross-section from pb in fb
    df['cross_fb'] = df['cross'] * 1000.
    # rename columns
    df.rename(columns={'v0params#1': 'ct', 'width#6000055': 'width'}, inplace=True)
    return df


def makePlot(df, process):
    ampl.use_atlas_style(usetex=False)
    fig, ax = plt.subplots(figsize=(9,8))

    ax.plot(df['ct'], df['cross'], 'o', color='black', label='MadGraph')

    # fit polynomial
    f = np.polynomial.Polynomial.fit(df['ct'], df['cross'], 4)
    p0, p1, p2, p3, p4 = f.coef
    ax.plot(df['ct'], f(df['ct']), color='red', label='Polynomial (deg=4) fit')

    # label plot
    plt.xlabel('$c_{t}$')
    plt.ylabel(f'$\\sigma$ ({process}) [fb]')
    plt.text(0.05, 0.75, f'p p > {process}, v1 > tt, $\\sqrt{{s}}$ = 13 TeV, $\\theta$ = $\\pi / 4$', transform=ax.transAxes)
    plt.text(0.05, 0.65, f'$\\sigma(x)$ = {p0:.3f} + ({p1:.3f}) $x$ + ({p2:.3f}) $x^{{2}}$  + ({p3:.3f}) $x^{{3}}$ + ({p4:.3f}) $x^{{4}}$', transform=ax.transAxes)

    plt.legend()

    # style plot
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.03)

    # save figure
    fig.savefig(f'plot_xsec_ct_{process}.png')



def main():
    args = getArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO)

    data = prepareData(args.inputFile)
    logging.info(data)

    makePlot(data, args.process)


if __name__ == "__main__":
    main()
