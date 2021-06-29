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


def width_analytic(ct, theta1, mv1):
    mt = 172.5 # top mass in GeV
    return (np.power(ct, 2) * mv1 / (8 * np.pi)) * \
           np.sqrt(1 - (4 * np.power(mt, 2)) / (np.power(mv1, 2))) * \
           (1 - (np.power(mt, 2)) / (np.power(mv1, 2)) * (1 - 3 * np.sin(2 * theta1)))


def makePlot(df, process):
    ampl.use_atlas_style(usetex=False)
    fig, ax = plt.subplots(figsize=(9,8))

    # automatic width calculation
    ax.plot(df['ct'], df['width'], 'o', color='black', label='MadGraph AUTO')
    # analytic width calculation
    ax.plot(df['ct'], width_analytic(df['ct'], np.pi / 4, 1500.), color='red', label='analytic')

    # label plot
    plt.xlabel('$c_{t}$')
    plt.ylabel(f'$\\Gamma$ ({process}) [GeV]')
    plt.text(0.3, 0.05, f'p p > {process}, v1 > tt, $\\sqrt{{s}}$ = 13 TeV, $\\theta$ = $\\pi / 4$', transform=ax.transAxes)
    plt.legend()

    # style plot
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7000)

    # save figure
    fig.savefig(f'plot_width_ct_{process}.png')



def main():
    args = getArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO)

    data = prepareData(args.inputFile)
    logging.info(data)

    makePlot(data, args.process)


if __name__ == "__main__":
    main()
