import hist
import vector
import uproot
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def getArgumentParser():
    parser = ArgumentParser()
    parser.add_argument('inputFile')
    return parser


def getArrays(filename, treename='LHEF'):
    with uproot.open(filename) as f:
        particles = f[treename]['Particle']

        p_status = particles['Particle.Status'].array()
        p_pid = particles['Particle.PID'].array()
        p_mother1 = particles['Particle.Mother1'].array()
        p_mother2 = particles['Particle.Mother2'].array()

        top1_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-1],
            "eta": particles['Particle.Eta'].array()[:,-1],
            "phi": particles['Particle.Phi'].array()[:,-1],
            "M": particles['Particle.M'].array()[:,-1]
        })
        top1_resonance = (
            particles['Particle.Mother1'].array()[:,-1] == \
            particles['Particle.Mother2'].array()[:,-1]
        )

        top2_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-2],
            "eta": particles['Particle.Eta'].array()[:,-2],
            "phi": particles['Particle.Phi'].array()[:,-2],
            "M": particles['Particle.M'].array()[:,-2]
        })
        top2_resonance = (
            particles['Particle.Mother1'].array()[:,-2] == \
            particles['Particle.Mother2'].array()[:,-2]
        )

        top3_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-3],
            "eta": particles['Particle.Eta'].array()[:,-3],
            "phi": particles['Particle.Phi'].array()[:,-3],
            "M": particles['Particle.M'].array()[:,-3]
        })
        top3_resonance = (
            particles['Particle.Mother1'].array()[:,-3] == \
            particles['Particle.Mother2'].array()[:,-3]
        )

        top4_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-4],
            "eta": particles['Particle.Eta'].array()[:,-4],
            "phi": particles['Particle.Phi'].array()[:,-4],
            "M": particles['Particle.M'].array()[:,-4]
        })
        top4_resonance = (
            particles['Particle.Mother1'].array()[:,-4] == \
            particles['Particle.Mother2'].array()[:,-4]
        )

    return {
        'top1': {
            "p4": top1_p4,
            "resonance": top1_resonance
        },
        'top2': {
            "p4": top2_p4,
            "resonance": top2_resonance
        },
        'top3': {
            "p4": top3_p4,
            "resonance": top3_resonance
        },
        'top4': {
            "p4": top4_p4,
            "resonance": top4_resonance
        },
    }


def plotEventVariables(tops_p4):
    # histograms
    h_mass_resonance = hist.Hist.new.Reg(30, 1000, 3200).Int64()
    h_mass_spectator = hist.Hist.new.Reg(30, 1000, 3200).Int64()
    h_dRtt_resonance = hist.Hist.new.Reg(30, 0, 6).Int64()
    h_dRtt_spectator = hist.Hist.new.Reg(30, 0, 6).Int64()

    inv_mass_resonance = (tops_p4[0] + tops_p4[1]).mass
    inv_mass_spectator = (tops_p4[2] + tops_p4[3]).mass
    delta_R_resonance = tops_p4[0].deltaR(tops_p4[1])
    delta_R_spectator = tops_p4[2].deltaR(tops_p4[3])

    h_mass_resonance.fill(inv_mass_resonance)
    h_mass_spectator.fill(inv_mass_spectator)
    h_dRtt_resonance.fill(delta_R_resonance)
    h_dRtt_spectator.fill(delta_R_spectator)

    # style
    hep.style.use(hep.style.ATLAS)

    # plots
    fig, ax = plt.subplots()
    h_mass_resonance.plot(ax=ax, label=r'$m=1.5$ TeV, $c_{t}=1$, $\theta=0.8$')
    plt.xlabel(r'$m_{tt}$ (resonance) [GeV]')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('test_mass_resonance.png')
    plt.close()

    fig, ax = plt.subplots()
    h_mass_spectator.plot(ax=ax, label=r'$m=1.5$ TeV, $c_{t}=1$, $\theta=0.8$', color='orange')
    plt.xlabel(r'$m_{tt}$ (spectator) [GeV]')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('test_mass_spectator.png')
    plt.close()

    fig, ax = plt.subplots()
    h_dRtt_resonance.plot(ax=ax, label=r'$m=1.5$ TeV, $c_{t}=1$, $\theta=0.8$')
    plt.xlabel(r'$\Delta R(tt)$ (resonance)')
    plt.ylabel('Events')
    plt.legend(loc=2)
    fig.savefig('test_dRtt_resonance.png')
    plt.close()

    fig, ax = plt.subplots()
    h_dRtt_spectator.plot(ax=ax, label=r'$m=1.5$ TeV, $c_{t}=1$, $\theta=0.8$', color='orange')
    plt.xlabel(r'$\Delta R(tt)$ (spectator)')
    plt.ylabel('Events')
    plt.legend(loc=2)
    fig.savefig('test_dRtt_spectator.png')
    plt.close()


def plotTopVariables(tops_p4):
    h_top_pt = []
    h_top_eta = []
    h_top_phi = []
    # loop over four top quarks and plot top quark kinematics
    for i, p4 in enumerate(tops_p4):
        # histograms for top quark four-vector
        h_top_pt.append(hist.Hist.new.Reg(30, 0, 1000).Int64())
        h_top_eta.append(hist.Hist.new.Reg(30, -5, 5).Int64())
        h_top_phi.append(hist.Hist.new.Reg(30, -4, 4).Int64())

        h_top_pt[i].fill(tops_p4[i].pt)
        h_top_eta[i].fill(tops_p4[i].eta)
        h_top_phi[i].fill(tops_p4[i].phi)

    # style
    hep.style.use(hep.style.ATLAS)

    # plots
    fig, ax = plt.subplots()
    for i, h in enumerate(h_top_pt):
        h.plot(ax=ax, label='top quark #{i}'.format(i=i))
    plt.xlabel(r'$p_{T}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('test_top_pt.png')
    plt.close()

    fig, ax = plt.subplots()
    for i, h in enumerate(h_top_eta):
        h.plot(ax=ax, label='top quark #{i}'.format(i=i))
    plt.xlabel(r'$\eta$')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('test_top_eta.png')
    plt.close()

    fig, ax = plt.subplots()
    for i, h in enumerate(h_top_phi):
        h.plot(ax=ax, label='top quark #{i}'.format(i=i))
    plt.xlabel(r'$\phi$')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('test_top_phi.png')
    plt.close()


def main():
    # get histograms from files
    args = getArgumentParser().parse_args()

    data = getArrays(args.inputFile)
    tops_p4 = [
        data['top1']['p4'],
        data['top2']['p4'],
        data['top3']['p4'],
        data['top4']['p4']
    ]
    plotTopVariables(tops_p4)
    plotEventVariables(tops_p4)


if __name__ == '__main__':
    main()
