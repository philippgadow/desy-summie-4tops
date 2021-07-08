import uproot
import numpy as np


# Input / Output helper functions

def getDataFrame(filename, treename='LHEF'):
    """Utility function providing pandas dataframe from ROOT input file."""
    with uproot.open(filename) as f:
        tree = f[treename]['Particle']
        df = tree.arrays(
                [
                    "Particle.PID", "Particle.Mother1", "Particle.Mother2",
                    "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.M"
                ],
                library="pd")
    return df


def cleanDataFrame(df):
    """Utility function to clean pandas dataframe and process information about resonance candidate."""
    df['resonance'] = df.apply(lambda row: row["Particle.Mother1"] == row["Particle.Mother2"], axis=1)
    df = df[np.abs(df["Particle.PID"]) == 6]
    df = df.drop(["Particle.Mother1", "Particle.Mother2", "Particle.PID"], axis=1)
    return df
