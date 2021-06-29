## Search for new resonances in four-top-quark final state

The top quark is the heaviest elementary particle known to date. The large Yukawa coupling between the top quark and the Higgs boson is an important building block in many models describing phenomena beyond the standard model.
At the Large Hadron Collider it is possible to investigate very rare processes involving top quarks, such as the four-top-quark final state. The production cross-section in the Standard Model is about 12 femto-barn. Recent searches have measured roughly twice the observed cross-section, giving rise to investigate the four-top-quark final state with more scrutiny.

An interesting research problem is the hypothesis of heavy resonances which couple preferrably to top quarks, decay to pairs of top quarks and are produced in a four top quark final state. The top quarks produced in such a process are two top quarks which originate from the decay of the resonance and two so-called spectator tops which originate from the associated production of the resonance.

Exploiting the kinematics and invariant mass of the resonance top quarks allows inferring the properties of a resonance (if present in the data), e.g. its mass which would manifest as a peak in the invariant top quark pair mass spectrum.

This is not easy! A top, being an elementary particle with mass close to that of a gold atom, decays almost instantly after its production to a W boson and a bottom quark. The W boson itself also decays, either hadronically to a light quark pair or leptonically to a lepton and a neutrino. You can imagine that a four-top quark final state can be quite messy to reconstruct!

A challenge in the reconstruction therefore is the correct association of top quark decay products to the top quarks and then the association of the correct top quarks to the resonance.

We will study reconstruction techniques for the four-top-quark process using machine learning techniques.

Our strategy involves several steps:

1. Simulate how events involving such a top-philic resonance resulting in a four-top-quark final state look like in our detector

2. Plot several observables to get familiar with the final state and look at a few exemplary event displays.

3. Construct a graph neural network to reconstruct the four-top-quark final state.




### Signal generation

I have prepared a github project with all necessary ingredients to simulate the signal process:

1. Simulation of the hard scattering process at leading order in QCD using MadGraphAMC@NLO
2. Description of the hadronisation and parton shower using Pythia8
3. Simulation of the ATLAS detector using Delphes

Three specialised programs which are commonly used in High Energy Physics are used.
You don't need to run them individually, as they are already interfaced within MadGraphAMC@NLO and can very conveniently be controlled using so-called configuration "cards".

If you are happy with running the signal generation as a black box, follow the instructions below.
Of course, you will learn what is happening behind the scenes in the black box in due time!


First, you need to install Docker on your laptop or local machine.
Go to https://www.docker.com/get-started and install docker.


Then, execute in a terminal:

```bash
mkdir output
docker run -it --rm --volume $PWD/output:/var/MG_outputs --volume /tmp:/tmp philippgadow/mg5-standalone
/home/hep/MG5_aMC_v2_9_1_2/bin/mg5_aMC /home/hep/commands/atlas_ufo/generate_bsm4top_atlasufo_ttv1_mv1-1500_wv1-auto.cmnd
exit
```

As a result, you will find in `output` a folder with the results of the simulation.


