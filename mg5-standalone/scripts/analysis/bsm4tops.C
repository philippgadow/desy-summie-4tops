/*
Simple macro showing how to access branches from the delphes output root file,
loop over events, store histograms in a root file and print them as image files.
root -l bsm4tops.C'("delphes_output.root")'
*/

#include "TH1.h"
#include "TSystem.h"

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#endif

#ifdef __CLING__
R__LOAD_LIBRARY(libfastjet)
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/Filter.hh"
#endif


int getRegion(int addjet, int bjet) {
  if (addjet == 2 && bjet == 2) return 1;
  if (addjet == 3 && bjet == 2) return 2;
  if (addjet >= 4 && bjet == 2) return 3;
  if (addjet == 2 && bjet == 3) return 4;
  if (addjet == 3 && bjet == 3) return 5;
  if (addjet >= 4 && bjet == 3) return 6;
  if (addjet == 2 && bjet >= 4) return 7;
  if (addjet == 3 && bjet >= 4) return 8;
  if (addjet >= 4 && bjet >= 4) return 9;
  return -1;
}

std::string getRegionName(int region) {
  if (region == 0) return "2+ add. jets, 2+ b-jets";
  if (region == 1) return "2 add. jets, 2 b-jets";
  if (region == 2) return "3 add. jets, 2 b-jets";
  if (region == 3) return "4+ add. jets, 2 b-jets";
  if (region == 4) return "2 add. jets, 3 b-jets";
  if (region == 5) return "3 add. jets, 3 b-jets";
  if (region == 6) return "4+ add. jets, 3 b-jets";
  if (region == 7) return "2 add. jets, 4+ b-jets";
  if (region == 8) return "3 add. jets, 4+ b-jets";
  if (region == 9) return "4+ add. jets, 4+ b-jets";
  return "";
}

//------------------------------------------------------------------------------

struct MyPlots {
  TH1 *fCutflow;
  TH1 *fCutflow_weighted;

  TH2 *fRegions;

  TH1 *fNJets[10];
  TH1 *fNAddJets[10];
  TH1 *fNBjets[10];
  TH1 *fNReclusteredJets[10];

  TH1 *fReclusteredJetM1[10];
  TH1 *fReclusteredJetM2[10];

  TH1 *fReclusteredJetPt1[10];
  TH1 *fReclusteredJetPt2[10];

  TH1 *fResonanceM[10];
};

//------------------------------------------------------------------------------

class ExRootResult;
class ExRootTreeReader;

//------------------------------------------------------------------------------

void BookHistograms(ExRootResult *result, MyPlots *plots) {
  // book cutflow histogram
  plots->fCutflow = result->AddHist1D(
      "cutflow", "cutflow",
      "cutflow", "Events",
      17, -7.5, 9.5);
  plots->fCutflow_weighted = result->AddHist1D(
      "cutflow_weighted", "cutflow_weighted",
      "cutflow_weighted", "Events",
      17, -7.5, 9.5);
  const char *cuts[17] = {"Initial events",
                          "Lepton",
                          "1 Lepton",
                          "2+ jets",
                          "2+ add. jets",
                          "2+ b-jets",
                          "2+ top-jets",
                          "Preselection",
                          "2a 2b",
                          "2a 3b",
                          "2a 4+b",
                          "3a 2b",
                          "3a 3b",
                          "3a 4+b",
                          "4+a 2b",
                          "4+a 3b",
                          "4+a 4+b"};
  for (int i = 1; i < 18; i++) plots->fCutflow->GetXaxis()->SetBinLabel(i, cuts[i-1]);
  for (int i = 1; i < 18; i++) plots->fCutflow_weighted->GetXaxis()->SetBinLabel(i, cuts[i-1]);

  // region overview
  plots->fRegions = result->AddHist2D(
    ("regions", "Events in region",
    "Number of b-jets", "Number of additional jets",
    3, 1.5, 4.5, 3, 1.5, 4.5);

  // book histograms for all regions
  // 0: inclusive (2+ addjet, 2+ bjet)
  // 1-9: signal, control, and validation regions
  for (int i = 0; i < 10; i++) {

    // add box with region name
    TPaveText* comment = result->AddComment(0.64, 0.86, 0.98, 0.98);
    comment->AddText(getRegionName(i).c_str());

    plots->fNJets[i] = result->AddHist1D(
      ("jets_central_N_" + std::to_string(i)).c_str(), "Jet multiplicity",
      "Number of jets", "Events",
      16, -.5, 15.5);
    result->Attach(plots->fNJets[i], comment);

    plots->fNAddJets[i] = result->AddHist1D(
      ("jets_add_N_" + std::to_string(i)).c_str(), "Additional jet multiplicity",
      "Number of additional jets", "Events",
      10, -.5, 9.5);
    result->Attach(plots->fNAddJets[i], comment);

    plots->fNBjets[i] = result->AddHist1D(
      ("jets_bjet_N_" + std::to_string(i)).c_str(), "B-jet multiplicity",
      "Number of b-jets", "Events",
      10, -.5, 9.5);
    result->Attach(plots->fNBjets[i], comment);

    plots->fNReclusteredJets[i] = result->AddHist1D(
      ("jets_reclusteredjet_N_" + std::to_string(i)).c_str(), "Reclustered jet multiplicity",
      "Number of reclustered jets", "Events",
      5, -.5, 4.5);
    result->Attach(plots->fNReclusteredJets[i], comment);

    plots->fReclusteredJetM1[i] = result->AddHist1D(
      ("jet_reclustered_m1_" + std::to_string(i)).c_str(), "Leading reclustered jet mass",
      "Leading reclustered jet mass [GeV]", "Events",
      10, 0.0, 500.0);
    result->Attach(plots->fReclusteredJetM1[i], comment);

    plots->fReclusteredJetM2[i] = result->AddHist1D(
      ("jet_reclustered_m2_" + std::to_string(i)).c_str(), "Sub-leading reclustered jet mass",
      "Sub-leading reclustered jet mass [GeV]", "Events",
      10, 0.0, 500.0);
    result->Attach(plots->fReclusteredJetM2[i], comment);

    plots->fReclusteredJetPt1[i] = result->AddHist1D(
      ("jet_reclustered_pt1_" + std::to_string(i)).c_str(), "Leading reclustered jet pt",
      "Leading reclustered jet p_{T} [GeV]", "Events",
      10, 0.0, 1000.0);
    result->Attach(plots->fReclusteredJetPt1[i], comment);

    plots->fReclusteredJetPt2[i] = result->AddHist1D(
      ("jet_reclustered_pt2_" + std::to_string(i)).c_str(), "Sub-leading reclustered jet pt",
      "Sub-leading reclustered jet p_{T} [GeV]", "Events",
      10, 0.0, 1000.0);
    result->Attach(plots->fReclusteredJetPt2[i], comment);

    plots->fResonanceM[i] = result->AddHist1D(
      ("resonance_m_" + std::to_string(i)).c_str(), "Reconstructed resonance mass",
      "m(tt) [GeV]", "Events",
      40, 0.0, 4000.0);
    result->Attach(plots->fResonanceM[i], comment);
  }
}

//------------------------------------------------------------------------------

void AnalyseEvents(ExRootTreeReader *treeReader, MyPlots *plots) {
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchElectron = treeReader->UseBranch("Electron");
  TClonesArray *branchMuon = treeReader->UseBranch("Muon");
  TClonesArray *branchWeight = treeReader->UseBranch("Weight");

  Long64_t allEntries = treeReader->GetEntries();
  Long64_t preselectionEntries = 0;
  Long64_t controlregionEntries = 0;
  Long64_t signalregion4a4bEntries = 0;


  std::cout << "** Chain contains " << allEntries << " events" << std::endl;

  Jet *jet;
  MissingET *met;
  Electron *electron;
  Muon *muon;

  Long64_t entry;
  Int_t i;
  TLorentzVector p4;
  float weight;
  float sumOfWeights = 0;
  float luminosity = 139.; // fb
  // dummy cross section of 1 pb
  float xsec = 1000.0; // fb
  // ttv1
  // float xsec = 0.45355093; // fb
  // tjv1
  // float xsec = 0.21494597; // fb
  // tWv1
  // float xsec = 0.25056313; // fb

  // Get sum of weights
  for (entry = 0; entry < allEntries; ++entry) {
    treeReader->ReadEntry(entry);
    weight = ((Weight*) branchWeight->At(0))->Weight;
    sumOfWeights += weight;
  }

  std::cout << "Sum of weights " << sumOfWeights << " events" << std::endl;

  // Loop over all events
  for (entry = 0; entry < allEntries; ++entry) {
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);

    // Event baseline selection: compute variables
    int nSignalLeptons = 0;
    int nBaselineLeptons = 0;
    int nJets = 0;
    int nReclusteredJets = 0;
    int nBjets = 0;
    int nAddJets = 0;

    weight = ((Weight*) branchWeight->At(0))->Weight * \
             luminosity * xsec / sumOfWeights;

    // Lepton multiplicity
    for (i = 0; i < branchElectron->GetEntriesFast(); ++i) {
      electron = (Electron*) branchElectron->At(i);
      if (electron->PT > 28. && (abs(electron->Eta) < 1.37 || (abs(electron->Eta) > 1.52 && abs(electron->Eta) < 2.47))) ++nSignalLeptons;
      if (electron->PT > 10.) ++nBaselineLeptons;
    }

    for (i = 0; i < branchMuon->GetEntriesFast(); ++i) {
      muon = (Muon*) branchMuon->At(i);
      if (muon->PT > 28. && abs(muon->Eta)< 2.5) ++nSignalLeptons;
      if (muon->PT > 10.) ++nBaselineLeptons;
    }

    // Jet selection
    std::vector<fastjet::PseudoJet> jets;
    for (i = 0; i < branchJet->GetEntriesFast(); ++i) {
      jet = (Jet*) branchJet->At(i);
      if (jet->PT < 25. || abs(jet->Eta) > 2.5) continue;
      p4.SetPtEtaPhiM(jet->PT, jet->Eta, jet->Phi, jet->Mass);
      jets.push_back(fastjet::PseudoJet(p4.Px(), p4.Py(), p4.Pz(), p4.E()));
      // jet multiplicity
      ++nJets;
      if (jet->BTag) ++nBjets;
    }

    // jet reclustering
    double R = 1.0;
    fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
    fastjet::ClusterSequence cs(jets, jet_def);
    std::vector<fastjet::PseudoJet> proto_reclustered_jets = fastjet::sorted_by_pt(cs.inclusive_jets());

    fastjet::Selector select_pt = fastjet::SelectorPtMin(300.0);
    fastjet::Selector select_mass = fastjet::SelectorMassMin(100.0);
    fastjet::Selector select_both = select_pt && select_mass;

    // jet trimming
    double Rtrim = 0.2;
    double ptfrac = 0.05;
    fastjet::Filter trimmer(fastjet::JetDefinition(fastjet::kt_algorithm, Rtrim), fastjet::SelectorPtFractionMin(ptfrac));
    std::vector<fastjet::PseudoJet> reclustered_jets;
    for (auto ijet : select_both(proto_reclustered_jets)) {
      fastjet::PseudoJet groomed_jet = trimmer(ijet);
      reclustered_jets.push_back(groomed_jet);
    }

    // reclustered jet multiplicity and additional jet multiplicity
    nReclusteredJets = reclustered_jets.size();
    for (auto ijet : jets) {
      // check if jet overlaps with reclustered jet
      bool overlaps = false;
      for (auto rjet : reclustered_jets) {
        if (ijet.delta_R(rjet) < 1.0) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) ++nAddJets;
    }

    // Event baseline selection: reject events
    //initial events
    plots->fCutflow->Fill(-7);
    plots->fCutflow_weighted->Fill(-7, weight);
    // signal lepton cut
    if (!(nSignalLeptons == 1)) continue;
    plots->fCutflow->Fill(-6);
    plots->fCutflow_weighted->Fill(-6, weight);
    // no additional leptons cut
    if (!(nBaselineLeptons == 1)) continue;
    plots->fCutflow->Fill(-5);
    plots->fCutflow_weighted->Fill(-5, weight);
    // central jets cut
    if (!(nJets >= 2)) continue;
    plots->fCutflow->Fill(-4);
    plots->fCutflow_weighted->Fill(-4, weight);
    // additional jets cut
    if (!(nAddJets >= 2)) continue;
    plots->fCutflow->Fill(-3);
    plots->fCutflow_weighted->Fill(-3, weight);
    // b-jets cut
    if (!(nBjets >= 2)) continue;
    plots->fCutflow->Fill(-2);
    plots->fCutflow_weighted->Fill(-2, weight);
    // reclustered jets cut
    if (!(nReclusteredJets >= 2)) continue;
    plots->fCutflow->Fill(-1);
    plots->fCutflow_weighted->Fill(-1, weight);
    // preselection
    plots->fCutflow->Fill(0.);
    plots->fCutflow_weighted->Fill(0., weight);
    // individual regions
    if (nAddJets == 2 && nBjets == 2) plots->fCutflow->Fill(1);
    if (nAddJets == 2 && nBjets == 2) plots->fCutflow_weighted->Fill(1, weight);
    if (nAddJets == 3 && nBjets == 2) plots->fCutflow->Fill(2);
    if (nAddJets == 3 && nBjets == 2) plots->fCutflow_weighted->Fill(2, weight);
    if (nAddJets >= 4 && nBjets == 2) plots->fCutflow->Fill(3);
    if (nAddJets >= 4 && nBjets == 2) plots->fCutflow_weighted->Fill(3, weight);
    if (nAddJets == 2 && nBjets == 3) plots->fCutflow->Fill(4);
    if (nAddJets == 2 && nBjets == 3) plots->fCutflow_weighted->Fill(4, weight);
    if (nAddJets == 3 && nBjets == 3) plots->fCutflow->Fill(5);
    if (nAddJets == 3 && nBjets == 3) plots->fCutflow_weighted->Fill(5, weight);
    if (nAddJets >= 4 && nBjets == 3) plots->fCutflow->Fill(6);
    if (nAddJets >= 4 && nBjets == 3) plots->fCutflow_weighted->Fill(6, weight);
    if (nAddJets == 2 && nBjets >= 4) plots->fCutflow->Fill(7);
    if (nAddJets == 2 && nBjets >= 4) plots->fCutflow_weighted->Fill(7, weight);
    if (nAddJets == 3 && nBjets >= 4) plots->fCutflow->Fill(8);
    if (nAddJets == 3 && nBjets >= 4) plots->fCutflow_weighted->Fill(8, weight);
    if (nAddJets >= 4 && nBjets >= 4) plots->fCutflow->Fill(9);
    if (nAddJets >= 4 && nBjets >= 4) plots->fCutflow_weighted->Fill(9, weight);

    // Increment event counter
    ++preselectionEntries;
    if (nAddJets == 2 && nBjets == 2) ++controlregionEntries;
    if (nAddJets >= 4 && nBjets >= 4) ++signalregion4a4bEntries;

    // Fill histograms
    int nBjets_capped = nBjets;
    if (nBjets_capped > 4) nBjets_capped = 4;
    int nAddJets_capped = nAddJets;
    if (nAddJets_capped > 4) nAddJets_capped = 4;
    plots->fRegions->Fill(nBjets_capped, nAddJets_capped, weight);

    int region = getRegion(nAddJets, nBjets);
    plots->fNJets[0]->Fill(nJets, weight);
    plots->fNJets[region]->Fill(nJets, weight);
    plots->fNAddJets[0]->Fill(nAddJets, weight);
    plots->fNAddJets[region]->Fill(nAddJets, weight);
    plots->fNBjets[0]->Fill(nBjets, weight);
    plots->fNBjets[region]->Fill(nBjets, weight);
    plots->fNReclusteredJets[0]->Fill(nReclusteredJets, weight);
    plots->fNReclusteredJets[region]->Fill(nReclusteredJets, weight);

    plots->fReclusteredJetM1[0]->Fill(reclustered_jets[0].m(), weight);
    plots->fReclusteredJetM1[region]->Fill(reclustered_jets[0].m(), weight);
    plots->fReclusteredJetM2[0]->Fill(reclustered_jets[1].m(), weight);
    plots->fReclusteredJetM2[region]->Fill(reclustered_jets[1].m(), weight);

    plots->fReclusteredJetPt1[0]->Fill(reclustered_jets[0].perp(), weight);
    plots->fReclusteredJetPt1[region]->Fill(reclustered_jets[0].perp(), weight);
    plots->fReclusteredJetPt2[0]->Fill(reclustered_jets[1].perp(), weight);
    plots->fReclusteredJetPt2[region]->Fill(reclustered_jets[1].perp(), weight);

    float m_ttbar = (reclustered_jets[0] + reclustered_jets[1]).m();
    plots->fResonanceM[0]->Fill(m_ttbar, weight);
    plots->fResonanceM[region]->Fill(m_ttbar, weight);
  }

  std::cout << "** Initial events " << allEntries << " events" << std::endl;
  std::cout << "** Preselection " << preselectionEntries << " events" << std::endl;
  std::cout << "** Control region " << controlregionEntries << " events" << std::endl;
  std::cout << "** Signal region " << signalregion4a4bEntries << " events" << std::endl;
  std::cout << "Sum of weights " << sumOfWeights << " events" << std::endl;
}

//------------------------------------------------------------------------------

void PrintHistograms(ExRootResult *result, MyPlots *plots) {
  result->Print("png");
}

//------------------------------------------------------------------------------

void bsm4tops(const char *inputFile) {
  gSystem->Load("libDelphes");
  gSystem->Load("libfastjet");
  gROOT->SetBatch();

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  MyPlots *plots = new MyPlots;

  BookHistograms(result, plots);

  AnalyseEvents(treeReader, plots);

  PrintHistograms(result, plots);

  result->Write("results.root");

  cout << "** Exiting..." << endl;

  delete plots;
  delete result;
  delete treeReader;
  delete chain;

  gApplication->Terminate();
}
//------------------------------------------------------------------------------