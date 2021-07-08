import numpy as np
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm
from utils import getDataFrame, cleanDataFrame
from utils import visualizeBDTScore

import torch
from torch_geometric.data import Data, InMemoryDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def getArgumentParser():
    parser = ArgumentParser()
    parser.add_argument('inputFile')
    return parser


def doGNN(df):
    class BSM4topsDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(BSM4topsDataset, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])
        @property
        def raw_file_names(self):
            return []
        @property
        def processed_file_names(self):
            return ['bsm4tops.dataset']
        def download(self):
            pass
        def process(self):
            data_list = []
            # loop over events and shuffle order of top quarks
            for event, new_df in tqdm(df.groupby(level=0)):
                new_df = new_df.sample(frac=1).reset_index(drop=True)
                edge_index = torch.tensor(
                    [[0, 0, 0, 1, 1, 2, 1, 2, 3, 2, 3, 3],
                     [1, 2, 3, 2, 3, 3, 0, 0, 0, 1, 1, 2]],
                    dtype=torch.long)

                x = torch.tensor([[new_df['Particle.PT'].to_numpy()[0], new_df['Particle.Eta'].to_numpy()[0], new_df['Particle.Phi'].to_numpy()[0], new_df['Particle.M'].to_numpy()[0]],
                                  [new_df['Particle.PT'].to_numpy()[1], new_df['Particle.Eta'].to_numpy()[1], new_df['Particle.Phi'].to_numpy()[1], new_df['Particle.M'].to_numpy()[0]],
                                  [new_df['Particle.PT'].to_numpy()[2], new_df['Particle.Eta'].to_numpy()[2], new_df['Particle.Phi'].to_numpy()[2], new_df['Particle.M'].to_numpy()[0]],
                                  [new_df['Particle.PT'].to_numpy()[3], new_df['Particle.Eta'].to_numpy()[3], new_df['Particle.Phi'].to_numpy()[3], new_df['Particle.M'].to_numpy()[0]]],
                                  dtype=torch.float)
                y = torch.LongTensor(new_df['resonance'].to_numpy())
                data = Data(x=x, y=y, edge_index=edge_index)
                data_list.append(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    dataset = BSM4topsDataset('data/')

    # shuffle and split into training, validation and testing
    dataset = dataset.shuffle()

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Get the first graph object in training dataset.
    data = dataset[0]

    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    # visualize graph
    visualizeGraph(data)

    # build GNN - shamelessly stolen and slightly modified from Javier Duarte's course
    # https://github.com/jmduarte/capstone-particle-physics-domain/blob/master/weeks/08-extending.ipynb
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric.transforms as T
    from torch_geometric.nn import EdgeConv, global_mean_pool
    from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
    from torch_scatter import scatter_mean
    from torch_geometric.nn import MetaLayer

    inputs = 3
    hidden = 12
    outputs = 1

    class EdgeBlock(torch.nn.Module):
        def __init__(self):
            super(EdgeBlock, self).__init__()
            self.edge_mlp = Seq(Lin(inputs*2, hidden), 
                                BatchNorm1d(hidden),
                                ReLU(),
                                Lin(hidden, hidden))

        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest], 1)
            return self.edge_mlp(out)
    class NodeBlock(torch.nn.Module):
        def __init__(self):
            super(NodeBlock, self).__init__()
            self.node_mlp_1 = Seq(Lin(inputs+hidden, hidden), 
                                  BatchNorm1d(hidden),
                                  ReLU(), 
                                  Lin(hidden, hidden))
            self.node_mlp_2 = Seq(Lin(inputs+hidden, hidden), 
                                  BatchNorm1d(hidden),
                                  ReLU(), 
                                  Lin(hidden, hidden))

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out], dim=1)
            return self.node_mlp_2(out)
    class GlobalBlock(torch.nn.Module):
        def __init__(self):
            super(GlobalBlock, self).__init__()
            self.global_mlp = Seq(Lin(hidden, hidden),                               
                                  BatchNorm1d(hidden),
                                  ReLU(), 
                                  Lin(hidden, outputs))

        def forward(self, x, edge_index, edge_attr, u, batch):
            out = scatter_mean(x, batch, dim=0)
            return self.global_mlp(out)
    class InteractionNetwork(torch.nn.Module):
        def __init__(self):
            super(InteractionNetwork, self).__init__()
            self.interactionnetwork = MetaLayer(EdgeBlock(), NodeBlock(), GlobalBlock())
            self.bn = BatchNorm1d(inputs)
            
        def forward(self, x, edge_index, batch):
            
            x = self.bn(x)
            x, edge_attr, u = self.interactionnetwork(x, edge_index, None, None, batch)
            return u

    model = InteractionNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    @torch.no_grad()
    def test(model,loader,total,batch_size,leave=False):
        model.eval()
        xentropy = nn.CrossEntropyLoss(reduction='mean')
        sum_loss = 0.
        t = tqdm(enumerate(loader),total=total/batch_size,leave=leave)
        for i,data in t:
            data = data.to(device)
            y = torch.argmax(data.y,dim=1)
            batch_output = model(data.x, data.edge_index, data.batch)
            batch_loss_item = xentropy(batch_output, y).item()
            sum_loss += batch_loss_item
            t.set_description("loss = %.5f" % (batch_loss_item))
            t.refresh() # to show immediately the update
        return sum_loss/(i+1)

    def train(model, optimizer, loader, total, batch_size,leave=False):
        model.train()
        xentropy = nn.CrossEntropyLoss(reduction='mean')
        sum_loss = 0.
        t = tqdm(enumerate(loader),total=total/batch_size,leave=leave)
        for i, data in t:
            data = data.to(device)
            y = torch.argmax(data.y,dim=1)
            optimizer.zero_grad()
            batch_output = model(data.x, data.edge_index, data.batch)
            batch_loss = xentropy(batch_output, y)
            batch_loss.backward()
            batch_loss_item = batch_loss.item()
            t.set_description("loss = %.5f" % batch_loss_item)
            t.refresh() # to show immediately the update
            sum_loss += batch_loss_item
            optimizer.step()
        return sum_loss/(i+1)

    from torch_geometric.data import Data, DataListLoader, Batch
    from torch.utils.data import random_split

    def collate(items):
        return items
        l = sum(items, [])
        return Batch.from_data_list(l)


    torch.manual_seed(0)
    valid_frac = 0.10
    full_length = len(dataset)
    valid_num = int(valid_frac*full_length)
    batch_size = 32

    train_dataset, valid_dataset = random_split(dataset, [full_length-valid_num,valid_num])

    train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate
    # test_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    # test_loader.collate_fn = collate

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    # test_samples = len(valid_dataset)

    import os.path as osp
    n_epochs = 10
    stale_epochs = 0
    best_valid_loss = 99999
    patience = 5
    t = tqdm(range(0, n_epochs))

    for epoch in t:
        # loss = train(model, optimizer, train_loader, train_samples, batch_size,leave=bool(epoch==n_epochs-1))
        loss = train(model, optimizer, train_dataset, train_samples, batch_size,leave=bool(epoch==n_epochs-1))
        # valid_loss = test(model, valid_loader, valid_samples, batch_size,leave=bool(epoch==n_epochs-1))
        valid_loss = test(model, valid_dataset, valid_samples, batch_size,leave=bool(epoch==n_epochs-1))
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('           Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            modpath = osp.join('interactionnetwork_best.pth')
            print('New best model saved to:',modpath)
            torch.save(model.state_dict(),modpath)
            stale_epochs = 0
        else:
            print('Stale epoch')
            stale_epochs += 1
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break


    # model.eval()
    # t = tqdm(enumerate(test_loader),total=test_samples/batch_size)
    # y_test = []
    # y_predict = []
    # for i,data in t:
    #     data = data.to(device)    
    #     batch_output = model(data.x, data.edge_index, data.batch)    
    #     y_predict.append(batch_output.detach().cpu().numpy())
    #     y_test.append(data.y.cpu().numpy())
    # y_test = np.concatenate(y_test)
    # y_predict = np.concatenate(y_predict)

    # from sklearn.metrics import roc_curve, auc
    # import matplotlib.pyplot as plt
    # import mplhep as hep
    # plt.style.use(hep.style.ROOT)
    # # create ROC curves
    # fpr_gnn, tpr_gnn, threshold_gnn = roc_curve(y_test[:,1], y_predict[:,1])
        
    # # plot ROC curves
    # plt.figure()
    # plt.plot(tpr_gnn, fpr_gnn, lw=2.5, label="GNN, AUC = {:.1f}%".format(auc(fpr_gnn,tpr_gnn)*100))
    # plt.xlabel(r'True positive rate')
    # plt.ylabel(r'False positive rate')
    # plt.semilogy()
    # plt.ylim(0.001,1)
    # plt.xlim(0,1)
    # plt.grid(True)
    # plt.legend(loc='upper left')
    # plt.show()



def doBDT(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, confusion_matrix


    target_names = ['resonance']
    y = df.pop('resonance').values
    feature_names = ['Particle.PT', 'Particle.Eta', 'Particle.Phi', 'Particle.M']
    X = df

    # split in training and testing part
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


    # set up classifier
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             algorithm="SAMME",
                             n_estimators=200)
    # train classifier
    bdt.fit(X_train, y_train)

    # plot BDT score for training dataset
    twoclass_output_train = bdt.decision_function(X_train)
    class_names = ['resonance', 'spectator']
    plot_colors = ['blue', 'orange']
    plot_range = (twoclass_output_train.min(), twoclass_output_train.max())
    visualizeBDTScore(twoclass_output_train, y_train, class_names, plot_range, plot_colors, 'bdt_score_train.png')

    # plot BDT score for test dataset
    twoclass_output_test = bdt.decision_function(X_test)
    plot_range = (twoclass_output_test.min(), twoclass_output_test.max())
    visualizeBDTScore(twoclass_output_test, y_test, class_names, plot_range, plot_colors, 'bdt_score_test.png')


    # ROC curve
    from sklearn.metrics import roc_auc_score, roc_curve
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], twoclass_output_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), twoclass_output_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def main():
    # get histograms from files
    args = getArgumentParser().parse_args()

    df = getDataFrame(args.inputFile)
    df = cleanDataFrame(df)

    # doGNN(df)

    doBDT(df)



if __name__ == '__main__':
    main()
