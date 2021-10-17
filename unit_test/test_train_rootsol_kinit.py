import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import pathlib
import gzip
import pickle
from utility import instancetypes, modes
from models import *
import matplotlib.pyplot as plt


def train(model, data_loader, optimizer=None):
    """
    training function
    :param model:
    :param data_loader:
    :param optimizer:
    :return:
    """
    mean_loss = 0
    n_samples_precessed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            k_model = model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            k_init = batch.k_init
            loss = F.l1_loss(k_model.float(), k_init.float())
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss.item() * batch.num_graphs
            n_samples_precessed += batch.num_graphs
    mean_loss /= n_samples_precessed

    return mean_loss


def test(model, data_loader):
    n_samples_precessed = 0
    loss_list = []
    k_model_list = []
    k_init_list = []
    graph_index =[]
    for batch in data_loader:
        k_model = model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
        k_init = batch.k_init
        loss = F.mse_loss(k_model, k_init)

        if batch.num_graphs == 1:
            loss_list.append(loss.item())
            k_model_list.append(k_model.item())
            k_init_list.append(k_init)
            graph_index.append(n_samples_precessed)
            n_samples_precessed += 1

        else:

            for g in range(batch.num_graphs):
                loss_list.append(loss.item()[g])
                k_model_list.append(k_model[g])
                k_init_list.append(k_init(g))
                graph_index.append(n_samples_precessed)
                n_samples_precessed += 1

    loss_list = np.array(loss_list).reshape(-1)
    k_model_list = np.array(k_model_list).reshape(-1)
    k_init_list = np.array(k_init_list).reshape(-1)
    graph_index = np.array(graph_index).reshape(-1)

    loss_ave = loss_list.mean()
    k_model_ave = k_model_list.mean()
    k_init_ave = k_init_list.mean()

    return loss_ave, k_model_ave, k_init_ave


def loaddata(instancetype, mode, incumbent_mode):

    directory = './result/generated_instances/' + instancetype + '/' + mode + '/'
    directory_samples = directory + 'samples-' + incumbent_mode + '_kinit/'

    filename = 'sample-kinit-'+ instancetype +'-*.pkl'
    print(filename)
    sample_files = [str(path) for path in pathlib.Path(directory_samples).glob(filename)]
    train_files = sample_files[:int(0.8*len(sample_files))]
    valid_files = sample_files[int(0.8*len(sample_files)):]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=1, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=1, shuffle=False)

    return train_loader, valid_loader


SEED = 100
np.random.seed(SEED)
torch.manual_seed(SEED)

train_loaders = {}
val_loaders = {}

instancetype = instancetypes[11]
mode = modes[4]
incumbent_mode = 'rootsol'
saved_directory = './result/saved_models/'
pathlib.Path(saved_directory).mkdir(parents=True, exist_ok=True)

train_loader, valid_loader = loaddata(instancetype, mode, incumbent_mode)
train_loaders[instancetype] = train_loader
val_loaders[instancetype] = valid_loader

# instancetype = instancetypes[1]
# train_loader, valid_loader = loaddata(instancetype, mode, incumbent_mode)
#
# # temporal method for partial samples set
# directory = './result/generated_instances/' + instancetype + '/' + mode + '/'
# directory_samples = directory + 'samples-' + incumbent_mode + '_kinit/'
#
# filename = 'sample-kinit-'+ instancetype +'-*.pkl'
# print(filename)
# sample_files = [str(path) for path in pathlib.Path(directory_samples).glob(filename)]
# train_files = sample_files
# valid_files = sample_files
#
# train_data = GraphDataset(train_files)
# train_loader = torch_geometric.data.DataLoader(train_data, batch_size=1, shuffle=True)
# valid_data = GraphDataset(valid_files)
# valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=1, shuffle=False)
#
# train_loaders[instancetype] = train_loader
# val_loaders[instancetype] = valid_loader


model_gnn = GNNPolicy()
# model_gnn2 = GNNPolicy()


train_instancetype = instancetypes[11]
valid_instancetype = instancetypes[11]
LEARNING_RATE = 0.0000001 # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001
EPOCHS = 20

optimizer = torch.optim.Adam(model_gnn.parameters(), lr = LEARNING_RATE)
k_init = []
k_model = []
loss = []
epochs = []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")

    if epoch==0:
        optim = None
    else:
        optim = optimizer

    train_loader = train_loaders[train_instancetype]
    train_loss = train(model_gnn, train_loader, optim)
    print(f"Train loss: {train_loss:0.6f}")

    # torch.save(model_gnn.state_dict(), 'trained_params_' + train_instancetype + '.pth')
    # model_gnn2.load_state_dict(torch.load('trained_params_' + train_instancetype + '.pth'))

    valid_loader = val_loaders[valid_instancetype]
    valid_loss = train(model_gnn, valid_loader, None)
    print(f"Valid loss: {valid_loss:0.6f}")

    loss_ave, k_model_ave, k_init_ave = test(model_gnn, valid_loader)

    loss.append(loss_ave)
    k_model.append(k_model_ave)
    k_init.append(k_init_ave)
    epochs.append(epoch)

loss_np = np.array(loss).reshape(-1)
k_model_np = np.array(k_model).reshape(-1)
k_init_np = np.array(k_init).reshape(-1)
epochs_np = np.array(epochs).reshape(-1)

plt.close('all')
plt.clf()
fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
fig.suptitle("Test Result: prediction of initial k")
fig.subplots_adjust(top=0.5)
ax[0].set_title(valid_instancetype, loc='right')
ax[0].plot(epochs_np, loss_np)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel("loss")
ax[1].plot(epochs_np, k_model_np, label='k-prediction')

ax[1].plot(epochs_np, k_init_np, label='k-label')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel("k")
ax[1].set_ylim([0, 1.1])
ax[1].legend()
plt.show()

torch.save(model_gnn.state_dict(), saved_directory + 'trained_params_' + train_instancetype + '_' + incumbent_mode + '.pth')



