import os
import numpy as np
import h5py

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from torch_geometric.nn import GCNConv, ARMAConv, SAGEConv, TAGConv, TransformerConv
from torch_geometric.nn import Sequential
from torchmetrics import F1Score, FBetaScore, Recall, Precision, R2Score


def convert_binary_array_to_index(binary_array):
    length_input = len(binary_array)
    new_array = []
    for i in range(length_input):
        if binary_array == True:
            new_array.append(i)
    return new_array


def apply_activation(x, activation):
    if activation == None:
        return x
    if activation == "None":
        return x
    if activation == "relu":
        return F.relu(x)


class gnn_snbs_surv(gDataset):
    def __init__(self, grid_path, task, task_type, use_dtype_double=False, slice_index=slice(0, 0), normalize_targets=False):
        super(gnn_snbs_surv, self).__init__()
        self.normalize_targets = normalize_targets
        self.start_index = slice_index.start + 1
        self.data_len = slice_index.stop - slice_index.start + 1

        self.path = grid_path
        # self.num_classes = 1
        self.task = task
        self.task_type = task_type
        if use_dtype_double == True:
            self.dtype = 'float64'
        else:
            self.dtype = 'float32'
        self.data = {}
        self.read_in_all_data()
        # self.positive_weight = self.compute_pos_weight()

    def read_targets(self):
        all_targets = {}
        file_targets = self.path / f'{self.task}.h5'
        hf = h5py.File(file_targets, 'r')
        for index_grid in range(self.start_index, self.start_index + self.data_len):
            all_targets[index_grid] = np.array(
                hf.get(str(index_grid)), dtype=self.dtype)
        return all_targets

    def read_in_all_data(self):
        targets = self.read_targets()
        file_to_read = str(self.path) + "/input_data.h5"
        f = h5py.File(file_to_read, 'r')
        dset_grids = f["grids"]
        for index_grid in range(self.start_index, self.start_index + self.data_len):
            node_features = np.array(dset_grids[str(index_grid)].get(
                "node_features"), dtype=self.dtype).transpose()
            edge_index = (np.array(dset_grids[str(index_grid)].get(
                "edge_index"), dtype='int64') - 1).transpose()
            edge_attr = np.array(
                dset_grids[str(index_grid)].get("edge_attr"), dtype=self.dtype)
            y = torch.tensor(targets[index_grid])
            self.data[index_grid - self.start_index] = gData(x=(torch.tensor(node_features).unsqueeze(-1)), edge_index=torch.tensor(
                edge_index), edge_attr=torch.tensor(edge_attr), y=y)

        if self.task_type in ['classification', 'regressionThresholding']:
            targets_all_in_one_array = targets[self.start_index]
            for index_grid in range(self.start_index+1, self.start_index + self.data_len):
                targets_all_in_one_array = np.concatenate(
                    (targets_all_in_one_array, targets[index_grid]))
            if self.task_type == "regressionThresholding":
                targets_classified = np.where(
                    targets_all_in_one_array < 15., 0., 1.)
            else:
                targets_classified = targets_all_in_one_array
            self.positive_weight = torch.tensor((np.size(
                targets_classified) - np.count_nonzero(targets_classified))/np.count_nonzero(targets_classified))

    def len(self):
        return len(self.data)

    def get(self, index):
        return self.data[index]


class ArmaConvModule(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, activation, num_internal_layers, num_internal_stacks, batch_norm=False, shared_weights=True, dropout=0.25):
        super(ArmaConvModule, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.conv = ARMAConv(in_channels=num_channels_in, out_channels=num_channels_out,
                             num_stacks=num_internal_stacks, num_layers=num_internal_layers, shared_weights=shared_weights, dropout=dropout)
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(num_channels_out)

    def forward(self, data, x):
        edge_index, edge_weight = data.edge_index, data.edge_attr

        x = self.conv(x, edge_index=edge_index,
                      edge_weight=edge_weight.float())
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        x = apply_activation(x, self.activation)
        return x


class GCNConvModule(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, activation, improved, batch_norm=False):
        super(GCNConvModule, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.conv = GCNConv(num_channels_in, num_channels_out, improved)
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(num_channels_out)

    def forward(self, data, x):
        edge_index, edge_weight = data.edge_index, data.edge_attr

        x = self.conv(x, edge_index=edge_index,
                      edge_weight=edge_weight.float())
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        x = apply_activation(x, self.activation)
        return x


class SAGEConvModule(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, activation, batch_norm=False):
        super(SAGEConvModule, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.conv = SAGEConv(num_channels_in, num_channels_out)
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(num_channels_out)

    def forward(self, data, x):
        edge_index, edge_weight = data.edge_index, data.edge_attr

        x = self.conv(x, edge_index)
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        x = apply_activation(x, self.activation)
        return x


class TAGConvModule(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, activation, K, batch_norm=False):
        super(TAGConvModule, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.conv = TAGConv(num_channels_in, num_channels_out, K=K)
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(num_channels_out)

    def forward(self, data, x):
        edge_index, edge_weight = data.edge_index, data.edge_attr

        x = self.conv(x, edge_index=edge_index,
                      edge_weight=edge_weight.float())
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        x = apply_activation(x, self.activation)
        return x


class TransformConvModule(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, activation, heads, dropout, batch_norm=False):
        super(TransformConvModule, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        # self.conv = TransformerConv(num_channels_in, num_channels_out, edge_dim=0, dropout=.1)
        self.conv = TransformerConv(
            num_channels_in, num_channels_out, heads=heads, dropout=dropout)
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(num_channels_out)

    def forward(self, data, x):
        edge_index, edge_weight = data.edge_index, data.edge_attr
        # x = self.conv(x, edge_index=edge_index,
        #               edge_attr=edge_weight.float())
        x = self.conv(x, edge_index=edge_index)
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        x = apply_activation(x, self.activation)
        return x


class ArmaNet_bench(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_layers=4, num_stacks=3, final_sigmoid_layer=True):
        super(ArmaNet_bench, self).__init__()
        self.conv1 = ARMAConv(num_node_features, 16, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2 = ARMAConv(16, num_classes, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25, act=None)
        self.conv2_bn = nn.BatchNorm1d(num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.final_sigmoid_layer = final_sigmoid_layer
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x=x, edge_index=edge_index,
                       edge_weight=edge_weight.float())
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index=edge_index,
                       edge_weight=edge_weight.float())
        # x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.endLinear.reset_parameters()


class ArmaNet_ray(torch.nn.Module):
    def __init__(self, num_layers, num_channels, activation, num_internal_layers, num_internal_stacks, batch_norm_index, shared_weights, dropout, final_linear_layer, final_sigmoid_layer=True):
        super(ArmaNet_ray, self).__init__()
        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer

        self.convlist = nn.ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            num_c_out = num_channels[i+1]
            num_s = num_internal_stacks[i]
            num_l = num_internal_layers[i]
            conv = ArmaConvModule(num_channels_in=num_c_in, num_channels_out=num_c_out, activation=activation[i], num_internal_layers=num_l,
                                  num_internal_stacks=num_s, batch_norm=batch_norm_index[i], shared_weights=shared_weights, dropout=dropout)
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = nn.Linear(1, 1)
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class GCNNet_bench(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10):
        super(GCNNet_bench, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2 = GCNConv(16, 4)
        self.conv2_bn = nn.BatchNorm1d(4)
        self.conv3 = GCNConv(4, num_classes)
        self.conv3_bn = nn.BatchNorm1d(num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv1_bn(x.float())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x.float(), edge_index=edge_index,
                       edge_weight=edge_weight)
        x = self.conv2_bn(x.float())
        x = F.relu(x)
        x = self.conv3(x=x.float(), edge_index=edge_index,
                       edge_weight=edge_weight)
        x = self.conv3_bn(x.float())
        x = self.endLinear(x.float())
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.endLinear.reset_parameters()


class GCNNet_ray(torch.nn.Module):
    def __init__(self, num_layers, num_channels, activation, improved, batch_norm_index, final_linear_layer, final_sigmoid_layer=True):
        super(GCNNet_ray, self).__init__()

        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer

        self.convlist = nn.ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            num_c_out = num_channels[i+1]
            conv = GCNConvModule(
                num_channels_in=num_c_in, num_channels_out=num_c_out, activation=activation[i], improved=improved, batch_norm=batch_norm_index[i])
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = nn.Linear(1, 1)
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class SAGENet_ray(torch.nn.Module):
    def __init__(self, num_layers, num_channels, activation, batch_norm_index, final_linear_layer, final_sigmoid_layer=True):
        super(SAGENet_ray, self).__init__()
        self.activation = activation
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer

        self.convlist = nn.ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            num_c_out = num_channels[i+1]
            conv = SAGEConvModule(
                num_channels_in=num_c_in,  num_channels_out=num_c_out, activation=activation[i], batch_norm=batch_norm_index[i])
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = nn.Linear(1, 1)
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class TAGNet_bench(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10):
        super(TAGNet_bench, self).__init__()
        self.conv1 = TAGConv(num_node_features, 4)
        self.conv2 = TAGConv(4, num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.float()

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.endLinear.reset_parameters()


class TAGNet_ray(torch.nn.Module):
    def __init__(self, num_layers, num_channels, activation, K_hops, batch_norm_index, final_linear_layer, final_sigmoid_layer=True):
        super(TAGNet_ray, self).__init__()

        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer

        self.convlist = nn.ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            num_c_out = num_channels[i+1]
            K = K_hops[i]

            conv = TAGConvModule(
                num_channels_in=num_c_in, num_channels_out=num_c_out, activation=activation[i], K=K, batch_norm=batch_norm_index[i])
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = nn.Linear(1, 1)
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class TransformerNet_ray(torch.nn.Module):
    def __init__(self, num_layers, num_channels, activation, batch_norm_index, heads, dropout, final_linear_layer, final_sigmoid_layer=True):
        super(TransformerNet_ray, self).__init__()

        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer

        self.convlist = nn.ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            if i > 0:
                num_c_in = num_c_in * heads
            num_c_out = num_channels[i+1]
            conv = TransformConvModule(
                num_channels_in=num_c_in, num_channels_out=num_c_out, activation=activation[i], heads=heads, dropout=dropout, batch_norm=batch_norm_index[i])
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = nn.Linear(num_c_out*heads, 1)
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class GNNmodule(nn.Module):
    def __init__(self, config, criterion_positive_weight=False):
        super(GNNmodule, self).__init__()
        cuda = config["cuda"]
        if "Fbeta::beta" in config:
            self.beta = config["Fbeta::beta"]
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.cuda = True
            print("cuda availabe:: send model to GPU")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
            print("cuda unavailable:: train model on cpu")
        self.critierion_positive_weight = criterion_positive_weight
        if type(self.critierion_positive_weight) != bool:
            self.critierion_positive_weight = torch.tensor(
                self.critierion_positive_weight).to(self.device)

        # seeds
        torch.manual_seed(config["manual_seed"])
        torch.cuda.manual_seed(config["manual_seed"])
        np.random.seed(config["manual_seed"])
        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # num_channels
        if config["model_name"] != "ArmaNet_bench":
            num_channels = self.make_list_number_of_channels(config)
        task_type = config["task_type"]
        if "final_sigmoid_layer" in config:
            final_sigmoid_layer = config["final_sigmoid_layer"]
        else:
            if task_type == "classification":
                final_sigmoid_layer = False
                print("information to final sigmoid layer missing, setting it to False")
            if task_type == "regression":
                final_sigmoid_layer = True
                print("information to final sigmoid layer missing, setting it to True")
            else:
                final_sigmoid_layer = True
                print("information to final sigmoid layer missing, setting it to True")
        if config["model_name"] == "ArmaNet_bench":
            model = ArmaNet_bench(
                final_sigmoid_layer=final_sigmoid_layer)
        elif config["model_name"] == "ArmaNet_ray":
            num_internal_layers = self.make_list_Arma_internal_layers(config)
            num_internal_stacks = self.make_list_Arma_internal_stacks(config)
            model = ArmaNet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"],
                                num_internal_layers=num_internal_layers, num_internal_stacks=num_internal_stacks, batch_norm_index=config["batch_norm_index"], shared_weights=config["ARMA::shared_weights"], dropout=config["ARMA::dropout"], final_linear_layer=config["final_linear_layer"], final_sigmoid_layer=final_sigmoid_layer)
        elif config["model_name"] == "GCNNet_bench":
            model = GCNNet_bench()
        elif config["model_name"] == "GCNNet_ray":
            model = GCNNet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"], improved=config["GCN::improved"],
                               batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"], final_sigmoid_layer=final_sigmoid_layer)
        elif config["model_name"] == "SAGENet_ray":
            model = SAGENet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"],
                                batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"], final_sigmoid_layer=final_sigmoid_layer)
        elif config["model_name"] == "TAGNet_bench":
            model = TAGNet_bench()
        elif config["model_name"] == "TAGNet_ray":
            K_hops = self.make_list_Tag_hops(config)
            model = TAGNet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"], K_hops=K_hops,
                               batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"], final_sigmoid_layer=final_sigmoid_layer)
        elif config["model_name"] == "TransformerNet_ray":
            model = TransformerNet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"], batch_norm_index=config["batch_norm_index"],
                                       heads=config["Transformer::heads"], dropout=config["Transformer::dropout"], final_linear_layer=config["final_linear_layer"], final_sigmoid_layer=final_sigmoid_layer)
        else:
            print("error: model type unkown")

        # model.double()
        model.to(self.device)

        self.model = model

        # criterion
        if config["criterion"] == "MSELoss":
            if criterion_positive_weight == True:
                self.criterion = nn.MSELoss(reduction="none")
            else:
                self.criterion = nn.MSELoss()
        if config["criterion"] == "BCEWithLogitsLoss":
            if criterion_positive_weight == False:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(criterion_positive_weight))
                print("positive_weigt used for criterion: ",
                      criterion_positive_weight)
        if config["criterion"] == "BCELoss":
            self.criterion = nn.BCELoss()
        self.criterion.to(self.device)

        # set opimizer
        if config["optim::optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config["optim::LR"], momentum=config["optim::momentum"], weight_decay=config["optim::weight_decay"])
        if config["optim::optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(
            ), lr=config["optim::LR"], weight_decay=config["optim::weight_decay"])
        self.optimizer = optimizer

        # scheduler
        scheduler_name = config["optim::scheduler"]
        self.scheduler_name = scheduler_name
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=config["optim::ReducePlat_patience"], factor=config["optim::LR_reduce_factor"])
        elif scheduler_name == "stepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config["optim::stepLR_step_size"], gamma=config["optim::LR_reduce_factor"])
        elif scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=.1, last_epoch=-1)
        elif scheduler_name == "None":
            scheduler = None
        elif scheduler_name == None:
            scheduler = None
        self.scheduler = scheduler

    def forward(self, x):
        # compute model prediction
        y = self.model(x)
        return y

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            # print(fname)
            perf_dict["state_dict"] = self.model.state_dict()
            torch.save(perf_dict, fname)
        return None

    def scheduler_step(self, criterion):
        scheduler_name = self.scheduler_name
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(criterion)
        if scheduler_name == "stepLR":
            self.scheduler.step()
        if scheduler_name == "ExponentialLR":
            self.scheduler.step()

    def train_epoch_regression(self, data_loader, threshold):
        self.model.train()
        loss = 0.
        correct = 0
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        all_predictions = torch.Tensor(0).to(self.device)
        for _, (batch) in enumerate(data_loader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(self.model.forward(batch))
            labels = batch.y
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            correct += torch.sum((torch.abs(output - labels) < threshold))
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_predictions = torch.cat([all_predictions, output])
        r2score = R2Score().to(self.device)
        R2 = r2score(all_predictions, all_labels)
        # accuracy
        accuracy = 100 * correct / all_labels.shape[0]
        self.scheduler_step(loss)
        return loss, accuracy.item(), R2.item()

    def train_epoch_regressionThresholding(self, data_loader):
        if self.critierion_positive_weight == False:
            return self.train_epoch_regressionThresholdingMSE(data_loader)
        else:
            return self.train_epoch_regressionThresholdingWeighted(data_loader)

    def train_epoch_regressionThresholdingWeighted(self, data_loader):
        self.model.train()
        loss = 0.
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for iter, (batch) in enumerate(data_loader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(batch).squeeze()
            labels = batch.y
            temp_loss = self.criterion(output, labels)
            labels_class = torch.where(labels < 15., torch.tensor(0.).to(
                self.device), torch.tensor(1.).to(self.device)).int().to(self.device)
            weights = torch.where(labels_class == torch.tensor(
                1), self.critierion_positive_weight, torch.tensor(1.).to(self.device))
            temp_loss = temp_loss * weights
            temp_loss = temp_loss.mean()
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])

        r2score = R2Score().to(self.device)
        R2 = r2score(all_outputs, all_labels)

        labels_classification = torch.where(all_labels < 15., 0., 1.).int()
        outputs_classification = torch.where(all_outputs < 15., 0., 1.).int()
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(outputs_classification, labels_classification)
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(outputs_classification, labels_classification)
        self.scheduler_step(loss)
        return loss, R2.item(), fbeta.item(), recall.item()

    def train_epoch_regressionThresholdingMSE(self, data_loader):
        self.model.train()
        loss = 0.
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for iter, (batch) in enumerate(data_loader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(batch).squeeze()
            labels = batch.y
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])

        r2score = R2Score().to(self.device)
        R2 = r2score(all_outputs, all_labels)

        labels_classification = torch.where(all_labels < 15., 0., 1.).int()
        outputs_classification = torch.where(all_outputs < 15., 0., 1.).int()
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(outputs_classification, labels_classification)
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(outputs_classification, labels_classification)
        self.scheduler_step(loss)
        return loss, R2.item(), fbeta.item(), recall.item()

    def train_epoch_classification(self, data_loader):
        self.model.train()
        loss = 0.
        correct = 0
        all_labels = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for _, (batch) in enumerate(data_loader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(batch).squeeze()
            labels = batch.y
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])
        sigmoid_layer = torch.nn.Sigmoid().to(self.device)
        all_outputs_bin = sigmoid_layer(all_outputs) > .5
        correct += (all_outputs_bin == all_labels).sum()
        f1 = F1Score(multiclass=False).to(self.device)
        f1 = f1(all_outputs_bin, all_labels.int())
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(all_outputs_bin, all_labels.int())
        accuracy = 100 * correct / all_labels.shape[0]
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(all_outputs_bin, all_labels.int())
        precision = Precision(multiclass=False)
        precision = precision.to(self.device)
        precision = precision(all_outputs_bin, all_labels.int())
        self.scheduler_step(loss)
        return loss, accuracy.item(), f1.item(), fbeta.item(), recall.item(), precision.item()

    def eval_model_regression(self, data_loader, threshold):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            all_predictions = torch.Tensor(0).to(self.device)
            for batch in data_loader:
                batch.to(self.device)
                labels = batch.y
                output = torch.squeeze(self.model(batch))
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                correct += torch.sum((torch.abs(output - labels) < threshold))
                all_predictions = torch.cat([all_predictions, output])
                all_labels = torch.cat([all_labels, labels])
            accuracy = 100 * correct / all_labels.shape[0]
        r2score = R2Score().to(self.device)
        R2 = r2score(all_predictions, all_labels)
        return loss, accuracy.item(), R2.item()

    def eval_model_regressionThresholding(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for batch in data_loader:
                batch.to(self.device)
                labels = batch.y
                output = self.model(batch).squeeze()
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                all_labels = torch.cat([all_labels, labels])
                all_outputs = torch.cat([all_outputs, output])
        r2score = R2Score().to(self.device)
        R2 = r2score(all_outputs, all_labels)

        labels_classification = torch.where(all_labels < 15., 0., 1.).int()
        outputs_classification = torch.where(all_outputs < 15., 0., 1.).int()
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(outputs_classification, labels_classification)
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(outputs_classification, labels_classification)
        return loss, R2.item(), fbeta.item(), recall.item()

    def eval_model_classification(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            all_labels = torch.Tensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for batch in data_loader:
                batch.to(self.device)
                labels = batch.y
                output = self.model(batch).squeeze()
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                all_labels = torch.cat([all_labels, labels])
                all_outputs = torch.cat([all_outputs, output])
        sigmoid_layer = torch.nn.Sigmoid().to(self.device)
        all_outputs_bin = sigmoid_layer(all_outputs) > .5
        correct += (all_outputs_bin == all_labels).sum()
        f1 = F1Score(multiclass=False).to(self.device)
        f1 = f1(all_outputs_bin, all_labels.int())
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(all_outputs_bin, all_labels.int())
        accuracy = 100 * correct / all_labels.shape[0]
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(all_outputs_bin, all_labels.int())
        precision = Precision(multiclass=False)
        precision = precision.to(self.device)
        precision = precision(all_outputs_bin, all_labels.int())
        return loss, accuracy.item(), f1.item(), fbeta.item(), recall.item(), precision.item()

    def aggregate_list_from_config(self, config, key_word, index_start, index_end):
        new_list = [config[key_word+str(index_start)]]
        for i in range(index_start+1, index_end+1):
            index_name = key_word + str(i)
            new_list.append(config[index_name])
        return new_list

    def make_list_number_of_channels(self, config):
        key_word = "num_channels"
        index_start = 1
        index_end = config["num_layers"] + 1
        num_channels = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return num_channels

    def make_list_Tag_hops(self, config):
        key_word = "TAG::K_hops"
        index_start = 1
        index_end = config["num_layers"]
        list_k_hops = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return list_k_hops

    def make_list_Arma_internal_stacks(self, config):
        key_word = "ARMA::num_internal_stacks"
        index_start = 1
        index_end = config["num_layers"]
        list_internal_stacks = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return list_internal_stacks

    def make_list_Arma_internal_layers(self, config):
        key_word = "ARMA::num_internal_layers"
        index_start = 1
        index_end = config["num_layers"]
        list_internal_layers = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return list_internal_layers


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()
