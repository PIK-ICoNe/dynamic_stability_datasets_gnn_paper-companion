import os
import numpy as np
import h5py

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from torch_geometric.nn import GCNConv, ARMAConv, SAGEConv, TAGConv
from torch_geometric.nn import Sequential


def get_length_of_dataset(grid_path):
    count = 0
    for file in sorted(os.listdir(grid_path)):
        if file.startswith('grid_data_'):
            if count == 0:
                startIndex = int(os.path.splitext(
                    file)[0].split('grid_data_')[1])
                digits = (os.path.splitext(
                    file)[0].split('grid_data_')[1])
            count += 1
    return count, startIndex, digits


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
    def __init__(self, grid_path, task, slice_index=slice(0, 0)):
        if slice_index.stop == 0:
            self.data_len, self.start_index, digits = get_length_of_dataset(
                grid_path)
        else:
            _, _, digits = get_length_of_dataset(grid_path)
            self.start_index = slice_index.start + 1
            self.data_len = slice_index.stop - slice_index.start + 1

        self.path = grid_path
        self.num_digits = '0' + str(digits.__str__().__len__())
        self.num_classes = 1
        self.task = task
        self.data = {}
        self.read_in_all_data()

    def read_in_all_data(self):
        for i in range(self.start_index, self.start_index + self.data_len):
            self.data[i] = self.__getitem_from_disk__(i)

    def __len__(self):
        return self.data_len

    def num_classes(self):
        return self.num_classes

    def __get_input__(self, index):
        id_index = format(index, self.num_digits)
        file_to_read = str(self.path)+'/grid_data_'+str(id_index) + '.h5'
        hf = h5py.File(file_to_read, 'r')
        # read in sources/sinks
        dataset_P = hf.get('P')
        P = np.array(dataset_P)
        # read in edge_index
        dataset_edge_index = hf.get('edge_index')
        edge_index = np.array(dataset_edge_index)-1
        # read in edge_attr
        dataset_edge_attr = hf.get('edge_attr')
        edge_attr = np.array(dataset_edge_attr)

        hf.close()
        return torch.tensor(P).unsqueeze(0).transpose(0, 1), torch.tensor(edge_index).transpose(1, 0), torch.tensor(edge_attr)

    def __get_P__(self, index):
        P, _, _ = self.__get_input__(index)
        return P

    def __get_edge_index__(self, index):
        _, edge_index, _ = self.__get_input__(index)
        return edge_index

    def __get_edge_attr__(self, index):
        _, _, edge_attr = self.__get_input__(index)
        return edge_attr

    def __get_label__(self, index):
        id_index = format(index, self.num_digits)
        if self.task == "snbs":
            file_to_read = str(self.path)+'/snbs_'+str(id_index) + '.h5'
        elif self.task == "surv":
            file_to_read = str(self.path)+'/surv_'+str(id_index) + '.h5'
        hf = h5py.File(file_to_read, 'r')
        if self.task == "snbs":
            dataset_target = hf.get('snbs')
        elif self.task == "surv":
            dataset_target = hf.get('surv')
        targets = np.array(dataset_target)
        hf.close()
        return torch.tensor(targets).unsqueeze(1)

    def __getitem_from_disk__(self, index):
        x, edge_index, edge_attr = self.__get_input__(index)
        y = self.__get_label__(index)
        data = gData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

    def __getitem__(self, index):
        return self.data[index+self.start_index]


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


class ArmaNet_bench(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_layers=4, num_stacks=3):
        super(ArmaNet_bench, self).__init__()
        self.conv1 = ARMAConv(num_node_features, 16, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2 = ARMAConv(16, num_classes, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25, act=None)
        self.conv2_bn = nn.BatchNorm1d(num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
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
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.endLinear.reset_parameters()


class ArmaNet_ray(torch.nn.Module):
    def __init__(self, num_layers, num_channels, activation, num_internal_layers, num_internal_stacks, batch_norm_index, shared_weights, dropout, final_linear_layer):
        super(ArmaNet_ray, self).__init__()
        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer

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
        self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
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
    def __init__(self, num_layers, num_channels, activation, improved, batch_norm_index, final_linear_layer):
        super(GCNNet_ray, self).__init__()

        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer

        self.convlist = nn.ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            num_c_out = num_channels[i+1]
            conv = GCNConvModule(
                num_channels_in=num_c_in, num_channels_out=num_c_out, activation=activation[i], improved=improved, batch_norm=batch_norm_index[i])
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = nn.Linear(1, 1)
        self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x


class SAGENet_ray(torch.nn.Module):
    def __init__(self, num_layers, num_channels, activation, batch_norm_index, final_linear_layer):
        super(SAGENet_ray, self).__init__()
        self.activation = activation
        self.final_linear_layer = final_linear_layer

        self.convlist = nn.ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            num_c_out = num_channels[i+1]
            conv = SAGEConvModule(
                num_channels_in=num_c_in,  num_channels_out=num_c_out, activation=activation[i], batch_norm=batch_norm_index[i])
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = nn.Linear(1, 1)
        self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
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
    def __init__(self, num_layers, num_channels, activation, K_hops, batch_norm_index, final_linear_layer):
        super(TAGNet_ray, self).__init__()

        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer

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
        self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x


class GNNmodule(nn.Module):
    def __init__(self, config):
        super(GNNmodule, self).__init__()
        cuda = config["cuda"]
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.cuda = True
            print("cuda availabe:: send model to GPU")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
            print("cuda unavailable:: train model on cpu")

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

        if config["model_name"] == "ArmaNet_bench":
            model = ArmaNet_bench()
        elif config["model_name"] == "ArmaNet_ray":
            num_internal_layers = self.make_list_Arma_internal_layers(config)
            num_internal_stacks = self.make_list_Arma_internal_stacks(config)
            model = ArmaNet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"],
                                num_internal_layers=num_internal_layers, num_internal_stacks=num_internal_stacks, batch_norm_index=config["batch_norm_index"], shared_weights=config["ARMA::shared_weights"], dropout=config["ARMA::dropout"], final_linear_layer=config["final_linear_layer"])
        elif config["model_name"] == "GCNNet_bench":
            model = GCNNet_bench()
        elif config["model_name"] == "GCNNet_ray":
            model = GCNNet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"], improved=config["GCN::improved"],
                               batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"])
        elif config["model_name"] == "SAGENet_ray":
            model = SAGENet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"],
                                batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"])
        elif config["model_name"] == "TAGNet_bench":
            model = TAGNet_bench()
        elif config["model_name"] == "TAGNet_ray":
            K_hops = self.make_list_Tag_hops(config)
            model = TAGNet_ray(num_layers=config["num_layers"], num_channels=num_channels, activation=config["activation"], K_hops=K_hops,
                               batch_norm_index=config["batch_norm_index"], final_linear_layer=config["final_linear_layer"])
        else:
            print("error: model type unkown")

        model.double()
        model.to(self.device)

        self.model = model

        # criterion
        if config["criterion"] == "MSELoss":
            self.criterion = nn.MSELoss()
        self.criterion.to(self.device)

        # set opimizer
        if config["optim::optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config["optim::LR"], momentum=config["optim::momentum"], weight_decay=config["optim::weight_decay"])
        if config["optim::optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(
            ), lr=config["optim::LR"], weight_decay=config["optim::weight_decay"])
        self.optimizer = optimizer
        # self.optimizer.to(self.device)

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

        # self.epoch = 0
        # self.best_epoch = 0
        # self.loss_best = 1e15

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

    def train_epoch(self, data_loader, threshold):
        self.model.train()
        # scheduler = self.scheduler
        N = data_loader.dataset[0].x.shape[0]
        loss = 0.
        correct = 0
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        for iter, (batch) in enumerate(data_loader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(batch)
            labels = batch.y
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            correct += self.get_prediction(output, labels, threshold)
            loss += temp_loss.item()
            # R2
            mse_trained += torch.sum((output - labels) ** 2)
            all_labels = torch.cat([all_labels, labels])
        # self.epoch += 1
        # R2
        mean_labels = torch.mean(all_labels)
        array_ones = torch.ones(all_labels.shape[0], 1)
        array_ones = array_ones.to(self.device)
        output_mean = mean_labels * array_ones
        mse_mean = torch.sum((output_mean-all_labels)**2)
        R2 = (1 - mse_trained/mse_mean).item()
        # R2 = 1 - mse_trained/mse_mean
        # accuracy
        accuracy = 100 * correct / all_labels.shape[0]
        self.scheduler_step(loss)
        return loss, accuracy, R2

    def eval_model(self, data_loader, threshold):
        self.model.eval()
        with torch.no_grad():
            N = data_loader.dataset[0].x.shape[0]
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            for batch in data_loader:
                batch.to(self.device)
                labels = batch.y
                output = self.model(batch)
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                correct += self.get_prediction(output, labels, threshold)
                # R2
                mse_trained += torch.sum((output - labels) ** 2)
                all_labels = torch.cat([all_labels, labels])
            accuracy = 100 * correct / all_labels.shape[0]
        # R2
        mean_labels = torch.mean(all_labels)
        array_ones = torch.ones(all_labels.shape[0], 1)
        array_ones = array_ones.to(self.device)
        output_mean = mean_labels * array_ones
        mse_mean = torch.sum((output_mean-all_labels)**2)
        R2 = (1 - mse_trained/mse_mean).item()
        return loss, accuracy, R2

    def get_prediction(self, output, label, tol):
        count = 0
        output = output.view(-1, 1).view(-1)
        label = label.view(-1, 1).view(-1)
        batchSize = output.size(-1)
        for i in range(batchSize):
            if ((abs(output[i] - label[i]) < tol).item() == True):
                count += 1
        return count

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
