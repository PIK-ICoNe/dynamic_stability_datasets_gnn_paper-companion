import wget
import os
import zipfile

from pathlib import Path

from gnn_models import GNNmodule, gnn_snbs_surv
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

##########
# download and unzip datasets
def automatically_download_datasets(path_for_datasets):
    zenodo_url = 'https://zenodo.org/record/6572973/files/'
    dataset020_url = zenodo_url + 'dataset020.zip'
    dataset100_url = zenodo_url + 'dataset100.zip'
    datasettexas_url = zenodo_url + 'dataset_texas.zip'

    if os.path.isdir(path_for_datasets)==False:
        os.mkdir(path_for_datasets)

    dataset020_zip = path_for_datasets + '/dataset020.zip'
    dataset100_zip = path_for_datasets + '/dataset100.zip'
    if os.path.isfile(dataset020_zip) == False:
        wget.download(dataset020_url,"downloaded_datasets")
        
    if os.path.isfile(dataset100_zip) == False:
        wget.download(dataset100_url,"downloaded_datasets")


def unzip_datasets(path_for_datasets):
    dataset020_zip = path_for_datasets + '/dataset020.zip'
    dataset100_zip = path_for_datasets + '/dataset100.zip'
    with zipfile.ZipFile(dataset020_zip, 'r') as zip_ref:
        zip_ref.extractall(path_for_datasets)
    with zipfile.ZipFile(dataset100_zip, 'r') as zip_ref:
        zip_ref.extractall(path_for_datasets)



# The datasets will be downloaded and stored in path_for_datasets, change if desired.
path_for_datasets = 'downloaded_datasets'
automatically_download_datasets(path_for_datasets) ## in case of manual download, put the zip files in path_for_dataset and remove this line

unzip_datasets(path_for_datasets)
########


result_path= Path("training_run_directory")
if os.path.isdir(result_path)==False:
    os.mkdir(result_path)




# config for training
cfg = {}
# dataset
cfg["dataset::path"] = Path(path_for_datasets + '/dataset020')
cfg["task"] = "snbs"
# cfg["train_set::start_index"] = 0
# cfg["train_set::end_index"] = 799

# dataset batch sizes
cfg["train_set::batchsize"] = 228
# cfg["train_set::batchsize"] = tune.choice([150, 1000])
# cfg["train_set::batchsize"] = tune.randint(150,1000)
cfg["test_set::batchsize"] = 500
cfg["valid_set::batchsize"] = 500
cfg["train_set::shuffle"] = True
cfg["test_set::shuffle"] = False
cfg["valid_set::shuffle"] = False


# ray settings
cfg["save_after_epochs"] = 100
cfg["checkpoint_freq"] = 100
cfg["num_samples"] = 5
cfg["ray_name"] = "ArmaNet3l"

# model settings
cfg["model_name"] = "ArmaNet_ray"
cfg["final_linear_layer"] = False
cfg["num_layers"] = 3
# cfg["max_num_channels"] = 157 
cfg["num_channels1"] = 1  #tune.randint(1, cfg["max_num_channels"])
cfg["num_channels2"] = 35  #tune.randint(1, cfg["max_num_channels"])
cfg["num_channels3"] = 96 #tune.randint(1, cfg["max_num_channels"])
cfg["num_channels4"] = 1  #tune.randint(1, cfg["max_num_channels"]

cfg["batch_norm_index"] = [True, True, True]
cfg["activation"] = ["relu","relu","None"]

# ARMA
cfg["ARMA::num_internal_layers"] = [7, 9, 1]
cfg["ARMA::num_internal_stacks"] = [51, 7, 45]

cfg["ARMA::max_num_internal_layers"] = 4
cfg["ARMA::num_internal_layers1"] = 7 #tune.randint(1,cfg["ARMA::max_num_internal_layers"])
cfg["ARMA::num_internal_layers2"] = 9 #tune.randint(1,cfg["ARMA::max_num_internal_layers"])
cfg["ARMA::num_internal_layers3"] = 1
## cfg["ARMA::num_internal_layers4"] = 4
## cfg["ARMA::num_internal_layers5"] = 4
## cfg["ARMA::num_internal_layers6"] = 4


#cfg["ARMA::max_num_internal_stacks"] = 3
cfg["ARMA::num_internal_stacks1"] = 51 # tune.randint(1,cfg["ARMA::max_num_internal_stacks"])
cfg["ARMA::num_internal_stacks2"] = 7 # tune.randint(1,cfg["ARMA::max_num_internal_stacks"])
cfg["ARMA::num_internal_stacks3"] = 45


# cfg["ARMA::num_internal_layers"] = [tune.randint(1,10), tune.randint(1,10)]
# cfg["ARMA::num_internal_stacks"] = [tune.randint(1,100), tune.randint(1,100)]
cfg["ARMA::dropout"] = .25
cfg["ARMA::shared_weights"] = True
## GCN
#cfg["GCN::improved"] = True
#
## TAG
#cfg["TAG::K_hops"] = [tune.randint(1,12), tune.randint(1,12), tune.randint(1,12)]

# training settings
cfg["cuda"] = True
#cfg["num_workers"] = 1
#cfg["num_threads"] = 2
# cfg["manual_seed"] = 1
# cfg["manual_seed"] = [tune.choice([1,2,3,4,5])]
cfg["manual_seed"] = 4
# cfg["epochs"] = 1500
cfg["epochs"] = 400
cfg["optim::optimizer"] = "SGD"
cfg["optim::LR"] = 3.0
# cfg["optim::LR"] = tune.loguniform(1e-4, 2e1)
# cfg["optim::LR"] = tune.choice([1e-4, 1e-2])
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1e-9
cfg["optim::scheduler"] = None
# cfg["optim::scheduler"] = tune.choice(["None", "stepLR", "ReduceLROnPlateau"])
cfg["optim::ReducePlat_patience"] = 20
cfg["optim::LR_reduce_factor"] = .7
cfg["optim::stepLR_step_size"] = 30
# cfg["optim::scheduler"] = "stepLR"
cfg["criterion"] = "MSELoss"
cfg["search_alg"] = "Optuna"
# cfg["search_alg"] = None

# evaluation
cfg["eval::threshold"] = .1

# initialize model
gnnmodule = GNNmodule(cfg)


# init datasets and dataloader
train_set = gnn_snbs_surv(cfg["dataset::path"] / 'train',cfg["task"])
valid_set = gnn_snbs_surv(cfg["dataset::path"] / 'valid', cfg["task"])
test_set = gnn_snbs_surv(cfg["dataset::path"] / 'test', cfg["task"])
train_loader = DataLoader(
    train_set, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"])
valid_loader = DataLoader(
    train_set, batch_size=cfg["valid_set::batchsize"], shuffle=cfg["valid_set::shuffle"])
test_loader = DataLoader(
    test_set, batch_size=cfg["test_set::batchsize"], shuffle=cfg["test_set::shuffle"])

train_loss_all_epochs = []
train_accu_all_epochs = []
train_R2_all_epochs = []

test_loss_all_epochs = []
test_accu_all_epochs = []
test_R2_all_epochs = []

epochs = cfg["epochs"]
for epoch in range(1,epochs):
    print(f"Epoch {epoch}/{epochs}.. ")
    train_loss, train_accu, train_R2 = gnnmodule.train_epoch(train_loader, cfg["eval::threshold"])
    train_loss_all_epochs.append(train_loss)
    train_accu_all_epochs.append(train_accu)
    train_R2_all_epochs.append(train_R2)
    test_loss, test_accu, test_R2 = gnnmodule.eval_model(test_loader, cfg["eval::threshold"])
    test_loss_all_epochs.append(test_loss)
    test_accu_all_epochs.append(test_accu)
    test_R2_all_epochs.append(test_R2)
    print('train R2: ''{:3.2f}'.format(100 * train_R2) + '%')
    print('train accu: ''{:3.2f}'.format(train_accu) + '%')
    print('test R2: ''{:3.2f}'.format(100 * test_R2) + '%')
    print('test accu: ''{:3.2f}'.format(test_accu) + '%')
print("finished")
