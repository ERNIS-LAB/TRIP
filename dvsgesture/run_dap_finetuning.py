import os
import torch
import torch.optim as optim
import pickle
from train_utils import training

""" This is the code used to fine-tune a pre-trained model with dynamic average pooling ROI generation. """

""" Select the experiment to load the model from. """
exp_name = 'TRIP_NO_SPARSE'
experiment_number = 5

""" TRAINING PARAMETERS """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_batch_size = 32
test_batch_size = 32
epoch = 1000
lr = 1e-4
use_scheduler = False
lr_decay_epochs = 1
lr_gamma = 0.08747
spars_loss = False
L1_wt = 0.1

""" NETWORK PARAMETERS """
input_size = 128

""" LOAD MODEL """
session_file_dir = "./save_models/" + exp_name
load_model_from = session_file_dir + '/' + str(experiment_number) + "/" + exp_name + "_" + str(experiment_number) + "_max_acc"
model = torch.load(load_model_from)
directory = "TRIP_DAP_NO_SPARSE"

try:
    os.mkdir( "./save_models/")
except FileExistsError:
    pass
try:
    save_dir = "./save_models/" + directory
    os.mkdir(save_dir)
    print("Created", save_dir, "directory for saving trained models.")
except FileExistsError:
    pass

run_num = 1
loop = True
while(loop):
    try:
        save_dir = "./save_models/" + directory + '/' +  str(run_num) 
        os.mkdir(save_dir)
        print("Directory " + save_dir + " created, the results of this training experiment will be saved here.")
        loop = False
    except FileExistsError:
        run_num +=1

run_name = directory 

optimizer = optim.Adam(model.parameters(), lr=lr)
if use_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_gamma)
else:
    scheduler = None

""" Fix ROI Prediction network """
no_frozen = 0
for name, param in model.named_parameters():
    if 'findCNN' in name or 'RNN' in name or 'RoI' in name:
        param.requires_grad = False
        no_frozen += 1
assert(no_frozen == 18)

""" DATASET LOCATION """
dataset_dir ='../datasets/dvsgesture/'

training(dataset_dir, model, optimizer, train_batch_size, test_batch_size, device, directory, run_name, save_dir, run_num, epoch=epoch, 
        spars_loss=spars_loss, L1_wt=L1_wt, scheduler=scheduler, DAP=True)