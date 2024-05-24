import os
import torch
import torch.optim as optim
from TRIP import TRIP
from train_utils import training

""" This is the code used to train TRIP on the DVSGesture dataset. """
number_of_experiments = 5
""" TRAINING PARAMETERS """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_batch_size = 32
test_batch_size = 32
epoch = 1000
lr = 1e-4
use_scheduler = False
lr_decay_epochs = 1
lr_gamma = 0.08747
spars_loss = True
L1_wt = 0.1
""" NETWORK PARAMETERS """
input_size = 128
""" MAKE DIRECTORIES """
directory = "TRIP_SPARSE"
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
""" DATASET LOCATION """
dataset_dir ='../datasets/dvsgesture/'

for i in range(number_of_experiments):

    """ SELECT MODEL """
    model = TRIP(input_size=input_size, device=device)

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

    """ START TRAINING """
    training(dataset_dir, model, optimizer, train_batch_size, test_batch_size, device, directory, run_name, save_dir, run_num, epoch=epoch, 
            spars_loss=spars_loss, L1_wt=L1_wt, scheduler=scheduler)