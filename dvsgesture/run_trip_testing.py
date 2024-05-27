import torch
from test_utils import testing 

""" Select the experiment to load the model from. """
exp_name = 'TRIP_NO_SPARSE'
experiment_number = 5

""" Enable printing effective MACs. """
print_MACs = True

test_batch_size = 32

dataset_dir ='../datasets/dvsgesture/'

""" Select number of timebins to use in inference """
number_of_timebins_for_inference = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


run_name = exp_name + "_" + str(experiment_number) + "_max_acc"
print("Loading " + str(run_name) + ".")

testing(dataset_dir, device, exp_name, experiment_number, run_name, test_batch_size=test_batch_size, 
        number_of_timebins_for_inference=number_of_timebins_for_inference, print_MACs=print_MACs)
