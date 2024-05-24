import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import pad_sequence_collate
from torchvision.transforms import RandomPerspective, RandomResizedCrop, RandomRotation
import tonic
import torchvision
from tqdm import tqdm

def training(dataset_dir, network, optimizer, batch_size, test_batch_size, device, 
             session_name, model_name, save_dir, run_num, epoch, spars_loss=False, L1_wt=0.1, scheduler=None, DAP=False):
    
    """
    Args:
        network (Torch nn module): Torch neural network model
        optimizer (Torch Function): Optimizer
        batch_size (int): batch size for training
        test_batch_size (int): batch size for testing
        device (device): device
        session_name (str): name of the training session and directory to save model to
        model_name (str): name of the model 
        save_dir (str): path to directory for saving models
        run_num (int): number of experiment and name of folder within session_name to save model in
        epoch (int): number of epochs
        spars_loss (Boolean): sparsity-aware training enabled/disabled 
        L1_wt (float): sparsity-aware training L1 loss weight value
        scheduler (Boolean): learning rate scheduler enabled/disabled
        DAP (Boolean): Dynamic average pooling enabled/disabled
    """
    aug_transform = tonic.transforms.Compose([torch.tensor,
        RandomResizedCrop(
                tonic.datasets.DVSGesture.sensor_size[:-1],
                scale=(0.6, 1.0),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        RandomPerspective(),
        RandomRotation(25)])

    # Train and validation dataloader
    split_by = 'number'
    number_of_timebins = 32
    train_dataset = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=number_of_timebins, split_by=split_by, transform=aug_transform)
    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=pad_sequence_collate, batch_size=batch_size,
        shuffle=True, num_workers=4,
        drop_last=True,
        pin_memory=True)
    test_dataset = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=number_of_timebins, split_by=split_by)
    test_dataloader = DataLoader(dataset=test_dataset, collate_fn=pad_sequence_collate, batch_size=test_batch_size,
        shuffle=False, num_workers=4,
        drop_last=False,
        pin_memory=True)
   
    test_num = len(test_dataset)

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    # List for save accuracy
    train_loss_list, test_accuracy_list = [], []
    max_test_accuracy = 0

    # Define tensorboard
    tf_writer = SummaryWriter(comment='_' + session_name)

    # Start training
    network.to(device)

    """ Print number of network parameters. """
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # if not DAP:
    #     print("Number of network parameters: " + str(count_parameters(network)))
    
    for ee in tqdm(range(epoch)):
        running_loss = 0.0
        running_batch_num = 0
        for data in train_dataloader:
            event_img, labels, _ = data
            event_img = torch.permute(event_img, (0, 2, 3, 4, 1))
            event_img, labels = event_img.to(device), labels.to(device)
            optimizer.zero_grad()

            if spars_loss:
                outputs, activ_dict = network(event_img, train=1, spars_loss=True, DAP=DAP)
                activ_loss = 0
                activ_layer_wt = 1.0
                activ_reg_wt = L1_wt
                for key, activ_tensor in activ_dict.items():
                   activ_loss += activ_layer_wt * torch.mean(torch.abs(activ_tensor))
                loss = criterion(outputs, labels) + (activ_loss*activ_reg_wt)
            else:
                outputs = network(event_img, train=1, DAP=DAP) 
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_batch_num += 1

        if scheduler != None:
            scheduler.step()

        train_loss_list.append(running_loss / running_batch_num)
        tf_writer.add_scalar('dvsgesture_exp/train_loss', train_loss_list[-1], ee)

        test_correct_num = 0
        network.eval()
        with torch.no_grad():
            for data in test_dataloader:
                event_img, labels, _ = data
                event_img = torch.permute(event_img, (0, 2, 3, 4, 1))  
                event_img, labels = event_img.to(device), labels.to(device)
                outputs = network(event_img, train=1, DAP=DAP) 
                _, predicted = torch.max(outputs, 1)
                test_correct_num += ((predicted == labels).sum().to("cpu")).item()
        test_accuracy_list.append(test_correct_num / test_num)
        tf_writer.add_scalar('dvsgesture_exp/test_accuracy', test_accuracy_list[-1])
        network.train()

        if test_accuracy_list[-1] > max_test_accuracy:
            max_test_accuracy = test_accuracy_list[-1]
            torch.save(network, save_dir + '/' + model_name + "_" + str(run_num) + '_max_acc')
    
    print("Test Accuracy %.4f" % (max_test_accuracy))

    print("End Training")