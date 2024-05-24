import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import pad_sequence_collate


def visualize(event_array, RoI_params, labels, batch_to_visualize, inf_ts):

    import matplotlib.pyplot as plt

    gesture_classes = {0 : 'hand clapping',
                       1 : 'right hand wave',
                       2 : 'other gestures',
                       3:  'left hand wave',
                       4 : 'right arm clockwise',
                       5 : 'right arm counter clockwise',
                       6 : 'left arm clockwise',
                       7 : 'left arm counter clockwise',
                       8 : 'arm roll',
                       9 : 'air drums',
                       10 : 'air guitar',
                       }


    event_array = event_array.cpu()

    image_frame_list = []
    print("Gesture class: " + gesture_classes[labels[batch_to_visualize].item()])
    for ii in range(inf_ts):
        print("Timebin " + str(ii) + ":")
        convert_image = np.zeros((event_array.shape[1], event_array.shape[2]))
        convert_image[event_array[0, :, :, ii] == 1] = 0.5
        convert_image[event_array[1, :, :, ii] == 1] = 1.0

        ROI_size = 12
        GK_grid_distance_from_center_to_edge = (ROI_size-1)/2

        gx = np.round(RoI_params[0][ii].item()).astype(int)
        gy = np.round(RoI_params[1][ii].item()).astype(int)
        GK_grid_radius = np.round(RoI_params[2][ii].item()*(GK_grid_distance_from_center_to_edge)).astype(int)

        """ X, Y limits of center points in NxN grid of Gaussian Kernels """
        lim1 = min(128,max(0, gy-GK_grid_radius))
        lim2 = min(128,max(0, gy+GK_grid_radius))
        lim3 = min(128,max(0, gx-GK_grid_radius))
        lim4 = min(128,max(0, gx+GK_grid_radius))

        """ To avoid incorrect visualization when the receptive field goes off screen """
        if lim3 > 0:
            convert_image[lim1:lim2, lim3:lim3+2] = 1
        if lim4 < 128:
            convert_image[lim1:lim2, lim4-2:lim4] = 1
        if lim1 > 0:
            convert_image[lim1:lim1+2, lim3:lim4] = 1
        if lim2 < 128:
            convert_image[lim2-2:lim2, lim3:lim4] = 1

        im = plt.imshow(convert_image, animated=True)
        image_frame_list.append([im])
        plt.show()


def testing(dataset_dir, device, exp_name, run_num, model_name, test_batch_size=32, number_of_timebins_for_inference=0, 
            visualize_ROI=False, batch_to_visualize=0, sample_to_visualize=0, print_MACs=False):
 
    """
    Args:
        dataset_dir (str): testing dataset directory
        device (device): device
        exp_name (str): name of the experiment
        run_num (int): number of experiment and name of folder within session_name to save model in
        model_name (str): name of the model 
        test_batch_size (int): batch size for testing
        number_of_timebins_for_inference (int): number of timebins to process during inference (1-32)
        visualize_ROI (Boolean): enable/disabled visualization of timebins with ROI superimposed
        batch_to_visualize (int): select which batch to visualize sample from
        sample_to_visualize (int): select which sample in batch to visualize
        print_MACs (Boolean): effective MACs computing and printing enabled/disabled
     """
    split_by = 'number'
    number_of_timebins = 32
    test_dataset = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=number_of_timebins, split_by=split_by)
    test_dataloader = DataLoader(dataset=test_dataset, collate_fn=pad_sequence_collate, batch_size=test_batch_size,
        shuffle=False, num_workers=4,
        drop_last=False,
        pin_memory=True)

    test_num = len(test_dataset)    
    test_accuracy_list = []

    session_file_dir = "./save_models/" + exp_name
    network = torch.load(session_file_dir + '/' + str(run_num) + '/' + model_name)
  
    network.eval()
   
    test_correct_num = 0
    with torch.no_grad():
        tot_batch = []
        sample = 0
        for data in test_dataloader:
            event_img, labels, _ = data
            event_img = torch.permute(event_img, (0, 2, 3, 4, 1))
            event_img, labels = event_img.to(device), labels.to(device)
            if print_MACs:
                outputs, frames, RoI, RoI_params, activ_dict = network(event_img, train=0, inf_ts=number_of_timebins_for_inference, spars_loss=True)
                tot_frames = []
                for batch in range(32):
                    for frame_num in range(len(activ_dict)):
                        macs_tot = 0
                        activ_dict0 = activ_dict[frame_num]
                        for key, activ_tensor0 in activ_dict0.items():
                            add_with = 0
                            activ_tensor = activ_tensor0[batch]
                            tot_nonzero_inputs = len(activ_tensor[activ_tensor!=0])
                            if key == 'input':
                                multiply_with = 32*3*3
                            if key == 'findCNN1':
                                multiply_with = 64*3*3
                            if key == 'findCNN2':
                                multiply_with = 128*3*3
                            if key == 'findCNN3':
                                multiply_with = 256
                                add_with = (256*256)
                            if key == 'RNN':
                                multiply_with = 3
                            if key == 'roi':
                                multiply_with = 32*3*3
                            if key == 'glimpseCNN1':
                                multiply_with = 64*3*3
                            if key == 'glimpseCNN2':
                                multiply_with = 256
                            if key == 'clout2':
                                multiply_with = 11
                            macs_of_layer = tot_nonzero_inputs * multiply_with + add_with
                            macs_tot += macs_of_layer
                            tot_frames.append(macs_tot)
                tot_batch.append(np.mean(tot_frames)/1e6)

            else:
                outputs, frames, RoI, RoI_params = network(event_img, train=0, inf_ts=number_of_timebins_for_inference, batch_to_visualize=batch_to_visualize)
            if visualize_ROI:
                if sample == sample_to_visualize:
                    visualize(frames, RoI_params, labels, batch_to_visualize, number_of_timebins_for_inference) 
            _, predicted = torch.max(outputs, 1)
            test_correct_num += ((predicted == labels).sum().to("cpu")).item()
            sample += 1

    test_accuracy_list.append(test_correct_num / test_num)
    print("Test Accuracy %.4f" % (test_accuracy_list[-1]))
    print("End Testing.")
    if print_MACs:
        print("Average Effective MACs/frame: %.2fM" %np.mean(tot_batch))

