import math
import torch
import torch.nn as nn

class gaussianKernel():
    @staticmethod
    def get_matrices(gx, gy, sigma2, delta, N, A, B, device=0, trunc=0, eps=1e-8):
            if trunc == 0:
                grid_i = torch.reshape(torch.arange(0,N).type(torch.float32), [1, -1]).to(device)
                mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
                mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
                a = torch.reshape(torch.arange(0,A).type(torch.float32), [1, 1, -1]).to(device)
                b = torch.reshape(torch.arange(0,B).type(torch.float32), [1, 1, -1]).to(device)
                mu_x = torch.reshape(mu_x, [-1, N, 1]).to(device)
                mu_y = torch.reshape(mu_y, [-1, N, 1]).to(device)
                sigma2 = torch.reshape(sigma2, [-1, 1, 1]).to(device)
                Fx = torch.exp(-torch.square(a - mu_x) / (2*sigma2))
                Fy = torch.exp(-torch.square(b - mu_y) / (2*sigma2)) # batch x N x B
                if trunc == 0:
                    Fx=Fx/torch.maximum(torch.sum(Fx,2,keepdim=True),torch.tensor(eps))  # normalize, sum over A and B dims
                    Fy=Fy/torch.maximum(torch.sum(Fy,2,keepdim=True),torch.tensor(eps))
                    return Fx,Fy
                elif trunc == 1:
                    nnp = 32
                    Fx_c = torch.zeros((Fx.size(0), Fx.size(1), nnp)).to(device) #these are the "cropped" gaussian filters with dim2
                    Fy_c = torch.zeros((Fy.size(0), Fy.size(1), nnp)).to(device) # equal to nnp instead of 128
                    for batch_dim in range(Fx.size(0)):                         #iterate manually over batches 
                        for row in range(Fx.size(1)):
                            center_x = int(mu_x[batch_dim][row].item())
                            center_x = max(0, (min(127, center_x)))
                            if center_x < nnp:
                                cropped_x_row = Fx[batch_dim][row][0:nnp]
                            elif center_x > 127-nnp:
                                cropped_x_row = Fx[batch_dim][row][127-nnp:127]
                            else:
                                cropped_x_row = Fx[batch_dim][row][int(center_x-(nnp/2)): int(center_x+(nnp/2))]
                            Fx_c[batch_dim][row] = cropped_x_row
                        for row in range(Fy.size(1)):
                            center_y = int(mu_y[batch_dim][row].item())
                            center_y = max(0, (min(127, center_y)))
                            if center_y < nnp:
                                cropped_y_row = Fy[batch_dim][row][0:nnp]
                            elif center_y > 127-nnp:
                                cropped_y_row = Fy[batch_dim][row][127-nnp:127]
                            else:
                                cropped_y_row = Fy[batch_dim][row][int(center_y-(nnp/2)): int(center_y+(nnp/2))]
                            Fy_c[batch_dim][row] = cropped_y_row

                    return Fx_c, Fy_c, mu_x, mu_y, nnp

                elif trunc == 2:
                    nnp = 9

                    Fx=Fx/torch.maximum(torch.sum(Fx,2,keepdim=True),torch.tensor(eps))  # normalize, sum over A and B dims
                    Fy=Fy/torch.maximum(torch.sum(Fy,2,keepdim=True),torch.tensor(eps))

                    Fx_c = torch.zeros((Fx.size(0), Fx.size(1), A)).to(device) #these are the "cropped" gaussian filters with dim2
                    Fy_c = torch.zeros((Fy.size(0), Fy.size(1), B)).to(device) # equal to nnp instead of 128

                    for batch_dim in range(Fx.size(0)): #iterate manually over batches  
                        for row in range(Fx.size(1)):
                            center_x = int(mu_x[batch_dim][row].item())
                            center_x = max(0, (min(127, center_x)))
                            if center_x < nnp:
                                x_interval = [i for i in range (0, nnp)]
                            elif center_x > 127-nnp:
                                x_interval = [i for i in range (127-nnp, 127)]
                            else:
                                x_interval = [i for i in range (int(center_x-(nnp/2)), int(center_x+(nnp/2)))]
                            cropped_x_row = Fx[batch_dim][row][x_interval]
                            Fx_c[batch_dim][row][x_interval] = cropped_x_row
                        for row in range(Fy.size(1)):
                            center_y = int(mu_y[batch_dim][row].item())
                            center_y = max(0, (min(127, center_y)))
                            if center_y < nnp:
                                y_interval = [i for i in range (0, nnp)]
                            elif center_y > 127-nnp:
                                y_interval = [i for i in range (127-nnp, 127)]
                            else:
                                y_interval = [i for i in range (int(center_y-(nnp/2)), int(center_y+(nnp/2)))]
                            cropped_y_row = Fy[batch_dim][row][y_interval]
                            Fy_c[batch_dim][row][y_interval] = cropped_y_row
                    assert(len(Fy_c[Fy_c != 0]) <= (nnp*12*4))
                    assert(len(Fx_c[Fx_c != 0]) <= (nnp*12*4))

                    return Fx_c, Fy_c, mu_x, mu_y, nnp 
    
    @staticmethod
    def filter_matrices(vector, N, batch_size, A, B, device=0, trunc=0):
        gx_,gy_, log_delta = torch.chunk(vector, 3, 1)
        delta=4*(log_delta + 1)
        gx=A/2*(gx_+1)
        gy=B/2*(gy_+1)
       
        sigma2 = torch.zeros((batch_size, 1)).to(device)
        sigma2 = torch.add(sigma2, 2)
        gamma = torch.zeros((batch_size, 1)).to(device)
        gamma = torch.add(gamma, 4)
        return gaussianKernel.get_matrices(gx,gy,sigma2,delta,N, A,B, device, trunc)+(gamma,)+(delta, gx, gy, sigma2)

    @staticmethod
    def ROI_generation(img,Fx,Fy, gamma,device=0, snn=0, truncate=0, mu_x=0, mu_y=0, theta=0):
        if truncate == 0:
                Fxt=torch.permute(Fx, (0,2,1)).to(device)
                im1 = img[:, 0, :, :] #seperate two channels into two images and do filter MM twice
                im2 = img[:, 1, :, :]
                Fy = Fy.to(device)
                gamma = gamma.to(device)
                glimpse1=torch.bmm(Fy,torch.bmm(im1,Fxt))
                glimpse2=torch.bmm(Fy,torch.bmm(im2,Fxt))
                glimpse = torch.stack((glimpse1, glimpse2), 1) #recombine into one tensor
                return glimpse*torch.reshape(gamma,([-1,1,1,1]))
        elif truncate == 1:
                Fxt=torch.permute(Fx, (0,2,1)).to(device)

                im1 = img[:, 0, :, :] #seperate two channels into two images and do filter MM twice
                im2 = img[:, 1, :, :]

                im1_c = torch.zeros([img.size(0), Fx.size(1), Fy.size(1)]).to(device)
                im2_c = torch.zeros([img.size(0), Fx.size(1), Fy.size(1)]).to(device)

                for batch_dim in range(img.size(0)):
                 for row2 in range(Fy.size(1)):
                    center_y = int(mu_y[batch_dim][row2].item())
                    center_y = max(0, (min(127, center_y)))
                    if center_y < theta:
                        y_interval = [i for i in range (0,theta)]
                    elif center_y > 127-theta:
                        y_interval = [i for i in range (127-theta, 127)]
                    else:
                        y_interval = [i for i in range (int(center_y-(theta/2)), int(center_y+(theta/2)))]

                    for row in range(Fx.size(1)):
                        center_x = int(mu_x[batch_dim][row].item())
                        center_x = max(0, (min(127, center_x)))
                        if center_x < theta:
                            x_interval = [i for i in range (0,theta)]
                        elif center_x > 127-theta:
                            x_interval = [i for i in range (127-theta, 127)]
                        else:
                            x_interval = [i for i in range (int(center_x-(theta/2)), int(center_x+(theta/2)))]

                        img_region1 = im1[batch_dim][x_interval]
                        img_region2 = im2[batch_dim][x_interval]
                        
                        img_region1 = img_region1[:, y_interval]
                        img_region2 = img_region2[:, y_interval]

                        x_1 =  Fxt[batch_dim, :, :]
                        y_1 =  Fy[batch_dim, :, :]

                        res1 = torch.matmul(img_region1, x_1) 
                        res1 = torch.matmul(y_1, res1)
                        res2 = torch.matmul(img_region2, x_1)
                        res2 = torch.matmul(y_1, res2)

                        im1_c[batch_dim][row][row2] = res1[row][row2]
                        im2_c[batch_dim][row][row2] = res2[row][row2]                       

                Fy = Fy.to(device)
                gamma = gamma.to(device)
                glimpse = torch.stack((im1_c, im2_c), 1) #recombine into one tensor
                return glimpse*torch.reshape(gamma,([-1,1,1,1]))
        

class TRIP(nn.Module):
    def __init__(self,  input_size, device=0):
        super(TRIP, self).__init__()
        self.batch_size = 32 #just initializing values
        self.gex = 0
        self.gey = 0
        self.delt = 0
        self.sig = 0
        self.gam = 0

        self.eps = 1e-8 # epsilon for numerical stability
     
        self.A, self.B = input_size, input_size # image width, height

        self.device = device

        self.hidden_size = 256 

        self.RoI_size = 12
        self.lin_size = 64*3*3
     
        self.dsMaxpool = nn.MaxPool2d(8)
        self.findCNN = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.rnn_size = 128*2*2
        self.RNN = nn.RNN(self.rnn_size, self.hidden_size, nonlinearity='relu')
    
        self.RoIpred = nn.Sequential(
            nn.Linear(self.hidden_size, 3),                  
        )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.glimpseCNN = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 

        self.classfication = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.lin_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 11)  
        )
    
                   
    def TRIPforward(self, event_data, train, inf_ts, batch_to_visualize, spars_loss, DAP):
        """forward function

        Args:
            event_data (Tensor): event data input (batch_size, channel, x, y, spike_ts)
            test (int): if testing, return attn_window params for visualization
        """

        output_list = []
        self.batch_size = event_data.size(0)
        hidden = torch.zeros(1, event_data.size(0), self.hidden_size).to(self.device)
        glimpse_out = torch.zeros(event_data.size(0), self.lin_size).to(self.device)
   
        all_activs = []

        if inf_ts == 0:
            inference_timebins = event_data.size(-1)
        else:
            inference_timebins=inf_ts

        for tt in range(inference_timebins):
            input_event = event_data[:, :, :, :, tt]  
            downsampled_img = self.dsMaxpool(input_event)

            input_activ = downsampled_img.clone()

            if spars_loss: 
                findCNN1_activ = self.findCNN[0](downsampled_img) 
                findCNN1_activ = self.findCNN[1](findCNN1_activ) 
                findCNN1_activ = self.findCNN[2](findCNN1_activ) 
                findCNN1_activ = self.findCNN[3](findCNN1_activ) 
                findCNN1_activ = findCNN1_activ.clone()

                findCNN2_activ = self.findCNN[4](findCNN1_activ) 
                findCNN2_activ = self.findCNN[5](findCNN2_activ) 
                findCNN2_activ = self.findCNN[6](findCNN2_activ) 
                findCNN2_activ = self.findCNN[7](findCNN2_activ) 
                findCNN2_activ = findCNN2_activ.clone()

                findCNN3_activ = self.findCNN[8](findCNN2_activ) 
                findCNN3_activ = self.findCNN[9](findCNN3_activ) 
                findCNN3_activ = self.findCNN[10](findCNN3_activ) 
                findCNN3_activ = self.findCNN[11](findCNN3_activ) 
                findCNN3_activ = findCNN3_activ.clone()
                find_out = findCNN3_activ
            else:
                find_out = self.findCNN(downsampled_img) 
 
            find_out = find_out.view(1, find_out.size(0), self.rnn_size)

            rnn_out, hidden = self.RNN(find_out, hidden)

            xyd_out = self.RoIpred(rnn_out[0]) 
            if spars_loss:
                RNN_activ = rnn_out[0].clone()
            x, y, d = torch.chunk(xyd_out,3,1)
  
            x = self.tanh(x)
            y = self.tanh(y)
            d = self.sigmoid(d)
      
            xyd_out = torch.cat((x,y), dim=1)
            xyd_out = torch.cat((xyd_out,d), dim=1)


            if DAP:
                # Perform Dynamic Average Pooling
                RoI = torch.zeros((input_event.size(0), input_event.size(1), self.RoI_size, self.RoI_size)).to(self.device)
                gx__,gy__, log_delta_ = torch.chunk(xyd_out, 3, 1)
                for batch_dim in range(input_event.size(0)):

                    #1. Decode ROI parameters (eq. 1, 2, 3 in paper)
                    gx_ = gx__[0]
                    gy_ = gy__[0] 
                    log_delta = log_delta_[0]
                    delta=4*(log_delta + 1)
                    gx=self.A/2*(gx_+1)
                    gy=self.B/2*(gy_+1)
                    N = self.RoI_size

                    #2. Compute xmin, xmax, ymin, ymax (eq. 7, 8 in paper)
                    lower_x_bound = int((gx + (0 - N / 2 - 0.5) * delta) - 5)
                    upper_x_bound = int((gx + (11 - N / 2 - 0.5) * delta) + 5)
                    lower_y_bound = int((gy + (0 - N / 2 - 0.5) * delta) - 5)
                    upper_y_bound = int((gy + (11 - N / 2 - 0.5) * delta) + 5)

                    # Limit boundaries to (0, 128)
                    lower_x_bound = max(0, (min(lower_x_bound, 128)))
                    lower_y_bound = max(0, (min(lower_y_bound, 128)))
                    upper_x_bound = max(0, (min(upper_x_bound, 128)))
                    upper_y_bound = max(0, (min(upper_y_bound, 128)))

                    x_ran = (upper_x_bound - lower_x_bound)
                    y_ran = (upper_y_bound - lower_y_bound)

                    # Sometimes integer rounding results in x and y bounds being different by one pixel,
                    # this code corrects that so both x and y ranges of receptive field are the same size.
                    while x_ran != y_ran:
                        if upper_y_bound == 128:
                            lower_x_bound += math.floor((x_ran-y_ran)/2)
                            upper_x_bound -= math.ceil((x_ran-y_ran)/2)
                        elif upper_x_bound == 128:
                            lower_y_bound += math.floor((y_ran-x_ran)/2)
                            upper_y_bound -= math.ceil((y_ran-x_ran)/2)
                        elif lower_y_bound == 0:
                            lower_x_bound += math.floor((y_ran-x_ran)/2)
                            upper_x_bound -= math.ceil((y_ran-x_ran)/2)
                        elif lower_x_bound == 0:
                            lower_y_bound += math.floor((y_ran-x_ran)/2)
                            upper_y_bound -= math.ceil((y_ran-x_ran)/2)
                        lower_x_bound = max(0, (min(lower_x_bound, 128)))
                        lower_y_bound = max(0, (min(lower_y_bound, 128)))
                        upper_x_bound = max(0, (min(upper_x_bound, 128)))
                        upper_y_bound = max(0, (min(upper_y_bound, 128)))
                        x_ran = (upper_x_bound - lower_x_bound)
                        y_ran = (upper_y_bound - lower_y_bound)
                        if (x_ran > y_ran):
                            while (x_ran > y_ran):
                                lower_x_bound += 1
                                x_ran = (upper_x_bound - lower_x_bound)
                        elif (y_ran > x_ran):
                            while (y_ran > x_ran):
                                lower_y_bound += 1
                                y_ran = (upper_y_bound - lower_y_bound)
  
                  
                    # 3. Select receptive field
                    subsection = input_event[batch_dim, :, lower_y_bound:upper_y_bound, lower_x_bound:upper_x_bound]
                    subsection = subsection.view(1, subsection.size(0), subsection.size(1), subsection.size(2))

                    # 4. Compute kernel size (eq. 9 in paper)
                    kernel_size = math.floor(subsection.size(2)/12)
                    RoI_batch = torch.nn.functional.avg_pool2d(subsection, kernel_size)

                    # If output dimension is not correct, adjust receptive field pixel width to ensure correct output dim 
                    if ((RoI_batch.size(2) != 12)) or (RoI_batch.size(3) != 12):
                        new_y_low = lower_y_bound 
                        new_x_low = lower_x_bound
                        new_y_high = upper_y_bound 
                        new_x_high = upper_x_bound
                        while ((RoI_batch.size(2) != 12)) or (RoI_batch.size(3) != 12):
                            if new_x_low < 8:
                                new_x_high += 1
                            else:
                                new_x_low -= 1
                            if new_y_low < 8:
                                new_y_high += 1
                            else:
                                new_y_low -= 1
                            subsection = input_event[batch_dim, :, new_y_low:new_y_high, new_x_low:new_x_high]
                            subsection = subsection.view(1, subsection.size(0), subsection.size(1), subsection.size(2))
                            kernel_size = math.floor(subsection.size(2)/12)
                            RoI_batch = torch.nn.functional.avg_pool2d(subsection, kernel_size)
                    assert(RoI_batch.size(2) == 12)
                    assert(RoI_batch.size(3) == 12)
                    RoI[batch_dim] = RoI_batch
    
                gamma = torch.zeros((input_event.size(0), 1)).to(self.device)
                gamma = torch.add(gamma, 4)
                RoI=RoI*torch.reshape(gamma,([-1,1,1,1]))

                gx_,gy_, log_delta = torch.chunk(xyd_out, 3, 1)
                delta=4*(log_delta + 1)
                gx=self.A/2*(gx_+1)
                gy=self.B/2*(gy_+1)
                self.delt = delta
                self.gex = gx
                self.gey = gy
            else:
                # Perform Gaussian kernel ROI generation
                Fx2,Fy2,gamma2, self.delt, self.gex, self.gey, self.sig = gaussianKernel.filter_matrices(
                xyd_out, self.RoI_size, self.batch_size, self.A, self.B, device=self.device)
                RoI = gaussianKernel.ROI_generation(input_event, Fx2, Fy2, gamma2, device=self.device)
 

            RoI_activ = RoI.clone()
            
            if spars_loss:
                glimpseCNN1_activ = self.glimpseCNN[0](RoI) 
                glimpseCNN1_activ = self.glimpseCNN[1](glimpseCNN1_activ) 
                glimpseCNN1_activ = self.glimpseCNN[2](glimpseCNN1_activ) 
                glimpseCNN1_activ = self.glimpseCNN[3](glimpseCNN1_activ) 
                glimpseCNN1_activ = glimpseCNN1_activ.clone()

                glimpseCNN2_activ = self.glimpseCNN[4](glimpseCNN1_activ) 
                glimpseCNN2_activ = self.glimpseCNN[5](glimpseCNN2_activ) 
                glimpseCNN2_activ = self.glimpseCNN[6](glimpseCNN2_activ) 
                glimpseCNN2_activ = self.glimpseCNN[7](glimpseCNN2_activ) 
                glimpseCNN2_activ = glimpseCNN2_activ.clone()
                glimpse_out = glimpseCNN2_activ
            else:
                glimpse_out = self.glimpseCNN(RoI)

            glimpse_out = glimpse_out.view(glimpse_out.size(0), self.lin_size)
            

          
            if spars_loss:
                cl_out = self.classfication[0](glimpse_out)     
                cl_out = self.classfication[1](cl_out)  
                cl_out_2 = self.classfication[2](cl_out)   
                cl_out_3 = self.classfication[3](cl_out_2) 
                cl_out_3_activ = cl_out_3.clone()                      
                cl_out = self.classfication[4](cl_out_3)             
                cl_out = self.classfication[5](cl_out)             
           
            else:
                cl_out = self.classfication(glimpse_out)         

            if spars_loss:
                if train == 0:
                    activ_dict = {'input': input_activ, 'findCNN1':findCNN1_activ, 'findCNN2':findCNN2_activ, 'findCNN3':findCNN3_activ, 
                                  'roi': RoI_activ, 'glimpseCNN1':glimpseCNN1_activ, 'glimpseCNN2':glimpseCNN2_activ,  
                                   'RNN':RNN_activ, 'clout2': cl_out_3_activ}

                    all_activs.append(activ_dict)
                else:
                    activ_dict = {'findCNN1':findCNN1_activ, 'findCNN2':findCNN2_activ, 'findCNN3':findCNN3_activ, 
                                   'glimpseCNN1':glimpseCNN1_activ, 'glimpseCNN2':glimpseCNN2_activ,  
                                   'RNN':RNN_activ, 'clout2': cl_out_3_activ}


            output_list.append(cl_out)

            if train == 0:
                if tt == 0:
                    RoI_out = RoI[batch_to_visualize]
                    gex_out = self.gex[batch_to_visualize]
                    gey_out = self.gey[batch_to_visualize]
                    delt_out = self.delt[batch_to_visualize]
                else:
                    if tt == 1:
                        RoI_out = torch.stack((RoI_out, RoI[batch_to_visualize]), 3)
                    else:
                        RoI_1 = RoI[batch_to_visualize][:,:,:, None]
                        RoI_out = torch.cat((RoI_out, RoI_1), 3)
                    gex_out = torch.cat((gex_out, self.gex[batch_to_visualize]), )
                    gey_out = torch.cat((gey_out, self.gey[batch_to_visualize]), )
                    delt_out = torch.cat((delt_out, self.delt[batch_to_visualize]), )
                    RoI_params = [gex_out, gey_out, delt_out]
   
        output = torch.stack(output_list).mean(dim=0)
      
        if train == 0:
            if spars_loss:
                return output, event_data[batch_to_visualize], RoI_out, RoI_params, all_activs
            else:
                return output, event_data[batch_to_visualize], RoI_out, RoI_params 
      
        if spars_loss:
            return output, activ_dict
        return output


    def forward(self, event_data, train=0, inf_ts=0, batch_to_visualize=0, spars_loss=False, DAP=False):
        """forward function

        Args:
            event_data (Tensor): event data input (batch_size, channel, x, y, spike_ts)
        """
        return self.TRIPforward(event_data, train, inf_ts, batch_to_visualize, spars_loss, DAP)
        

