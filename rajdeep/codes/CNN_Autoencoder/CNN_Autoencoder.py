import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.nn import Linear, Conv1d, Flatten, MaxPool1d, ConvTranspose1d
from torch.nn.functional import relu, sigmoid, softmax
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from logger import Logger



class CNNAutoencoder(nn.Module):
    def __init__(self, opt):
        super(CNNAutoencoder, self).__init__()
        self.device = opt["device"]

        # Basic Network Params
        self.length = opt["length"]
        self.kernel_size = opt["kernel_size"]
        self.num_filters_1 = opt["num_filters_1"]
        self.num_filters_2 = opt["num_filters_2"]
        self.num_filters_3 = opt["num_filters_3"]
        self.output_layer_size = opt["output_layer_size"]
        self.padding = opt["padding"]

        self.conv_stride = opt["conv_stride"]
        self.pool_size_1 = opt["pool_size_1"]
        self.pool_size_2 = opt["pool_size_2"]
        self.pool_strides_1 = opt["pool_strides_1"]
        self.pool_strides_2 = opt["pool_strides_2"]
        
        self.model_directory = opt["model_directory"]
        self.model_name = opt["model_name"]
        self.batch_size = opt["batch_size"]
        self.epochs = opt["epochs"]
        self.lr =  opt["lr"]
        self.save_every = opt["save_every"]

        
        # Encoder
        self.conv_layer_1 = Conv1d(in_channels=1, out_channels=self.num_filters_1, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        # self.pool_layer_1 = MaxPool1d(kernel_size=self.pool_size_1, stride=self.pool_strides_1)
        self.lout_conv1 = int((self.length + 2*0- 1*(self.kernel_size-1)-1+self.conv_stride)/self.conv_stride)

        self.conv_layer_2 = Conv1d(in_channels=self.num_filters_1, out_channels=self.num_filters_2, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        # self.pool_layer_2 = MaxPool1d(kernel_size=self.pool_size_2, stride=self.pool_strides_2)
        self.lout_conv2 = int((self.lout_conv1 + 2*0- 1*(self.kernel_size-1)-1+self.conv_stride)/self.conv_stride)

        self.conv_layer_3 = Conv1d(in_channels=self.num_filters_2, out_channels=self.num_filters_3, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        # self.pool_layer_3 = 

        self.deconv_layer_1 = ConvTranspose1d(in_channels=self.num_filters_3, out_channels=self.num_filters_2, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.deconv_layer_2 = ConvTranspose1d(in_channels=self.num_filters_2, out_channels=self.num_filters_1, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.deconv_layer_3 = ConvTranspose1d(in_channels=self.num_filters_1, out_channels=1, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        # self.output_layer = Linear(in_features=self.num_filters_2*self.lout_conv2, out_features=self.output_layer_size)
    
        self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr)

        self.Criterion = nn.MSELoss()

    def get_batch_data(self, batch):
        kpi_history = batch[0]
        kpi_history = (torch.stack(kpi_history)).permute(1, 0)
        kpi_history = kpi_history.unsqueeze(1)

        # current_kpi = batch[1].unsqueeze(1).float()

        # return kpi_history, current_kpi
        labels = batch[1]
        labels = (torch.stack(labels)).permute(1, 0)
        labels = labels.unsqueeze(1)
        return kpi_history, labels

    def forward(self, kpi_history):
        conv_layer_1_output = relu(self.conv_layer_1(kpi_history))
        conv_layer_2_output = relu(self.conv_layer_2(conv_layer_1_output))
        conv_layer_3_output = relu(self.conv_layer_3(conv_layer_2_output))
        
        deconv_layer_1_output = relu(self.deconv_layer_1(conv_layer_3_output))
        deconv_layer_2_output = relu(self.deconv_layer_2(deconv_layer_1_output))
        deconv_layer_3_output = self.deconv_layer_3(deconv_layer_2_output)

        return conv_layer_3_output, deconv_layer_3_output

    def process_batch(self, batch, train=False):
        kpi_history, current_kpi = self.get_batch_data(batch)
        kpi_history = kpi_history.float()

        kpi_history = kpi_history.to(self.device)
        current_kpi = (current_kpi.to(self.device)).squeeze(1).long()
        
        if train:
            self.optimizer.zero_grad()
        
        hidden_representation, output_scores = self(kpi_history)
        batch_loss = self.Criterion(output_scores, kpi_history)
        if train:
            batch_loss.backward()
            self.optimizer.step()
        return hidden_representation, output_scores, batch_loss.item()

    def evaluate_model(self, dataLoader):
        self.eval()
        total_loss = 0
        latent_representations = []
        for batch in dataLoader:
            hidden_representation, _, batch_loss = self.process_batch(batch, train=False)
            total_loss += batch_loss
            latent_representations.append(hidden_representation)
        latent_representations = torch.cat(latent_representations, dim=0)
        print("Evaluation Loss: %f"%(total_loss))
        return latent_representations, total_loss

    def train_model(self, trainDataLoader, devDataLoader):
        logger = Logger("../logger/")
        self.optimizer.zero_grad()
        self.evaluate_model(devDataLoader)
        self.train()
        ins=0
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            for idx, batch in tqdm(enumerate(trainDataLoader)):
                _, _, batch_loss = self.process_batch(batch, train=True)
                train_loss += batch_loss
            dev_loss = self.evaluate_model(devDataLoader)
            print("Iteration: %d,Train Loss = %f" %(epoch, train_loss))
            
            # # Logging parameters
            # p = list(self.named_parameters())
            # logger.scalar_summary("Train Loss", train_loss, ins+1)
            # logger.scalar_summary("Dev Loss", dev_loss, ins+1)
            # for tag, value in self.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(tag, value.data.cpu().numpy(), ins+1)
            #     if value.grad != None:
            #         logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), ins+1)
            ins+=1
            if (epoch+1)%self.save_every==0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))
    
    
        