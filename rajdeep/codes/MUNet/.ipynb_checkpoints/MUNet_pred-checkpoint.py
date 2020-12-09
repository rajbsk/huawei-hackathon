import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.nn import Linear, Conv1d, Flatten, MaxPool1d, BatchNorm1d, ConvTranspose1d, Upsample
from torch.nn.functional import relu, sigmoid, softmax
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from logger import Logger

def dice_loss(input, target):
    smooth = 0.000001

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class MUNet(nn.Module):
    def __init__(self, opt):
        super(MUNet, self).__init__()
        self.device = opt["device"]

        # Basic Network Params
        self.length = opt["length"]
        self.kernel_size = opt["kernel_size"]
        self.num_filters_1 = opt["num_filters_1"]
        self.num_filters_2 = opt["num_filters_2"]
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

        # Encoder Layers
        # 256,1
        self.conv_block_1_layer_1 = Conv1d(in_channels=1, out_channels=16, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_1_layer_1 = BatchNorm1d(16)
        self.conv_block_1_layer_2 = Conv1d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_1_layer_2 = BatchNorm1d(16)
        self.pool_block_1 = MaxPool1d(kernel_size=4, stride=4)
        # 64, 16

        # 64, 16
        self.conv_block_2_layer_1 = Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_2_layer_1 = BatchNorm1d(32)
        self.conv_block_2_layer_2 = Conv1d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_2_layer_2 = BatchNorm1d(32)
        self.pool_block_2 = MaxPool1d(kernel_size=4, stride=4)
        # 16, 32

        # 16, 32
        self.conv_block_3_layer_1 = Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_3_layer_1 = BatchNorm1d(64)
        self.conv_block_3_layer_2 = Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_3_layer_2 = BatchNorm1d(64)
        self.pool_block_3 = MaxPool1d(kernel_size=4, stride=4)
        # 4, 64

        # 4, 64
        self.conv_block_4_layer_1 = Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_4_layer_1 = BatchNorm1d(128)
        self.conv_block_4_layer_2 = Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_block_4_layer_2 = BatchNorm1d(128)
        # 4, 128

        # Decoder Layers
        # 4, 128
        self.Upsample_block_1 = Upsample(scale_factor=4)
        self.deconv_block_1_layer_0 = Conv1d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.deconv_block_1_layer_1 = Conv1d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_1_layer_1 = BatchNorm1d(64)
        self.deconv_block_1_layer_2 = Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_1_layer_2 = BatchNorm1d(64)
        # 16, 64

        # 16, 64
        self.Upsample_block_2 = Upsample(scale_factor=4)
        self.deconv_block_2_layer_0 = Conv1d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.deconv_block_2_layer_1 = Conv1d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_2_layer_1 = BatchNorm1d(32)
        self.deconv_block_2_layer_2 = Conv1d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_2_layer_2 = BatchNorm1d(32)
        # 64, 32

        # 64, 32
        self.Upsample_block_3 = Upsample(scale_factor=4)
        self.deconv_block_3_layer_0 = Conv1d(in_channels=32, out_channels=16, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.deconv_block_3_layer_1 = Conv1d(in_channels=32, out_channels=16, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_3_layer_1 = BatchNorm1d(16)
        self.deconv_block_3_layer_2 = Conv1d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_3_layer_2 = BatchNorm1d(16)
        # 256, 16

        # 256, 16
        self.Upsample_block_4 = Upsample(scale_factor=4)
        self.deconv_block_4_layer_1 = Conv1d(in_channels=32, out_channels=16, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_4_layer_1 = BatchNorm1d(16)
        self.deconv_block_4_layer_2 = Conv1d(in_channels=16, out_channels=16, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        self.batchnorm_deconv_block_4_layer_2 = BatchNorm1d(1)

        self.output = Conv1d(in_channels=16, out_channels=1, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.padding)
        # 256, 1 
    
        self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr)

        self.Criterion = dice_loss
        # self.Criterion = nn.BCELoss()

    def get_batch_data(self, batch):
        kpi_history = batch[0]
        kpi_history = (torch.stack(kpi_history)).permute(1, 0)
        kpi_history = kpi_history.unsqueeze(1)

        labels = torch.stack(batch[1]).permute(1, 0).float()

        return kpi_history, labels

    def forward(self, kpi_history):
        kpi_history = kpi_history.float()
        conv_block_1_layer_1_output = relu(self.batchnorm_block_1_layer_1(self.conv_block_1_layer_1(kpi_history))) #BX16X256
        conv_block_1_layer_2_output = relu(self.batchnorm_block_1_layer_2(self.conv_block_1_layer_2(conv_block_1_layer_1_output)))#BX16X256
        conv_block_1_output  = self.pool_block_1(conv_block_1_layer_2_output)#BX16X64
        
        conv_block_2_layer_1_output = relu(self.batchnorm_block_2_layer_1(self.conv_block_2_layer_1(conv_block_1_output)))#BX32X64
        conv_block_2_layer_2_output = relu(self.batchnorm_block_2_layer_2(self.conv_block_2_layer_2(conv_block_2_layer_1_output)))#BX32X64
        conv_block_2_output  = self.pool_block_1(conv_block_2_layer_2_output)#BX32X16

        conv_block_3_layer_1_output = relu(self.batchnorm_block_3_layer_1(self.conv_block_3_layer_1(conv_block_2_output)))#BX64X16
        conv_block_3_layer_2_output = relu((self.conv_block_3_layer_2(conv_block_3_layer_1_output)))#BX64X16
        conv_block_3_output  = self.pool_block_1(conv_block_3_layer_2_output)#BX64X4

        conv_block_4_layer_1_output = relu(self.batchnorm_block_4_layer_1(self.conv_block_4_layer_1(conv_block_3_output)))#BX128X4
        conv_block_4_layer_2_output = relu(self.batchnorm_block_4_layer_2(self.conv_block_4_layer_2(conv_block_4_layer_1_output)))#BX128X4

        upample_block_1 = self.Upsample_block_1(conv_block_4_layer_2_output)#BX128X16
        upsample_conv_block_1 = self.deconv_block_1_layer_0(upample_block_1)#BX64X16
        upsample_block_1 = torch.cat((upsample_conv_block_1, conv_block_3_layer_2_output), 1)#BX128X16
        deconv_block_1_layer_1_output = relu(self.batchnorm_deconv_block_1_layer_1(self.deconv_block_1_layer_1(upsample_block_1)))#BX64X16
        deconv_block_1_layer_2_output = relu(self.batchnorm_deconv_block_1_layer_2(self.deconv_block_1_layer_2(deconv_block_1_layer_1_output)))#BX64X16

        upsample_block_2 = self.Upsample_block_2(deconv_block_1_layer_2_output)#BX64X64
        upsample_conv_block_2 = self.deconv_block_2_layer_0(upsample_block_2)#BX64X64
        upsample_block_2 = torch.cat((upsample_conv_block_2, conv_block_2_layer_2_output), 1)#BX128X64
        deconv_block_2_layer_1_output = relu(self.batchnorm_deconv_block_2_layer_1(self.deconv_block_2_layer_1(upsample_block_2)))
        deconv_block_2_layer_2_output = relu(self.batchnorm_deconv_block_2_layer_2(self.deconv_block_2_layer_2(deconv_block_2_layer_1_output)))

        upsample_block_3 = self.Upsample_block_3(deconv_block_2_layer_2_output)
        upsample_conv_block_3 = self.deconv_block_3_layer_0(upsample_block_3)
        upsample_block_3 = torch.cat((upsample_conv_block_3, conv_block_1_layer_2_output), 1)
        deconv_block_3_layer_1_output = relu(self.batchnorm_deconv_block_3_layer_1(self.deconv_block_3_layer_1(upsample_block_3)))
        deconv_block_3_layer_2_output = relu(self.batchnorm_deconv_block_3_layer_2(self.deconv_block_3_layer_2(deconv_block_3_layer_1_output)))

        # upsample_block_4 = self.Upsample_block_4(deconv_block_3_layer_2_output)
        # upsample_block_4 = torch.cat((upsample_block_4, conv_block_1_layer_2_output), 1)
        # deconv_block_4_layer_1_output = relu(self.batchnorm_deconv_block_4_layer_1(self.deconv_block_4_layer_1(upsample_block_4)))
        # deconv_block_4_layer_2_output = sigmoid(self.batchnorm_deconv_block_4_layer_2(self.deconv_block_4_layer_2(deconv_block_4_layer_1_output)))
        
        output = sigmoid(self.output(deconv_block_3_layer_2_output))
        return output

    def process_batch(self, batch, train=False):
        kpi_history, current_kpi = self.get_batch_data(batch)

        kpi_history = kpi_history.to(self.device)
        current_kpi = (current_kpi.to(self.device)).squeeze(1).float()
        
        if train:
            self.optimizer.zero_grad()
        
        output_scores = self(kpi_history)
        batch_loss = self.Criterion(output_scores, current_kpi)
        if train:
            batch_loss.backward()
            self.optimizer.step()
        return batch_loss.item()

    def evaluate_model(self, dataLoader):
        self.eval()
        total_loss = 0
        for batch in dataLoader:
            batch_loss = self.process_batch(batch, train=False)
            total_loss += batch_loss
        print("Evaluation Loss: %f"%(total_loss))
        return total_loss

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
                batch_loss = self.process_batch(batch, train=True)
                train_loss += batch_loss
            dev_loss = self.evaluate_model(devDataLoader)
            print("Iteration: %d,Train Loss = %f" %(epoch, train_loss))
            
            # Logging parameters
            p = list(self.named_parameters())
            logger.scalar_summary("Train Loss", train_loss, ins+1)
            logger.scalar_summary("Dev Loss", dev_loss, ins+1)
            for tag, value in self.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), ins+1)
                if value.grad != None:
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), ins+1)
            ins+=1
            if (epoch+1)%self.save_every==0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))
    
    
        