import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.nn import LSTM, Linear, Sigmoid, GRU

from tqdm import tqdm

from logger import Logger



class LSTMModel(nn.Module):
    def __init__(self, opt):
        super(LSTMModel, self).__init__()
        self.device = opt["device"]

        # Basic Network Params
        self.input_size = opt["input_size"]
        self.hidden_size = opt["hidden_size"]
        self.num_layers = opt["num_layers"]
        self.batch_first = True
        self.bidirectional = opt["bidirectional"]
        
        self.model_directory = opt["model_directory"]
        self.model_name = opt["model_name"]
        self.batch_size = opt["batch_size"]
        self.epochs = opt["epochs"]
        self.lr =  opt["learning_rate"]
        self.save_every = opt["save_every"]

        self.gru = GRU(input_size = self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        self.output_layer = Linear(in_features = self.hidden_size, out_features = 1)
    
        self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr)
        self.Criterion = nn.MSELoss()

    def get_batch_data(self, batch):
        kpi_history = batch[0]
        kpi_history = (torch.stack(kpi_history)).permute(1, 0)
        kpi_history = kpi_history.unsqueeze(2)

        current_kpi = batch[1].unsqueeze(1)

        return kpi_history, current_kpi

    def forward(self, kpi_history, current_kpi):
        kpi_history = kpi_history.float()
        _, h = self.gru(kpi_history)
        h = h.squeeze(0)
        output = self.output_layer(h)
        return output

    def process_batch(self, batch, train=False):
        kpi_history, current_kpi = self.get_batch_data(batch)

        kpi_history = kpi_history.to(self.device)
        current_kpi = current_kpi.to(self.device)
        
        if train:
            self.optimizer.zero_grad()
        
        output_scores = self(kpi_history, current_kpi)
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
            # torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch))
    
    
        