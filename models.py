import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalMaxPool1d(nn.Module):
    '''
    (batch_size, channel, len) -> (batch_size, channel, 1)
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=int(x.shape[-1]))
    
    
class CNN(nn.Module):
    def __init__(self,num_channels, kernel_sizes):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(
                nn.Conv1d(in_channels=50, out_channels=c, kernel_size=k)
            )
        self.pool = nn.Sequential(
            nn.SELU(),
            nn.Dropout(0.5),
            GlobalMaxPool1d()
        )
        # ( batch_size, sum(num_channels) ) -> (batch_size, num_classes)
        self.fc1 = nn.Linear(sum(num_channels), 2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  
        # (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim, seq_len)
        
        x = [conv(x) for conv in self.convs]
        #[ (batch_size, num_output_channel, seq_len - kernel_size + 1) ] ->
        # [ (batch_size, num_output_channel, 1) ] ->
        # [ (batch_size, num_output_channel) ]
        
        x = [self.pool(item).squeeze(-1) for item in x]
        # [ (batch_size, num_output_channel) ], len = num_kernels ->
        # ( batch_size, sum(num_channels) )
        
        x = torch.cat(x, dim=1)
        # ( batch_size, sum(num_channels) ) -> (batch_size, num_classes)
        
        x = self.fc1(x)
        #x = F.softmax(x, dim=1)
        return x

    
class RNN(nn.Module):
    def __init__(self,  hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(50, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out    
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linears = nn.Sequential(
            # (batch, max_seq_len * embed_dim) -> (batch, 1024)
            nn.Linear(64 * 50, 512),
            nn.BatchNorm1d(num_features=512),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )
        
    def forward(self, x):
        x = self.linears(x.view(x.shape[0], -1))
        return x
