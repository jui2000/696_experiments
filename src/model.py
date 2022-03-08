import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
                           
    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc6(out)
        return out
