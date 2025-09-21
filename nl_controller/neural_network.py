import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import os

Model = collections.namedtuple('Model', 'phi h options')

class Phi_Net(nn.Module):
    def __init__(self, options):
        super(Phi_Net, self).__init__()

        self.fc1 = nn.Linear(options['dim_x'], 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, options['dim_a'] - 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if len(x.shape) == 1:
            return torch.cat([x, torch.ones(1)])
        else:
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)


class H_Net_CrossEntropy(nn.Module):
    def __init__(self, options):
        super(H_Net_CrossEntropy, self).__init__()
        self.fc1 = nn.Linear(options['dim_a'], 20)
        self.fc2 = nn.Linear(20, options['num_c'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(modelname, modelfolder='./models/'):
    model_path = os.path.join(modelfolder, f"{modelname}.pth")
    model = torch.load(model_path)
    options = model['options']

    phi_net = Phi_Net(options=options)
    h_net = H_Net_CrossEntropy(options)

    phi_net.load_state_dict(model['phi_net_state_dict'])
    h_net.load_state_dict(model['h_net_state_dict'])

    phi_net.eval()
    h_net.eval()

    return Model(phi_net, h_net, options)