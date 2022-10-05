import torch
import torch.nn as nn
import torch.nn.functional as F

from env_carla import IM_HEIGHT
from env_carla import IM_WIDTH
from env_carla import N_ACTIONS

class DQN(nn.Module):
    def __init__(self, withAE=False):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)      # 5x5 kernel that moves 2 pixels per iteration
        self.bn1 = nn.BatchNorm2d(16)                               # Normalize input features so they are on the same scale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        #Compute number of linear input connections after conv2d layers (https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        if withAE:
            convw1 = conv2d_size_out(IM_WIDTH*2)
        else:
            convw1 = conv2d_size_out(IM_WIDTH)
        convh1 = conv2d_size_out(IM_HEIGHT)

        convw2 = conv2d_size_out(convw1)
        convh2 = conv2d_size_out(convh1)

        convw3 = conv2d_size_out(convw2)
        convh3 = conv2d_size_out(convh2)

        linear_input_size = convw3 * convh3 * 32                    # width * height * channels
        self.head = nn.Linear(linear_input_size, N_ACTIONS)

    def forward(self, t):
        '''Called with either one element to determine next action, or a batch during optimization'''
        t = self.conv1(t)
        t = self.bn1(t)
        t = F.relu(t)           # Activation function

        t = F.relu(self.bn2(self.conv2(t)))
        t = F.relu(self.bn3(self.conv3(t)))
        return self.head(t.view(t.size(0), -1)) # view reshapes the tensor to 't.size(0)' x '-1' (-1 means the number is unknown and should be determined)