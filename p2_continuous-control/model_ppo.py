import torch.nn as nn
import torch.nn.functional as F


# The output should be a continuous value between [-1,1]
# Which corresponds to the applied torque to 4 joints?
# This task should not be too hard, because the reward function is quite constant, which helps the agent to follow the trajectory
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        # TODO: Define the size of the Neural Network
        # Is this amount of layers enough?
        # TODO: CNN doesn't make sense for 37 states! It's made for images...

        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride 
        # (round up if not an integer)

        # output = 20x20 here
        self.conv1 = nn.Conv2d(2, 1, kernel_size=4, stride=4) 
        self.size=1*20*20
        
        # 1 fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        
        # TODO: Improve the model 
        x = F.relu(self.conv(x))
        # flatten the tensor
        x = x.view(-1,self.size)
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))

        # TODO: find a output coherent with the problem
        return x