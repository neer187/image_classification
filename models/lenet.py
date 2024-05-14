import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(LeNet, self).__init__()
        # Layer 1: Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        # Subsampling (Max Pooling)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2))

        # Layer 2: Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2))

        # Layer 3: Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))

        # Layer 4: Fully Connected Layer
        self.fc1 = nn.Linear(120, 84)

        # Layer 5: Fully Connected Layer
        self.fc2 = nn.Linear(84, num_classes)


    def forward(self, x):
        # Convolutional layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Convolutional layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Convolutional layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # Flatten the output for the fully connected layer
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
