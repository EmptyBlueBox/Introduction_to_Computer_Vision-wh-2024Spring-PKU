from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    '''
        The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
        Args:
        d: the dimension of the global feature, default is 1024.
        segmentation: whether to perform segmentation, default is True.
    '''

    def __init__(self, segmentation=False, d=1024):
        super(PointNetfeat, self).__init__()
        self.segmentation = segmentation
        self.d = d

        # Define the layers in the feature extractor using Conv1d
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, d, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d)

    def forward(self, x):
        '''
            If segmentation == True
                return the concatenated global feature and local feature. # (B, d+64, N)
            If segmentation == False
                return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
            Here, B is the batch size, N is the number of points, d is the dimension of the global feature.
        '''
        x = x.permute(0, 2, 1)  # (B, N, D) to (B, D, N)
        B, D, N = x.size()  # B: batch size, D: dimension of input points, N: number of points

        # Forward pass through the layers
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        first_layer_feature = x
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.bn3(self.conv3(x))  # (B, d, N)
        local_feature = x

        # Global feature by max pooling
        global_feature, _ = torch.max(x, 2)  # (B, d)

        if self.segmentation:
            global_feature = global_feature.view(B, self.d, 1).repeat(1, 1, N)  # (B, d, N)
            x = torch.cat([first_layer_feature, global_feature], 1)  # (B, d+64, N)
            return x
        else:
            return global_feature, local_feature.permute(0, 2, 1)  # (B, d), (B, N, d)


class PointNetCls1024D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''

    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        self.k = k
        self.pointnetfeat = PointNetfeat(segmentation=False)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
        '''
        x, _ = self.pointnetfeat(x)  # (B, d)
        x = F.relu(self.bn1(self.fc1(x)))   # (B, 512)
        x = F.relu(self.bn2(self.fc2(x)))   # (B, 256)
        x = self.fc3(x)     # (B, k)
        x = F.log_softmax(x, dim=-1)  # (B, k)
        return x, _


class PointNetCls256D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the upper right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''

    def __init__(self, k=2):
        super(PointNetCls256D, self).__init__()
        self.k = k
        self.pointnetfeat = PointNetfeat(segmentation=False, d=256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=256)
        '''
        x, _ = self.pointnetfeat(x)  # (B, d)
        x = F.relu(self.bn1(self.fc1(x)))   # (B, 128)
        x = self.fc2(x)  # (B, k)
        x = F.log_softmax(x, dim=-1)    # (B, k)
        return x, _


class PointNetSeg(nn.Module):
    '''
        The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''

    def __init__(self, k=2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.pointnetfeat = PointNetfeat(segmentation=True)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        '''
            Input:
                x: the concatenated global feature and local feature. # (B, d+64, N)
            Output:
                the log softmax of the segmentation result. # (B, N, k)
        '''
        x = self.pointnetfeat(x)  # (B, d+64, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 512, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 256, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, N)
        x = self.conv4(x)  # (B, k, N)

        x = x.permute(0, 2, 1)  # (B, N, k)

        x = F.log_softmax(x, dim=-1)  # (B, N, k)
        return x
