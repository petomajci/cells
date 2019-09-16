import torch
import torch.nn as nn

import torchvision
import torch.nn.functional as F

class DensNet_controls(nn.Module):
    def __init__(self, num_classes=1000, Ncontrols=31, num_channels=6, pretrained=True):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=pretrained)
        #preloaded = torchvision.models.DenseNet(16, (6, 12, 24, 16), 64)

        # Freeze model parameters
        #for param in preloaded.parameters():
        #    param.requires_grad = False

        w = preloaded.features.conv0.weight.clone()
        #print(w.shape)

        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3, bias=True)

        if (pretrained):
            print(self.features.conv0.weight.shape)
            self.features.conv0.weight = nn.Parameter(torch.cat((w,w),dim=1))
            #self.features.conv0.weight = nn.Parameter(torch.cat((w,
            #                        0.5*(w[:,:1,:,:]+w[:,2:,:,:])),dim=1))

        self.classifier = nn.Bilinear(1024 + Ncontrols,4, num_classes, bias=True) #grow=16,121 ->516, 121 -> 1024, 201->1920, 264->2688
        del preloaded

        self.controlNet = nn.Sequential(nn.Linear(31,62),nn.ReLU(inplace=True),nn.Linear(62,31),nn.ReLU(inplace=True))

    def forward(self, x):
        #x[0] -img1, x[1] - img2, x[2] - controls1, x[3] - controls2, x[4] cell line
        USEBOTHSITES=1

        if USEBOTHSITES==0:
            features = self.features(x[0])
            features1 = F.relu(features, inplace=True)
            features1 = F.adaptive_avg_pool2d(features1, (1, 1)).view(features.size(0), -1)
            out = self.classifier(features1,x[1])
            #return features1#, out
            return out
        else:
            features = self.features(x[0])
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
            control_signal = self.controlNet(x[2])
            out = torch.cat((out, control_signal), dim=1)
            out = self.classifier(out,x[4])

            features1 = self.features(x[1])
            out1 = F.relu(features1, inplace=True)
            out1 = F.adaptive_avg_pool2d(out1, (1, 1)).view(features1.size(0), -1)
            control_signal1 = self.controlNet(x[3])
            out1 = torch.cat((out1, control_signal1),dim=1)
            out1 = self.classifier(out1,x[4])

            #return torch.max(torch.cat((out.unsqueeze(1), out1.unsqueeze(1)), dim=1),1)[0]
            return torch.sum(torch.cat((out.unsqueeze(1), out1.unsqueeze(1)), dim=1),1)
