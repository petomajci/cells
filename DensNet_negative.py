import torch
import torch.nn as nn

import torchvision
import torch.nn.functional as F

from myDensenets import MyDenseNet

class DensNet(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6, pretrained=True, layers=201):
        super().__init__()
        if layers == 121:
            print("preloading 121")
            preloaded = torchvision.models.densenet121(pretrained=pretrained)
        else:
            print("preloading 201")
            preloaded = torchvision.models.densenet201(pretrained=pretrained)
        #preloaded = MyDenseNet(32, (6, 12, 24, 16, 8), 64)
        
        # this one was used for most experiments
        #preloaded = torchvision.models.DenseNet(32, (6, 12, 24, 16, 8), 64)

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

        self.classifier = nn.Bilinear(preloaded.classifier.weight.shape[1],4, num_classes, bias=True)
        del preloaded

    def forward(self, x):
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
            #out = self.classifier(out,x[2])

            features1 = self.features(x[1])
            out1 = F.relu(features1, inplace=True)
            out1 = F.adaptive_avg_pool2d(out1, (1, 1)).view(features1.size(0), -1)
            #out1 = self.classifier(out1,x[2])

            both = torch.stack((out,out1),dim=2)
            result = torch.matmul(both,x[3])[:,:,0]  # do my image features MINUS negative control features
            #return result
            return self.classifier(result,x[2])
            #return self.classifier(F.relu(result, inplace=True),x[2])
