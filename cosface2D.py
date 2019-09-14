import torch
import torch.nn as nn

class LMCL_loss2D(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2, NcellLines = 4):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.cellLines = NcellLines
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, NcellLines))

    def forward(self, feat, label, cellLine):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        #batchCenters = self.centers
        batchCenters = torch.matmul(self.centers, torch.transpose(cellLine,0,1))
        #print(batchCenters.shape)
        norms_c = torch.norm(batchCenters, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(batchCenters, norms_c)
        
        #logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 2)) # Nbatch * Nbatch * Nclasses
        l2 = torch.diagonal(logits,0,dim1=0,dim2=1)
        logits = l2.transpose(dim0=0,dim1=1)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = y_onehot.cuda()

        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)

        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits
