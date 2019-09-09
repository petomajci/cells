
class FullTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        #assert len(full_ds)<offset+length, Exception(“Parent Dataset not long enough”)
        #super(FullTrainingDataset, self).init()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i+self.offset]

def trainTestSplit(dataset, val_share=0.2):
    #print(f'size: {len(dataset)}')
    val_offset = int(len(dataset)*(1-val_share))
    return FullTrainingDataset(dataset, 0, val_offset), FullTrainingDataset(dataset, val_offset, len(dataset)-val_offset)
