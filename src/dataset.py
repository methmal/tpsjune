import torch
class TabDataset:
    def __init__(self ,features , targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self , item):
        # print(self.features[item , :])
        # print(self.targets[item , :])
        return {
                "x" : torch.tensor(self.features[item , :], dtype=torch.float) , 
                "y" : torch.tensor(self.targets[item , :] , dtype=torch.long),
                }



