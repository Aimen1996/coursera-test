import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple


class RnnDataset(Dataset):

    def __init__(self,
                 data: torch.tensor,
                 indices: list,
                 seq_len: int,
                 pred_len: int
                 ) -> None:

        super().__init__()
        self.indices = indices
        self.data = data
        print("From get_src_trg: data size = {}".format(data.size()))
        self.seq_len = seq_len
        self.pred_len = pred_len


    def __getitem__(self, index):
        # for i in (self.indices):
        # print('_______________',self.indices[index][0])
        # print('_______________', self.indices[index][1])
        # exit()


        seq_x = self.data[self.indices[index][0]: self.indices[index][0]+self.seq_len]# data_y[48:192]
        seq_y=self.data[self.indices[index][0]+self.seq_len:self.indices[index][0]+self.seq_len+self.pred_len]

        return seq_x, seq_y
            #,seq_dec #seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.indices)