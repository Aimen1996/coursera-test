import dataset as ds
import utils
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import time
import pandas as pd
import datetime
import torch.nn as nn
import torch.optim as opt
import non_sta_trans_difference as tst
import numpy as np
import lmdb
from tqdm import tqdm
import matplotlib.pyplot as plt

support_used = 1
start_support = 40

seq_len=120
label_len =int(seq_len/2)
pred_len=120

dim_val = 512

max_ = seq_len
batch_first = True
num_predicted_features=1

test_size = 0.1
batch_size = 32

window_size = seq_len + pred_len # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_ = seq_len

min_time=34992
max_time=39052


def difference(data, interval=1):
    diff=[]
    for i in range( len(data)):
        value = data[i] - data[i -interval]
        diff.append(value)
    return diff[1:]

def standarizeData(data): #z-score
    a = np.mean(data)
    b = np.std(data)
    z = (data-a) / b

    return z

db_path = '/home/aimen/PycharmProjects/HydraulicSupport_pressure/'
data = []
for id in range(start_support, start_support + support_used):  ## append the data for support id
    env = lmdb.open(db_path+ "support_id_lmdb" + str(id))
    txn = env.begin(write=False)
    temp = []
    for sample_time in range(min_time, max_time):
        data_value = txn.get(str(sample_time).encode())
        # print(sample_time, data_value)
        temp.append(np.float32(float(data_value)))

# data=standarizeData(temp) #prepare the data using z-score
# data=np.array(data)


# data=difference(data)
# print(data)
# exit()
data=np.array(temp)



training_data = data[:-(round(len(data)*test_size))]
testing_data=data[-(round(len(data)*test_size)):]
print(training_data,len(training_data))
# exit()


# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
# Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data,
    window_size=window_size,
    step_size=step_size)

testing_indices = utils.get_indices_entire_sequence(
    data=testing_data,
    window_size=window_size,
    step_size=step_size)

# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor(training_data).unsqueeze(1).float(),
    indices=training_indices,
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,

    )

testing_data = ds.TransformerDataset(
    data=torch.tensor(testing_data).unsqueeze(1).float(),
    indices=testing_indices,
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,

    )

model = tst.TimeSeriesTransformer(
    input_size=support_used,
    batch_first=batch_first,
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    num_predicted_features=1
    )

optimizer = opt.Adam(model.parameters(),betas=(0.9, 0.98), eps=1e-09, lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)# print(src,src.shape)
LossFunc = nn.MSELoss()

def train(dataloader_train):
    checkpoint = torch.load('nsd5.pt')
    model.load_state_dict(checkpoint['model.state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
    model.train()
    epoch_num = 300 #600
    tt=time.time()
    for epoch in range(epoch_num):
        # to=time.time()
        loss = 0
        train_bar = tqdm(dataloader_train)
        for i, data in enumerate(train_bar):
            batch_x, batch_y =data
            # print(batch_x.shape,batch_y.shape)

            output = model(batch_x,batch_y)
            # batch_y = batch_y[:, -pred_len:, :]
            loss = LossFunc(batch_y, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss += loss
            train_bar.desc = "train epoch[{}/{}] " \
                             "loss:{:.3f}".format(epoch + 1, epoch_num, loss)
            # scheduler.step()
        print("------", loss / len(dataloader_train))
    t_ = time.time()-tt
    time_format = time.strftime("%H:%M:%S", time.gmtime(t_))
    print("Ttaining time :-", time_format)
    torch.save({
        'epoch':epoch,
        'model.state_dict':model.state_dict(),
        'optimizer.state_dict':optimizer.state_dict(),
        'loss':loss},
        'nsd6.pt')

def eval(dataloader_test):
    checkpoint = torch.load('nsd5.pt')
    model.load_state_dict(checkpoint['model.state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])

    model.eval()
    for i, data in enumerate(dataloader_test):
        batch_x_, batch_y_=data
        # print(data)

        output_ = model(batch_x_, batch_y_)




        loss = LossFunc(batch_y_, output_)
        #
        print(loss)
        output_ = output_.squeeze(0).detach().numpy()
        batch_y_ = batch_y_.squeeze(0).detach().numpy()
        batch_x_ = batch_x_.squeeze(0).detach().numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(output_, 'r')
        plt.plot(batch_y_, 'b')
        plt.show()
def main():
    dataloader_train = DataLoader(training_data, batch_size=128, shuffle=False)
    dataloader_test = DataLoader(testing_data, batch_size=1, shuffle=True)
    # dataloader_test = DataLoader(training_data, batch_size=1, shuffle=True)

    train(dataloader_train)
    # eval(dataloader_test)
if __name__ == "__main__":
    main()