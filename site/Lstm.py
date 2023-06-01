
import torch
import torch.nn as nn
import torch.optim as opt
import utils
import matplotlib.pyplot as plt
from today1  import RnnDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import sleep
import numpy as np
import lmdb

support_num =1
start_num = 40

n_hidden = 100
layers = 1
test_size = 0.1

min_time=34992
max_time=39052
support_used = 1
start_support = 40


train_len = 10
pre_len=1
future=10
window_size = train_len + pre_len # used to slice data into sub-sequences
step_size = 1
feature=1
batch_size=1


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


data=np.array(temp[:3000])
data=standarizeData(data)


#
# training_data = data[:-(round(len(data)*test_size))]
# testing_data=data[-(round(len(data)*test_size)):]
# print(training_data,len(training_data))
# print(testing_data,len(testing_data))



# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
# Should be training data indices only
indices = utils.get_indices_entire_sequence(
    data=data,
    window_size=window_size,
    step_size=step_size)


training_indices= indices[:-future]
testing_indices = indices[-future:]
print(training_indices)

# exit()


# Making instance of custom dataset class
training_data = RnnDataset(
    data=torch.tensor(data).unsqueeze(1).float(),
    indices=training_indices,
    seq_len=train_len,
    pred_len=pre_len,)

testing_data = RnnDataset(
    data=torch.tensor(data).unsqueeze(1).float(),
    indices=testing_indices,
    seq_len=train_len,
    pred_len=pre_len,)




class RNN(nn.Module):
    def __init__(self, train_len,feature, n_hidden, pre_len,layers,batch_size):  # hidden size
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.feature=feature
        self.batch_size=batch_size
        self.train_len=train_len
        self.pre_len= pre_len
        self.lstm1 = nn.LSTM(feature, self.n_hidden, layers,True)
        self.linear = nn.Linear(self.n_hidden*train_len, pre_len)


    def forward(self, x_train):

        out, (h, c) = self.lstm1(x_train)
        # print(out.shape)
        out = out.view(-1, self.n_hidden * self.train_len)
        # print((out.shape))
        # exit()
        output = self.linear(out)
        output = output.unsqueeze(-1)
        return output

model = RNN(train_len=train_len,feature=feature, n_hidden=n_hidden,pre_len=pre_len,layers=layers,batch_size=batch_size)

optimizer = opt.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
LossFunc = nn.MSELoss()

def train(dataloader_train):
    # checkpoint = torch.load('e9.pt')
    # model.load_state_dict(checkpoint['model.state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer.state_dict'])

    model.train()
    epoch_num =2000
    for epoch in range(epoch_num):
        loss = 0
        train_bar = tqdm(dataloader_train)
        for i, data in enumerate(train_bar):
            x_train, y_train = data

            y_pre = model(x_train)
            # print(y_pre.shape)
            # exit()
            loss = LossFunc(y_train, y_pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss += loss
            train_bar.desc = "train epoch[{}/{}] " \
                             "loss:{:.3f}".format(epoch + 1, epoch_num, loss)
            # scheduler.step()
        print("------", loss / len(dataloader_train))
    torch.save({
        'epoch': epoch,
        'model.state_dict': model.state_dict(),
        'optimizer.state_dict': optimizer.state_dict(),
        'loss': loss},
        'e9.pt')
#
def eval(dataloader_test):
    checkpoint = torch.load('e9.pt')
    model.load_state_dict(checkpoint['model.state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
    model.eval()
    model_pred=[]
    actual=[]

    for i, data in enumerate(dataloader_test):
        x_test, y_test = data

        y_pred = model(x_test)
        # print(".................",y_test.shape,y_pred.shape)
        model_pred.append(y_pred.item())
        actual.append(y_test.item())
    model_pred=np.array(model_pred)
    actual=np.array(actual)

    loss = LossFunc(torch.from_numpy(model_pred), torch.from_numpy(actual))
    print(loss)
    #
    plt.figure(figsize=(10, 6))
    plt.plot(model_pred, 'r')
    plt.plot(actual, 'b')

    plt.show()



def main():
    dataloader_train = DataLoader(training_data, batch_size=16, shuffle=False)
    dataloader_test = DataLoader(testing_data, batch_size=1, shuffle=False)
    # dataloader_test = DataLoader(training_data, batch_size=1, shuffle=False)

    train(dataloader_train)
    # eval(dataloader_test)


if __name__ == "__main__":
    main()
