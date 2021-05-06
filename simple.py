from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
# %matplotlib inline
# import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch.utils.data.sampler import SubsetRandomSampler
from topology import Autopo
import numpy as np
import math
import csv
from scipy import stats

dataset = Autopo('./tmp')
#loader = DataLoader(dataset, batch_size=32, shuffle=True)


batch_size = 32
n_epoch=15
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 1-train_ratio-val_ratio

shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))

if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

n_train=int(dataset_size*train_ratio)
n_val=int(dataset_size*val_ratio)

train_indices, val_indices, test_indices = indices[:n_train], indices[n_train+1:n_train+n_val], indices[n_train+n_val+1:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        
        Din1=dataset.num_node_features
        Dout1=64
        Dout2=128
        Dout3=256
        Dout4=64
        Dedge1=Din1*Dout1
        Dedge2=Dout1*Dout2
        Dedge3=Dout2*Dout3
        Dedge4=Dout3*Dout4

        num_edge_features=3
        num_out=1

        self.nn1_a = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge1))

        self.nn1_b = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge2))

        self.nn1_c = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge3))

        self.nn1_d = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge4))


        self.conv1_a = NNConv(Din1, Dout1,self.nn1_a)
        self.conv1_b = NNConv(Dout1,Dout2,self.nn1_b)
        self.conv1_c = NNConv(Dout2,Dout3,self.nn1_c)
        self.conv1_d = NNConv(Dout3,Dout4,self.nn1_d)

        self.nn2_a = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge1))

        self.nn2_b = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge2))

        self.nn2_c = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge3))

        self.nn2_d = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,Dedge4))


        self.conv2_a = NNConv(Din1, Dout1,self.nn2_a)
        self.conv2_b = NNConv(Dout1,Dout2,self.nn2_b)
        self.conv2_c = NNConv(Dout2,Dout3,self.nn2_c)
        self.conv2_d = NNConv(Dout3,Dout4,self.nn2_d)


        self.lin1 = torch.nn.Linear(Dout4, 128)
        self.lin2 = torch.nn.Linear(128,64)
        self.lin3 = torch.nn.Linear(64,64)

        self.output=torch.nn.Linear(3*64, num_out)
 

    def forward(self, data):
        x, edge_index,edge_attr1,edge_attr2,batch = data.x, data.edge_index.long(), data.edge_attr1, data.edge_attr2, data.batch
        
        output_ind=self.output_indices(batch)
        input_ind=self.input_indices(batch)
        gnd_ind=self.gnd_indices(batch)


        x1 = self.conv1_a(x, edge_index, edge_attr1+edge_attr2)
        x1 = F.relu(x1)
        x1 = self.conv1_b(x1, edge_index, edge_attr1+edge_attr2)
        x1 = F.relu(x1)
        x1 = self.conv1_c(x1, edge_index, edge_attr1+edge_attr2)
        x1 = F.relu(x1)
        x1 = self.conv1_d(x1, edge_index, edge_attr1+edge_attr2)
        x1 = F.relu(x1)


        x3 = self.lin1(x1)
#        x3 = F.relu(x3)
        x3 = self.lin2(x3)
#        x3 = F.relu(x3)
        x3 = self.lin3(x3)
        x3=torch.cat((x3[input_ind],x3[output_ind],x3[gnd_ind]),1)
#        print(x3.shape)
#        x3= self.lin4(x3)
#        x3= F.relu(x3)
        x3 =self.output(x3)
      
        return x3

    def output_indices(self, batch):
        num_element=len(batch)
        output_ind=[]
        count=0
        previous_num=torch.tensor(0,dtype=int).to(device)
        current_num=torch.tensor(-1,dtype=int).to(device)
 
        for id,item in enumerate(batch):
#            print(id,item)
            if not torch.equal(item,current_num):
                count=0
            current_num=item
            count=count+1
            if torch.equal(current_num,previous_num) and count==2:
               output_ind.append(id)
               previous_num=previous_num+1
#        print(output_ind)
        return output_ind

    def input_indices(self, batch):
        num_element=len(batch)
        output_ind=[]
        count=0
        previous_num=torch.tensor(0,dtype=int).to(device)
        current_num=torch.tensor(-1,dtype=int).to(device)
 
        for id,item in enumerate(batch):
#            print(id,item)
            if not torch.equal(item,current_num):
                count=0
            current_num=item
            count=count+1
            if torch.equal(current_num,previous_num) and count==1:
               output_ind.append(id)
               previous_num=previous_num+1
#        print(output_ind)
        return output_ind

    def gnd_indices(self, batch):
        num_element=len(batch)
        gnd_ind=[]
        count=0
        previous_num=torch.tensor(0,dtype=int).to(device)
        current_num=torch.tensor(-1,dtype=int).to(device)
 
        for id,item in enumerate(batch):
#            print(id,item)
            if not torch.equal(item,current_num):
                count=0
            current_num=item
            count=count+1
            if torch.equal(current_num,previous_num) and count==3:
               gnd_ind.append(id)
               previous_num=previous_num+1
#        print(output_ind)
        return gnd_ind



def rse(y,yt):

    assert(y.shape==yt.shape)

    var=0
    m_yt=yt.mean()
    for i in range(len(yt)):
        var+=(yt[i]-m_yt)**2 

    mse=0
    for i in range(len(yt)):
        mse+=(y[i]-yt[i])**2

    rse=mse/var

    rmse=math.sqrt(mse/len(yt))

    print(rmse)

    return rse



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

criterion = MSELoss(reduction='mean').to(device)

train_perform=[]
val_perform=[]

loss=0
for epoch in range(n_epoch):

########### Training #################
    
    train_loss=0
    n_batch_train=0

    model.train()

    print("finished training")
#    print(len(train_loader))
    for i,data in enumerate(train_loader):
         n_batch_train=n_batch_train+1
         data.to(device)
         optimizer.zero_grad()
         out=model(data)
         out=out.reshape(data.y.shape)
         assert(out.shape == data.y.shape)
         loss=F.mse_loss(out, data.y.float())
         loss.backward()
         optimizer.step()

         train_loss += out.shape[0] * loss.item()

    if epoch % 1 == 0:
         print('%d epoch training loss: %.3f' %
                  (epoch, train_loss/n_batch_train/batch_size))

         train_perform.append(train_loss/n_batch_train/batch_size)

############## Evaluation ######################

    n_batch_val=0
    val_loss=0
 
    if epoch % 1 == 0:

         model.eval()
         for data in val_loader:
              n_batch_val=n_batch_val+1
              data.to(device)
              out=model(data)
              out = out.reshape(data.y.shape)
              assert(out.shape == data.y.shape)
              loss = F.mse_loss(out, data.y.float())
              val_loss += out.shape[0] * loss.item()

         val_perform.append(val_loss/n_batch_val/batch_size)
#         print("val loss: ",val_loss/n_batch_val/batch_size )
         n_batch_val=0
 

############# Test ################################

model.eval()

accuracy=0
n_batch_test=0
gold_list=[]
out_list=[]
for data in test_loader:
              n_batch_test+=1
              data.to(device)
              out=model(data).cpu().detach().numpy()
              gold=data.y.cpu().numpy()
              gold_list.extend(gold)
              out_list.extend(out)
              out=out.reshape(-1)
              out=np.array([x for x in out])
              gold=gold.reshape(-1)
              gold=np.array([x for x in gold])
              L=len(gold)
              rse_result=rse(out,gold)
              np.set_printoptions(precision=2,suppress=True)
#              print("RSE: ",rse_result)
#              print("Truth:   ",gold.reshape([L]))
#              print("Predict: ",out.reshape([L]))
print("Final RSE:",rse(np.reshape(out_list,-1),np.reshape(gold_list,-1)))
print(stats.ttest_ind(np.reshape(out_list,-1),np.reshape(gold_list,-1)))

#print((np.reshape(gold_list,-1)))
#print((np.reshape(out_list,-1)))
#np.set_printoptions(precision=2,suppress=True) 
#print("train_history: ",train_perform)
#print("val_history: ",val_perform)




