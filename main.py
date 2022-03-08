import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from src.model import LinearModel
from src.data import dataset
from src.runner import train, test
from src.viz import loss_visualize, acc_visualize


########### Loading the dataset #############################
data_normal = pd.read_csv("data_normal.csv")
data_normal.columns
data_normal.sample(10)

X_train, X_test, y_train, y_test = train_test_split(np.array(data_normal[["x_1", "x_2"]]), 
                           np.array(data_normal["y"]), test_size=0.3)

trainset = dataset(X_train, y_train)
testset = dataset(X_test, y_test)

#DataLoader
trainloader = DataLoader(trainset,batch_size=64,shuffle=True)
testloader = DataLoader(testset,batch_size=64,shuffle=True)


######################### Hyper-parameters #########################
input_size = 2
hidden_size = 32
out_size = 1 
num_epochs = 100
learning_rate = 0.1
# BATCH_SIZE_1 = 

model = LinearModel(input_size, hidden_size, out_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model, total_loss, total_acc = train(model, trainloader, optimizer, num_epochs, criterion)

loss_visualize(train_loss)
acc_visualize([total_acc], ["training accuracy"])

# test(model, testloader)

