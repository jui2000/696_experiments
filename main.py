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

######################### Hyper-parameters #########################
input_size = 2
hidden_size = [40,20]
out_size = 1 
num_epochs = 1000
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128


########### Loading the dataset #############################
data_normal = pd.read_csv("data_normal.csv")

X_train, X_test, y_train, y_test = train_test_split(np.array(data_normal[["x_1", "x_2"]]), 
                           np.array(data_normal["y"]), test_size=0.3)


trainset = dataset(torch.tensor(X_train,dtype=torch.float32).to(device), \
					torch.tensor(y_train,dtype=torch.float32).to(device))
testset = dataset(torch.tensor(X_test,dtype=torch.float32).to(device), \
					torch.tensor(y_test,dtype=torch.float32).to(device))

#DataLoader
trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
valloader = DataLoader(testset,batch_size=batch_size,shuffle=True)

# model definition
model = LinearModel(input_size, hidden_size, out_size).to(device)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model, tr_loss, tr_acc, val_acc = train(model, trainloader, valloader, \
						optimizer, num_epochs, criterion)

loss_visualize(tr_loss, "Loss vs iteration")
acc_visualize([tr_acc, val_acc], \
				["training accuracy", "validation accuracy"], \
				"Accuracy vs iteration")


