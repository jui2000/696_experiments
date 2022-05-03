import torch
import torch.nn as nn
import torchvision.transforms as tt
from torchvision.datasets import EMNIST
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import LinearModel
from runner import train, test
from viz import loss_visualize, acc_visualize
import sys

trainset = EMNIST(root="data/", split="byclass", download=True, train=True, 
                transform=tt.Compose([
                    lambda img: tt.functional.rotate(img, -90),
                    lambda img: tt.functional.hflip(img),
                    tt.ToTensor()
                ]))

testset = EMNIST(root="data/", split="byclass", download=True, train=False, 
                transform=tt.Compose([
                    lambda img: tt.functional.rotate(img, -90),
                    lambda img: tt.functional.hflip(img),
                    tt.ToTensor()
                ]))
######################### Hyper-parameters #########################
hidden_size = [int(sys.argv[1]), int(sys.argv[2])] #[256,128] #[40,20]
out_size = 62 
num_epochs = int(sys.argv[3])
learning_rate = float(sys.argv[4])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = int(sys.argv[5])


# modify this variable as required
with_clusters = bool(int(sys.argv[6]))
num_clusters = 2 if with_clusters else 0  #never used in this code

if with_clusters:
    input_size = 28*28 + 1
else:
    input_size = 28*28


#DataLoader
trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
valloader = DataLoader(testset,batch_size=batch_size,shuffle=True)

# model definition
model = LinearModel(input_size, hidden_size, out_size, with_clusters = with_clusters, num_clusters = num_clusters).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model, tr_loss, tr_acc, val_acc = train(model, trainloader, valloader, \
						optimizer, num_epochs, criterion, with_clusters = with_clusters)
torch.save(model.state_dict(), './models/' + str(with_clusters) +str(num_epochs) + '_' + str(hidden_size) + str(learning_rate) + '_' + str(batch_size) + '.pt' )

loss_visualize(tr_loss, str(num_epochs) + '_' + str(hidden_size) + str(learning_rate) + '_' + str(batch_size), "Loss vs iteration", with_clusters = with_clusters)
acc_visualize([tr_acc, val_acc], str(num_epochs) + '_' + str(hidden_size) + str(learning_rate) + '_' + str(batch_size) , \
				["training accuracy", "validation accuracy"], \
				"Accuracy vs epochs", with_clusters = with_clusters)

best_val_acc = max(val_acc)
print("val_accuracy" + str(best_val_acc))
print("train_acc" + str(max(tr_acc)))


# model.load_state_dict(torch.load( './models/' + str(with_clusters)  + str(num_epochs) + '_' + str(hidden_size) + str(learning_rate) + '_' + str(batch_size) + '.pt' ))
# print(test(model, valloader))