import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, optimizer, num_epochs, criterion, with_clusters):
    model.train()

    loss_all_epochs = []
    tr_acc_all_epochs = []
    val_acc_all_epochs = []
    for epoch_ in (range(num_epochs)):
        total_loss = []

        for i in (train_loader):
            
            data, target = i
            
            shape_0, _,shape_1,shape_2 = data.shape
            data = data.reshape(shape_0, shape_1*shape_2)
            
            if with_clusters:
                l = target.detach().tolist()
                
                for item_i in range(len(l)):
                    l[item_i] = 0 if l[item_i] in [0,1,2,3,4,5,6,7,8,9] else 1

                data = torch.cat((data,torch.FloatTensor(l).unsqueeze(1)),1)
            
            # print(torch.FloatTensor(l).shape)
            # plt.scatter(data[target==0,0].cpu(), data[target==0,1].cpu())
            # plt.scatter(data[target==1,0].cpu(), data[target==1,1].cpu())
            # plt.show()

            # zero the parameter gradients
            optimizer.zero_grad()
            #FORWARD PASS
            output = model(data.to(device))
            # output = output.type(torch.int64)
            # target = target.type(torch.int64)
            # print(output.shape)
            # print(target.shape)
            # print((output).dtype)
            # print((target).dtype)
            # exit()
            loss = criterion(output, target.to(device)) 
            #BACKWARD AND OPTIMIZE
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        
      
        if (epoch_ % 1) == 0:
            print(epoch_)
            print("Loss: ", np.mean(total_loss),flush=True)
            # print(test(model, train_loader, with_clusters), flush=True)
            # print(test(model, val_loader, with_clusters), flush=True)

        # PREDICTIONS 
        val_acc_all_epochs.append(test(model, val_loader, with_clusters))
        tr_acc_all_epochs.append(test(model, train_loader, with_clusters))
        
        # save the losses
        loss_all_epochs += total_loss

    return model, loss_all_epochs, tr_acc_all_epochs, val_acc_all_epochs


#TESTING THE MODEL
def test(model, test_loader, with_clusters):
    #model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []
    
    softmax = nn.Softmax(dim=1)

    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i in test_loader:
            #LOAD THE DATA IN A BATCH
            data,target = i
            
            shape_0, _,shape_1,shape_2 = data.shape
            data = data.reshape(shape_0, shape_1*shape_2)

            if with_clusters:
                l = target.detach().tolist()
                
                for item_i in range(len(l)):
                    l[item_i] = 0 if l[item_i] in [0,1,2,3,4,5,6,7,8,9] else 1

                data = torch.cat((data,torch.FloatTensor(l).unsqueeze(1)),1)

            # the model on the data
            output = model(data.to(device))
            output = softmax(output)
            _, output = torch.max(output, 1)
                       
            #PREDICTIONS
            pred = np.round(output.cpu().numpy())
            target = target.cpu().numpy()
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
    
    return accuracy_score(y_true,y_pred)
    
