import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train(model, train_loader, valloader, optimizer, num_epochs, criterion):
    model.train()
    y_true = []
    y_pred = []

    loss_all_epochs = []
    tr_acc_all_epochs = []
    val_acc_epochs = []
    for epoch_ in tqdm(range(num_epochs)):
        total_loss = 0
        total_acc = 0

        counter = 0
        for i in train_loader:
            counter += 1

            data, target = i

            #FORWARD PASS
            output = model(data.float())
            loss = criterion(output, target.unsqueeze(1)) 
            total_loss += loss.detach()

            #BACKWARD AND OPTIMIZE
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # PREDICTIONS 
            pred = np.round(output.detach())
            target = np.round(target.detach())             
            y_pred.extend(pred.tolist())
            y_true.extend(target.tolist())

            total_acc += accuracy_score(y_true,y_pred)


        val_acc_epochs.append(test(model, valloader))
        loss_all_epochs.append(total_loss/counter)
        tr_acc_all_epochs.append(total_acc/counter)
    return model, loss_all_epochs, tr_acc_all_epochs, val_acc_epochs


#TESTING THE MODEL
def test(model, test_loader):
    #model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []
    
    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i in test_loader:
            
            #LOAD THE DATA IN A BATCH
            data,target = i
            
            
            # the model on the data
            output = model(data.float())
                       
            #PREDICTIONS
            pred = np.round(output)
            target = target.float()
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
    
    return accuracy_score(y_true,y_pred)
    