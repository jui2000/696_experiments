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
        total_loss = []
        total_acc = []

        for i in train_loader:
            data, target = i
            #FORWARD PASS
            output = model(data.float())
            loss = criterion(output, target.unsqueeze(1)) 
            total_loss.append(loss.cpu().detach())

            #BACKWARD AND OPTIMIZE
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # PREDICTIONS 
            pred = np.round(output.cpu().detach().numpy())
            target = np.round(target.cpu().detach().numpy())             
            y_pred.extend(pred.tolist())
            y_true.extend(target.tolist())

            total_acc.append(accuracy_score(y_true,y_pred))


        val_acc_epochs.append(test(model, valloader))
        loss_all_epochs.append(np.mean(total_loss))
        tr_acc_all_epochs.append(np.mean(total_acc))
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
            pred = np.round(output.cpu().numpy())
            target = target.cpu().numpy()
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
    
    return accuracy_score(y_true,y_pred)
    