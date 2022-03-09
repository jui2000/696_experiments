import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train(model, train_loader, val_loader, optimizer, num_epochs, criterion):
    model.train()
    y_true = []
    y_pred = []

    loss_all_epochs = []
    tr_acc_all_epochs = []
    val_acc_all_epochs = []
    for epoch_ in tqdm(range(num_epochs)):
        total_loss = []

        for i in train_loader:
            data, target = i

            # zero the parameter gradients
            optimizer.zero_grad()

            #FORWARD PASS
            output = model(data)
            loss = criterion(output, target.unsqueeze(1)) 
            #BACKWARD AND OPTIMIZE
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        # PREDICTIONS 
        val_acc_all_epochs.append(test(model, val_loader))
        tr_acc_all_epochs.append(test(model, train_loader))
        
        # save the losses
        loss_all_epochs += total_loss

    return model, loss_all_epochs, tr_acc_all_epochs, val_acc_all_epochs


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
    
