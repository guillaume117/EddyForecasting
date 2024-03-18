
import torch
from torch.utils.data import DataLoader
import time
import copy


 
  
class TrainDataset:
    """
    This class is used to train the model.
    It is used to train the model on the train_dataset and validate it on the val_dataset.
    input : model : tensor : the model to train
            criterion : tensor : the criterion to use
            optimizer : tensor : the optimizer to use
            scheduler : tensor : the scheduler to use
            num_epochs : int : the number of epochs to train the model
            learning_rate : float : the learning rate to use
            batch_size : int : the batch size to use
            device : str : the device to use
            nan_mask_label : tensor : the mask to use
            running_instance : str : the name of the running instance
    output : model : tensor : the trained model
    """
    def __init__(self,model,criterion,
                 optimizer,scheduler,num_epochs,learning_rate,batch_size,device,nan_mask_label,running_instance,num_dates):
        self.device = device
        self.model = model
        self.criterion=criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.learning_rate =learning_rate
        self.nan_mask_label = nan_mask_label
        self.running_instance = running_instance
        self.num_dates = num_dates





    def train_model(self,train_dataset,val_dataset):
        """
        This function is used to train the model.
        input : train_dataset : tensor : the train dataset
                val_dataset : tensor : the validation dataset
        output : model : tensor : the trained model
        """
      
        val_acc = []
        val_loss = []
        train_acc = []
        train_loss = []
        epoch=0

        train_loader  = DataLoader(dataset = train_dataset, batch_size=self.batch_size, shuffle =True,drop_last=True)
        val_loader = DataLoader(dataset = val_dataset, batch_size=self.batch_size,shuffle=True, drop_last=True)

        dataloaders = {'train': train_loader, 'val': val_loader}
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0
        list = {'train': {'acc': train_acc, 'loss': train_loss}, 
            'val':{'acc': val_acc, 'loss': val_loss}}
       

        
        for epoch in range(self.num_epochs):

            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 100)

            for phase in ['train','val']:
                if phase =='train':
                    self.model.train()
                else:
                    self.model.eval()
                running_corrects = 0.0
                batch_number = 0
                batch_acc_single_element = 0
                for inputs, labels in dataloaders[phase]:
                    batch_number+=1

                    if self.device.type == 'mps':
                    
                   

                        inputs = inputs.to(self.device).float()
                        labels = labels.to(self.device).float()
                    else :
                        #if the device is cuda or cpu, we need to convert the inputs and labels to the device
                        print("cuda or cpu")
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        

                    self.optimizer.zero_grad()
                    #forward    with torch.no_grad() if phase == 'val' else torch.set_grad_enabled(phase == 'train'):
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs[~self.nan_mask_label],labels[~self.nan_mask_label])
                
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        if (batch_number%10==0 )|(phase == 'val')|(batch_number ==1):
                            batch_acc_single_element +=1
                            #reshaping the output to get the max indices and the one hot encoding over the 3 channels of the output
                            reshaped_output = outputs.view(outputs.size(0), 3, self.num_dates, outputs.size(2), outputs.size(3))
                                    
                            max_indices = torch.argmax(reshaped_output, dim=1)
                            one_hot_encoding = torch.zeros_like(reshaped_output)
                            preds = one_hot_encoding.scatter_(1, max_indices.unsqueeze(1), 1)
                            preds = preds.view_as(outputs)
                            batch_accuracy =torch.sum(torch.where((labels==preds)&(labels>0.5),torch.tensor(1).to(self.device),torch.tensor(0).to(self.device))[~self.nan_mask_label])
                            batch_card = torch.sum(labels[~self.nan_mask_label])
                            accuracy = batch_accuracy*100/(batch_card)
                            print(f"batch_loss {batch_number}_Epoch_{epoch}={loss.item():.2f}, accuracy = {accuracy:.2f}%")
                            running_corrects +=batch_accuracy
                epoch_acc =100*running_corrects/(batch_acc_single_element*batch_card)
                print(f"EPOCH ACCURACY = {epoch_acc:.2f} %")
                        
                            
                
                
                list[phase]['loss'].append(loss)
                list[phase]['acc'].append(epoch_acc)
                print("*"*100)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss.item(), epoch_acc))
                print("*"*100)
        
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), f'UNet_trained/UNet_{self.running_instance}_Epoch_{epoch}_valacc_{epoch_acc}.pth')
            
            print()
            
        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        
            
        return self.model







