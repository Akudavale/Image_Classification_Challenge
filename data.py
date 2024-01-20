
import torch
import torch.nn as nn
from torch.utils.data import Dataset,  DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import natsort
from model import Network

class ChristmasImages(Dataset):
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path
        
        #For training data
        self.transform1 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        #For validation data
        self.transform2 = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean= ([0.485, 0.456, 0.406]),std = ([0.229, 0.224, 0.225]))
            ])
        
        if self.training == True:
            self.dataset = ImageFolder(root=self.path.joinpath('train'),transform=self.transform1)
        else:
            self.path = path
            self.sorted_images = natsort.natsorted(os.listdir(self.path))          
            
    def __len__(self):
        if self.training:
            return len(self.dataset)
        else: 
            return len(self.sorted_images)

    def __getitem__(self, index):       
        if self.training == True:
            image, label = self.dataset[index]
            return image, label
        else:           
            img = os.path.join(self.path,self.sorted_images[index])            
            image = self.transform2(Image.open(img).convert("RGB")) 
            return image


def model_test(model, test_ds):
    model.eval()  

    predictions = []
    idx = []

    with torch.no_grad():
        for i, X in enumerate(test_ds, 0):
            X= X.unsqueeze(0)
            pred = model(X)
            _, predicted  = torch.max(pred, 1)
            predictions.extend(predicted.cpu().numpy())
            idx.append(i)

    return (idx,predictions)


def model_train(model, epochs, optimizer, scheduler, train_loader, loss_fn):

    model.train()

    for epoch in range(epochs):
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_loader):
            print(batch)
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        loss = train_loss / len(train_loader)
        acc = train_acc / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}  || Loss: {loss:.3f}  || Accuracy: {acc:.3f}")

        scheduler.step() 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Load the data and trasform
    path = Path("C:\\Users\\HP\\Desktop\\SEM3\\Deep-Learning\\Winter Task\\task\\data")
    test_path = Path("C:\\Users\\HP\\Desktop\\SEM3\\Deep-Learning\\Winter Task\\task\\data\\val")
    train_ds = ChristmasImages(path, training=True)
    test_ds = ChristmasImages(test_path, training=False)


    # divide data in bathes through Dataloder
    train_loader = DataLoader(dataset=train_ds, batch_size=32,shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    #print("Data loded")

    # Define Architecture
    model = Network()
    model.to(device)
    #print("model loded")

    # Define Loss functions
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    #print('training started')
    model_train(model=model, epochs=12, optimizer=optimizer,scheduler=scheduler,train_loader=train_loader,loss_fn=loss_fn)
    model.save_model()

    #loadmodel
    test_model = torch.load("model.pkl",map_location=torch.device('cpu'))
    #print("model loaded")
    
    #Predictions
    result = model_test(model=test_model, dataset=test_ds)

    #print("obtained reslts")
    #save results
    results_df = pd.DataFrame({"Id": result[0], "Category": result[1]})
    results_df.to_csv("test_results.csv", index=False)

    print("results saved to CSV")