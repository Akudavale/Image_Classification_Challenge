import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import pandas as pd
from pathlib import Path
from model import Network
from data import ChristmasImages
import logging

# Configure logging to write to a file
logging.basicConfig(filename='output.log', level=logging.INFO)

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
            #print(batch)
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

        logging.info(f"Epoch {epoch+1}/{epochs}  || Loss: {loss:.3f}  || Accuracy: {acc:.3f}")

        scheduler.step()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(device)
    torch.cuda.empty_cache()

    # Load the data and trasform
    path = Path("/home/g062386/DEEP Learning winter task/data")
    test_path = Path("/home/g062386/DEEP Learning winter task/data/val")
    train_ds = ChristmasImages(path, training=True)
    test_ds = ChristmasImages(test_path, training=False)

    # divide data in bathes through Dataloder
    train_loader = DataLoader(train_ds, batch_size=32,shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
   
    # Define Architecture
    model = Network()
    #model.load_state_dict(torch.load("2model.pkl"))
    model.to(device)
   
    # Define Loss functions
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.05)

    model_train(model=model, epochs=30, optimizer=optimizer,scheduler=scheduler,train_loader=train_loader,loss_fn=loss_fn)
    model.save_model()

    #loadmodel
    model.load_state_dict(torch.load("model.pkl", map_location=device))

    #Predictions
    result = model_test(model=model, test_ds=test_ds)

    #save results
    results_df = pd.DataFrame({"Id": result[0], "Category": result[1]})
    results_df.to_csv("test_results.csv", index=False)