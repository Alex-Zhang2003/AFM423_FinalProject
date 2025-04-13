import os
import numpy as np
from datetime import datetime
import torch
import pickle
from utils import save_model


MODEL_DIR = r'./trained_models'


def batch_train(model_name, model, criterion, optimizer, train_loader, val_loader, epochs):
    training_info = {
        'train_loss_hist': [],
        'val_loss_hist': [],
        'train_acc_hist': [],
        'val_acc_hist': []
    }

    for epoch in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        train_acc = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(model.device, dtype=torch.float32), targets.to(model.device, dtype=torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            tmp_acc = torch.count_nonzero(torch.argmax(outputs, dim = 1) == targets).item()/targets.size(0)
            train_acc.append(tmp_acc)
        # Get train loss and test loss
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)

        model.eval()
        val_loss = []
        val_acc = []
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(model.device, dtype=torch.float), targets.to(model.device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
            tmp_acc = torch.count_nonzero(torch.argmax(outputs, dim=1) == targets).item() / targets.size(0)
            val_acc.append(tmp_acc)
        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)

        # Save losses
        training_info['train_loss_hist'].append(train_loss)
        training_info['val_loss_hist'].append(val_loss)
        training_info['train_acc_hist'].append(train_acc)
        training_info['val_acc_hist'].append(val_acc)

        dt = datetime.now() - t0
        print(f'''
        Epoch {epoch + 1}/{epochs},
        Train Loss: {train_loss:.4f}, Train Acc: {train_acc: .4f},
        Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc: .4f},
        Duration: {dt}''')

    save_model(model, os.path.join(MODEL_DIR, f'{model_name}.pth'))
    print('model saved')

    with open(os.path.join(MODEL_DIR, f'{model_name}_training_process.pkl'), 'wb') as f:
        pickle.dump(training_info, f)
