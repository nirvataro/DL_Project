import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam





# Auxiliary Functions:
# function which calculate the accuracy of the model and return it
def calculate_acc(dataset, dataset_iter, model):
    n_correct, n_total = 0, 0
    comment_context, un_context = model.init_context()
    for batch_data in dataset_iter:
        if batch_data.c.shape[1] != dataset.batch_size:
            break
        with torch.no_grad():
            pred = model(batch_data, comment_context.to(device), un_context.to(device))
            # calculate output and get the prediction
            pred = torch.squeeze(pred)
            predictions = torch.argmax(pred, dim=1)
            n_correct += torch.sum(predictions == batch_data.y).type(torch.float32)
            n_total += batch_data.c.shape[1]
    acc = (n_correct / n_total).item()
    return acc


# function which plot the figure we need for each exercise, test accuracy will print only if we need
def plot_results(x, data, labels, test_acc=None):
    for d in data:
        plt.plot(list(range(x)), d)
    plt.legend(labels)
    plt.show()
    if test_acc:
        print('Test Accuracy: {}'.format(test_acc))


def train(dataset, model, name, device, max_epochs=1000):
    best, best_epoch, patience, clip_grad = np.inf, 0, 10, 1
    training_acc, validation_acc, training_loss, validation_loss = [], [], [], []

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # lr decrease

    for epoch in range(max_epochs):
        for state in ['train', 'validation']:
            if state == 'train':
                iter = dataset.train_iter
                model.train()
                optimizer.zero_grad()
            else:
                iter = dataset.valid_iter
                model.eval()
            epoch_losses = []
            comment_context, un_context = model.init_context()

            for i, batch in enumerate(iter):
                if batch.c.shape[1] != dataset.batch_size:
                    break
                print(i, '/', len(iter))
                y_pred = model(batch, comment_context, un_context)
                loss = criterion(y_pred, batch.y.to(device))
                if state == 'train':
                    loss.backward()
                    if clip_grad > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                epoch_losses.append(loss.detach())
            if state == 'train':
                scheduler.step()
                train_epoch_loss = torch.mean(torch.tensor(epoch_losses))
                training_loss.append(train_epoch_loss)
            else:
                val_epoch_loss = torch.mean(torch.tensor(epoch_losses))
                validation_loss.append(val_epoch_loss)

        # calculate training and validation accuracy
        val_acc = calculate_acc(dataset, dataset.valid_iter, model)
        train_acc = calculate_acc(dataset, dataset.train_iter, model)
        validation_acc.append(val_acc)
        training_acc.append(train_acc)

        print(f"Epoch: {epoch}")
        print("Validation Acc: {:.4f}\t\tValidation Loss: {:.4f}".format(val_acc, val_epoch_loss))
        print("Training Acc:   {:.4f}\t\tTraining Loss:   {:.4f}".format(train_acc, train_epoch_loss))

        # check for early stopping point
        if val_epoch_loss < best:
            torch.save(model.state_dict(), name)
            best = val_epoch_loss
            best_epoch = epoch
        print("Epochs since seen best: ", str(epoch - best_epoch), "\n")
        if epoch - best_epoch >= patience:
            model = model.to(device)
            model.load_state_dict(torch.load(name))
            model.eval()
            break

    # plot training, validation losses
    loss_data = [training_loss, validation_loss]
    labels = ["Training Loss", "Validation Loss"]
    plot_results(epoch + 1, loss_data, labels)

    # plot training, validation accuracy
    data = [training_acc, validation_acc]
    labels = ["Training Accuracy", "Validation Accuracy"]
    plot_results(epoch + 1, data, labels, calculate_acc(dataset, dataset.test_iter, model))
