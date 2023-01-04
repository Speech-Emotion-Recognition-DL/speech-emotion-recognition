import torch
import torch.nn as nn
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import koila
from koila import lazy

from DataManagement import DataManagement, ANNOTATIONS_FILE
from cnn_model import Convolutional_Neural_Network
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from cnn_model_definition import Convolutional_Speaker_Identification
# choose number of epochs higher than reasonable so we can manually stop training
num_epochs = 200
# pick minibatch size (of 32... always)
minibatch = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# instantiate lists to hold scalar performance metrics to plot later
train_losses = []
valid_losses = []
emotions_dict_3 = {
    0: 'positive',
    1: 'neutral',
    2: 'negative',
}


def criterion(predictions, targets):
    """ Loss/Criterion"""
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


# create training loop for one complete epoch (entire training set)
def train(optimizer, model, num_epochs, train_X, train_Y, valid_X, valid_Y, train_size):
    for epoch in range(num_epochs):

        # set model to train mode
        model.train()

        # shuffle entire training set in each epoch to randomize minibatch order
        train_indices = np.random.permutation(train_size)

        # shuffle the training set for each epoch:
        train_X = train_X[train_indices, :, :, :]
        train_Y = train_Y[train_indices]

        # instantiate scalar values to keep track of progress after each epoch so we can stop training when appropriate
        epoch_acc = 0
        epoch_loss = 0
        num_iterations = int(train_size / minibatch)

        # create a loop for each minibatch of 32 samples
        for i in range(num_iterations):
            # we have to track and update minibatch position for the current minibatch
            # if we take a random batch position from a set, we almost certainly will skip some of the data in that set
            # track minibatch position based on iteration number:
            batch_start = i * minibatch

            # ensure we don't go out of the bounds of our training set:
            batch_end = min(batch_start + minibatch, train_size)

            # ensure we don't have an index error
            actual_batch_size = batch_end - batch_start

            # get training minibatch with all channnels and 2D feature dims
            X = train_X[batch_start:batch_end, :, :, :]
            # get training minibatch labels
            Y = train_Y[batch_start:batch_end]

            # instantiate training tensors
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)

            # Pass input tensors thru 1 training step (fwd+backwards pass)
            loss, acc = train_step(X_tensor, Y_tensor)

            # aggregate batch accuracy to measure progress of entire epoch
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size

            # keep track of the iteration to see if the model's too slow
            print('\r' + f'Epoch {epoch}: iteration {i}/{num_iterations}', end='')

        # create tensors from validation set
        X_valid_tensor = torch.tensor(valid_X, device=device).float()
        Y_valid_tensor = torch.tensor(valid_Y, dtype=torch.long, device=device)

        # calculate validation metrics to keep track of progress; don't need predictions now
        valid_loss, valid_acc, _ = validate(X_valid_tensor, Y_valid_tensor)

        # accumulate scalar performance metrics at each epoch to track and plot later
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)

        print(
            f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')


def make_train_step(model, criterion, optimizer):
    # define the training step of the training phase
    def train_step(X, Y):
        # print(X.shape)
        # print(Y.shape)
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))

        # compute loss on logits because nn.CrossEntropyLoss implements log softmax
        loss = criterion(output_logits, Y)

        # compute gradients for the optimizer to use
        loss.backward()

        # update network parameters based on gradient stored (by calling loss.backward())
        optimizer.step()

        # zero out gradients for next pass
        # pytorch accumulates gradients from backwards passes (convenient for RNNs)
        optimizer.zero_grad()

        return loss.item(), accuracy * 100

    return train_step


def validate_fnc(model, criterion):
    def validate(X, Y):
        # don't want to update any network parameters on validation passes: don't need gradient
        # wrap in torch.no_grad to save memory and compute in validation phase:
        with torch.no_grad():
            # set model to validation phase i.e. turn off dropout and batchnorm layers
            model.eval()

            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax, dim=1)

            # calculate the mean accuracy over the entire validation set
            accuracy = torch.sum(Y == predictions) / float(len(Y))

            # compute error from logits (nn.crossentropy implements softmax)
            loss = criterion(output_logits, Y)

        return loss.item(), accuracy * 100, predictions

    return validate


if __name__ == '__main__':
    print()
    dm = DataManagement()
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = dm.get()

    # DNN model
    model = Convolutional_Speaker_Identification().to(device)
    # chosen optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)

    # get training set size to calculate iteration and minibatch indices
    train_size = train_X.shape[0]

    # instantiate the training step function
    train_step = make_train_step(model, criterion, optimizer)

    # instantiate the validation loop function
    validate = validate_fnc(model, criterion)

    # instantiate lists to hold scalar performance metrics to plot later
    train_losses = []
    valid_losses = []

    (train_X, train_Y) = lazy(train_X, train_Y, batch=0)
    (valid_X, valid_Y) = lazy(valid_X, valid_Y, batch=0)

    # train it! - YAY
    train(optimizer, model, num_epochs, train_X, train_Y, valid_X, valid_Y, train_size)

    plt.title('Loss Curve for Convolutional_Neural_Network Model')
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.plot(train_losses[:], 'b')
    plt.plot(valid_losses[:], 'r')
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

    # Save the model
    torch.save(model.cuda().state_dict(), 'model_with_aug1.pt')

    # Convert 4D test feature set array to tensor and move to GPU
    X_test_tensor = torch.tensor(test_X, device=device).float()
    # Convert 4D test label set array to tensor and move to GPU
    y_test_tensor = torch.tensor(test_Y, dtype=torch.long, device=device)
    # Move the model's weights to the specified device
    model.to(device)

    # Reinitialize the validation function with the updated model
    validate = validate_fnc(model, criterion)

    # Get the model's performance metrics using the validation function we defined
    test_loss, test_acc, predicted_emotions = validate(X_test_tensor, y_test_tensor)

    print(f'Test accuracy is {test_acc:.2f}%')

    # because model tested on GPU, move prediction tensor to CPU then convert to array
    predicted_emotions = predicted_emotions.cpu().numpy()
    # use labels from test set
    emotions_groundtruth = test_Y

    # build confusion matrix and normalized confusion matrix
    conf_matrix = confusion_matrix(emotions_groundtruth, predicted_emotions)
    conf_matrix_norm = confusion_matrix(emotions_groundtruth, predicted_emotions, normalize='true')

    # set labels for matrix axes from emotions
    emotion_names = [emotion for emotion in emotions_dict_3.values()]

    # make a confusion matrix with labels using a DataFrame
    confmatrix_df = pd.DataFrame(conf_matrix, index=emotion_names, columns=emotion_names)
    confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=emotion_names, columns=emotion_names)

    # plot confusion matrices
    plt.figure(figsize=(16, 6))
    sn.set(font_scale=1.8)  # emotion label and title size
    plt.subplot(1, 2, 1)
    plt.title('Confusion Matrix')
    sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18})  # annot_kws is value font
    plt.subplot(1, 2, 2)
    plt.title('Normalized Confusion Matrix')
    sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13})  # annot_kws is value font

    plt.show()

    # Save train_X
    torch.save(train_X, 'train_X_with_aug1.pt')

    # Save train_Y
    torch.save(train_Y, 'train_Y_with_aug1.pt')

    # Save valid_X
    torch.save(valid_X, 'valid_X_with_aug1.pt')

    # Save valid_Y
    torch.save(valid_Y, 'valid_Y_with_aug1.pt')

    # Save test_X
    torch.save(test_X, 'test_X_with_aug1.pt')

    # Save test_Y
    torch.save(test_Y, 'test_Y_with_aug1.pt')

    """ open train model and check for the prediction  """
    #
    #
    # # Load the model's weights and parameters from the file
    # state_dict = torch.load('model.pt')
    #
    # # Create a new model object
    # model = Convolutional_Speaker_Identification()
    #
    # # Load the weights and parameters into the model
    # model.load_state_dict(state_dict)
    #
    #
    # # Load train_X
    # train_X = torch.load('train_X.pt')
    #
    # # Load train_Y
    # train_Y = torch.load('train_Y.pt')
    #
    # # Load valid_X
    # valid_X = torch.load('valid_X.pt')
    #
    # # Load valid_Y
    # valid_Y = torch.load('valid_Y.pt')
    #
    # # Load test_X
    # test_X = torch.load('test_X.pt')
    #
    # # Load test_Y
    # test_Y = torch.load('test_Y.pt')
    #
    # #Convert 4D test feature set array to tensor and move to GPU
    # X_test_tensor = torch.tensor(test_X, device=device).float()
    # # Convert 4D test label set array to tensor and move to GPU
    # y_test_tensor = torch.tensor(test_Y, dtype=torch.long, device=device)
    # # Move the model's weights to the specified device
    # model.to(device)
    #
    # # Reinitialize the validation function with the updated model
    # validate = validate_fnc(model, criterion)
    #
    # # Get the model's performance metrics using the validation function we defined
    # test_loss, test_acc, predicted_emotions = validate(X_test_tensor, y_test_tensor)
    #
    # print(f'Test accuracy is {test_acc:.2f}%')
    #
    #
    #
    # # because model tested on GPU, move prediction tensor to CPU then convert to array
    # predicted_emotions = predicted_emotions.cpu().numpy()
    # # use labels from test set
    # emotions_groundtruth = test_Y
    #
    # # build confusion matrix and normalized confusion matrix
    # conf_matrix = confusion_matrix(emotions_groundtruth, predicted_emotions)
    # conf_matrix_norm = confusion_matrix(emotions_groundtruth, predicted_emotions, normalize='true')
    #
    # # set labels for matrix axes from emotions
    # emotion_names = [emotion for emotion in emotions_dict_3.values()]
    #
    # # make a confusion matrix with labels using a DataFrame
    # confmatrix_df = pd.DataFrame(conf_matrix, index=emotion_names, columns=emotion_names)
    # confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=emotion_names, columns=emotion_names)
    #
    # # plot confusion matrices
    # plt.figure(figsize=(16, 6))
    # sn.set(font_scale=1.8)  # emotion label and title size
    # plt.subplot(1, 2, 1)
    # plt.title('Confusion Matrix')
    # sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18})  # annot_kws is value font
    # plt.subplot(1, 2, 2)
    # plt.title('Normalized Confusion Matrix')
    # sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13})  # annot_kws is value font
    #
    # plt.show()
