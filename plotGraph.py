import scipy
from glob import glob
import matplotlib.pyplot as plt
import sys
#from data_loader import DataLoader
import numpy as np
import os
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle


PLOT_BASE_PATH = "C:/Users/Meghana/Downloads/History_Files"

def plot_loss(model_name, model_title, epochs, training_loss, validation_loss):
    
    # "Loss"
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title('Model %s' % (model_title))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig('%s/model_%s.png' %(PLOT_BASE_PATH,model_name))

if __name__ == "__main__":
    print("Hello")

    path = glob('%s/*.pkl' % (PLOT_BASE_PATH))
    print(path)

    # SRCNN PL History
    pickle_in = open(path[0],"rb")
    example_dict = pickle.load(pickle_in)
    train_loss = np.array(example_dict['train']['generator'])
    valid_loss = np.array(example_dict['test']['generator'])
    assert train_loss.shape == valid_loss.shape
    epochs = np.arange(0, train_loss.shape[0])
    plot_loss("srcnn_mse", "SRCNN with MSE Loss", epochs, train_loss, valid_loss)
    pickle_in.close()

    # SRCNN PL History
    # pickle_in = open(path[1],"rb")
    # example_dict = pickle.load(pickle_in)
    # train_loss = np.array(example_dict['train']['generator'])
    # valid_loss = np.array(example_dict['test']['generator'])
    # assert train_loss.shape == valid_loss.shape
    # epochs = np.arange(0, train_loss.shape[0])
    # plot_loss("srcnn_pl", "SRCNN with Perceptual Loss", epochs, train_loss, valid_loss)
    # pickle_in.close()

    # SRGAN MSE History
    # pickle_in = open(path[2],"rb")
    # srgan_mse_dict = pickle.load(pickle_in)
    # srgan_mse_train_generator = srgan_mse_dict['train']['generator']
    # train_loss = np.array([srgan_mse_train_generator[i][0] for i in range(len(srgan_mse_train_generator))])
    # srgan_mse_test_generator = srgan_mse_dict['test']['generator']
    # valid_loss = np.array([srgan_mse_test_generator[i][0] for i in range(len(srgan_mse_test_generator))])
    # assert train_loss.shape == valid_loss.shape
    # epochs = np.arange(0, train_loss.shape[0])
    # plot_loss("srgan_mse", "SRGAN Model with MSE Generator Loss", epochs, train_loss, valid_loss)
    # pickle_in.close()

    # SRGAN PL History
    # pickle_in = open(path[3],"rb")
    # srgan_pl_dict = pickle.load(pickle_in)
    # srgan_pl_train_generator = srgan_pl_dict['train']['generator']
    # train_loss = np.array([srgan_pl_train_generator[i][0] for i in range(len(srgan_pl_train_generator))])
    # srgan_pl_test_generator = srgan_pl_dict['test']['generator']
    # valid_loss = np.array([srgan_pl_test_generator[i][0] for i in range(len(srgan_pl_test_generator))])
    # assert train_loss.shape == valid_loss.shape
    # epochs = np.arange(0, train_loss.shape[0])
    # plot_loss("srgan_pl", "SRGAN Model with Perceptual Generator Loss", epochs, train_loss, valid_loss)
    # pickle_in.close()
