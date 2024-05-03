import pickle
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras import layers, optimizers
from keras.optimizers import Adam
from keras.layers import SimpleRNN
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization
from keras.losses import CategoricalCrossentropy



class NeuralNetTicTacToe():

    def __init__(self) -> None:
        self.i = 0
        self.load_data()
        self.feed_forward_network()


    def load_data(self):
        file_path = 'state_action.pkl'

        # Load the dictionary from the file using pickle
        with open(file_path, 'rb') as file:
            loaded_dict = pickle.load(file)
        states = loaded_dict.keys()
        states = [list(state) for state in states]
        actions =list(loaded_dict.values())

        y_categorical = to_categorical(actions, num_classes=9)
        x = states
        self.x = np.array(x)
        self.y_categorical = np.array(y_categorical)

    def feed_forward_network(self):
        self.batch_size = 50
        self.epochs = 50
        self.learning_rate = 0.001

        metrics = ["accuracy"]
        loss = CategoricalCrossentropy()
        opt  = Adam(learning_rate= self.learning_rate)


        model = Sequential()
        self.model = model
        model.add(Dense(1000, activation='relu', kernel_initializer='he_normal', input_shape = (9,)))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(750, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(9, activation='softmax'))
        model.compile(loss = loss, optimizer= opt, metrics= metrics )
        model.summary()


    def train(self):
        model_history =  self.model.fit(self.x, self.y_categorical,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            #shuffle=True,
                            validation_data = (self.x, self.y_categorical))


    def predict(self, board_state):
        game_board = np.array(board_state)

        # Use the trained model to make a move
        preds = self.model.predict(game_board.reshape(1, 9))
        best_moves_sorted = np.argsort(preds.ravel())[::-1]
        if self.i < 9:
            best_move_index = best_moves_sorted[self.i]
            row = best_move_index // 3
            col = best_move_index % 3
        else:
            raise Exception('no moves remaning.')
        return (row, col)
    
    def valid(self, was_valid):
        if was_valid:
            self.i = 0
        else:
            self.i += 1
