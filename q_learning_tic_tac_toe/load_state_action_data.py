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

# File path where the dictionary was saved
file_path = 'state_action.pkl'

# Load the dictionary from the file using pickle
with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)

# Print the loaded dictionary
print("Loaded dictionary:")
print(loaded_dict)

states = loaded_dict.keys()
states = [list(state) for state in states]
actions =list(loaded_dict.values())

data_y_categorical = to_categorical(actions, num_classes=9)
data_x = states

batch_size = 100
epochs = 70
learning_rate = 0.001

metrics = ["accuracy"] # keras.metrics.Accuracy(), keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()
loss = CategoricalCrossentropy() # "mean_absolute_error" #'mean_squared_error'
opt  = Adam(learning_rate= learning_rate) # default 0.001

model = Sequential()

model.add(Dense(2000, activation='relu', kernel_initializer='he_normal', input_shape = (9,)))

model.add(Dense(1000, activation='relu'))

model.add(Dense(500, activation='relu'))

model.add(Dense(9, activation='softmax'))

model.compile(loss = loss, optimizer= opt, metrics= metrics)

model.summary()

# Convert lists to NumPy arrays
data_x = np.array(data_x)
data_y_categorical = np.array(data_y_categorical)
print("Shape of data_x:", data_x.shape)
print("Shape of data_y_categorical:", data_y_categorical.shape)


model_history = model.fit(data_x, data_y_categorical,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          #shuffle=True,
                          validation_data = (data_x, data_y_categorical))

model.save('tic_tac_toe_model.h5')

board_size = 3
board = np.zeros((board_size, board_size), dtype = int)

board[0][0] = 1
board[0][1] = -1
board[1][0] = 1
board[1][1] = -1
preds = model.predict(board.reshape(1, 9))
print(preds)
print(np.argmax(preds))
