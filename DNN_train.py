import numpy as np
import time
import argparse
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import load_model 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Set Keras to use only 1 GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

parser = argparse.ArgumentParser(description='Train a neural network on embedded kmer counts')
parser.add_argument("dataDir", help="Directory which contains numpy array data", type=str)
parser.add_argument("-d", "--dropout", dest="dropout", default=0.1, type=float, help="Dropout fraction to use on dense layers")
parser.add_argument("-e", "--epochs", dest="epochs", default=100, type=int, help="Number of epochs")
parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
parser.add_argument("-n_f", "--nodes_f", dest="nodes_f", default=128, type=int, help="Number of nodes in first dense layer")
parser.add_argument("-n_s", "--nodes_s", dest="nodes_s", default=128, type=int, help="Number of nodes in second dense layer")
args = parser.parse_args()

print(args)

start_time = time.time()

# Load data
X_train = np.load(args.dataDir + "train_emb.npy")
y_train = np.load(args.dataDir + "train_y.npy").reshape(-1,1)
X_val = np.load(args.dataDir + "val_emb.npy")
y_val = np.load(args.dataDir + "val_y.npy").reshape(-1,1)

# Normalize embedding components
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Encode labels
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train = enc.fit_transform(y_train)
y_val = enc.transform(y_val)

# Build the model
model = Sequential()
model.add(Dense(input_shape=(100,), units=args.nodes_f, activation="relu"))

if args.dropout > 0.0:
    model.add(Dropout(args.dropout))
else:
    model.add(BatchNormalization())

model.add(Dense(units=args.nodes_s, activation="relu"))

if args.dropout > 0.0:
    model.add(Dropout(args.dropout))
else:
    model.add(BatchNormalization())
        
model.add(Dense(units=2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="DNN_checkpoint.hdf5", monitor="val_acc",
                               mode='auto', verbose=1, save_best_only=True)  

print(model.summary())

# Train the model
history = model.fit(X_train, y_train, batch_size = args.batch_size, shuffle=True,
                        validation_data=(X_val, y_val), epochs=args.epochs,
                        verbose=2, callbacks=[checkpointer])

# serialize model to JSON and save
model_json = model.to_json()

with open('DNN_model.json', 'w') as f:
    print(model_json, file=f)

print("Total time: %0.2f seconds." % (time.time()-start_time))
