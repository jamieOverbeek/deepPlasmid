import pickle
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import time
import argparse
from keras.optimizers import SGD
from keras.models import load_model 
from sklearn.preprocessing import OneHotEncoder

# Set Keras to use only 1 GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

parser = argparse.ArgumentParser(description='Train a neural network on one-hot DNA sequences')
parser.add_argument("data_dir", help="Directory which contains numpy array data", type=str)
parser.add_argument("-d", "--dropout", dest="dropout", default=0.0, type=float, help="Dropout fraction to use on dense layer")
parser.add_argument("-e", "--epochs", dest="epochs", default=20, type=int, help="Number of epochs")
parser.add_argument("-b", "--batch_size", dest="batch_size", default=32, type=int, help="Batch size")
parser.add_argument("-f", "--filters", dest="filters", default=512, type=int, help="Number of filters in convolutional layer")
parser.add_argument("-w", "--width", dest="width", default=12, type=int, help="Filter width")
parser.add_argument("-n", "--nodes", dest="nodes", default=64, type=int, help="Number of nodes in dense layer")
args = parser.parse_args()

print(args)

start_time = time.time()

# load data
args.data_dir = args.data_dir.rstrip('/')
val_data   = np.load(args.data_dir + "/val_onehot.npy").transpose(0,2,1)
val_labels = np.load(args.data_dir + "/val_y.npy").reshape(-1,1)
seq_len    = val_data.shape[1]
X = np.load(args.data_dir + "/train_onehot.npy").transpose(0,2,1)
y = np.load(args.data_dir + "/train_y.npy").reshape(-1,1)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y = enc.fit_transform(y)
val_labels = enc.transform(val_labels)

# model specification
model = Sequential()
model.add(Conv1D(input_shape=(seq_len,4), filters=args.filters,
                 kernel_size=args.width, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=seq_len))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(input_dim=args.filters, units=args.nodes, activation="relu"))

if args.dropout==0.0:
    model.add(BatchNormalization())
    
else:
    model.add(Dropout(args.dropout))
        
model.add(Dense(units=2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="CNN_checkpoint.hdf5", monitor="val_acc",
                               mode='auto', verbose=1, save_best_only=True)  

print(model.summary())

# Full training onehot data is too large to fit in GPU memory, 
# so train in chunks with a integer number of batches
chunk_size = 10000 - (10000 % args.batch_size)
ind = np.asarray(range(X.shape[0]))

# train the model
for epoch in range(args.epochs):
    print("Epoch = " + str(epoch+1) + " out of " + str( args.epochs))
    np.random.shuffle(ind)

    for f in range(0,X.shape[0]-chunk_size,chunk_size):
        X_train = X[ind[f:f+chunk_size]]
        y_train = y[ind[f:f+chunk_size]]
        history = model.fit(X_train, y_train, batch_size=args.batch_size,
                            validation_split=0.0, epochs=1, verbose=2)

    # train final chunk and do validation
    X_train = X[ind[f+chunk_size:]]
    y_train = y[ind[f+chunk_size:]]
    history = model.fit(X_train, y_train, batch_size = args.batch_size,
                        validation_data=(val_data, val_labels), epochs=1,
                        verbose=2, callbacks=[checkpointer])

# serialize model to JSON and save
model_json = model.to_json()

with open('CNN_model.json', 'w') as f:
    print(model_json, file=f)

print("Total time: %0.2f seconds." % (time.time()-start_time))
