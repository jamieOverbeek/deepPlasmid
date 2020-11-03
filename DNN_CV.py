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
from keras.models import model_from_json
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from seq_utils import *

## Set Keras to use only 1 GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

parser = argparse.ArgumentParser(description='Evaluate a neural network on embedded kmer counts')
parser.add_argument("data_dir", help="Directory which contains numpy array data", type=str)
args = parser.parse_args()

# Load data
X_train = np.load(args.data_dir + "train_emb.npy")
y_train = np.load(args.data_dir + "train_y.npy").reshape(-1,1)
X_val = np.load(args.data_dir + "val_emb.npy")
y_val = np.load(args.data_dir + "val_y.npy").reshape(-1,1)
X_test = np.load(args.data_dir + "test_emb.npy")
y_test = np.load(args.data_dir + "test_y.npy").reshape(-1,1)
X_under = np.load(args.data_dir + "underrep_emb.npy")
y_under = np.load(args.data_dir + "underrep_y.npy").reshape(-1,1)
X_over = np.load(args.data_dir + "overrep_emb.npy")
y_over = np.load(args.data_dir + "overrep_y.npy").reshape(-1,1)

# One-hot encoding labels
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train = enc.fit_transform(y_train)
y_val = enc.transform(y_val)
y_test = enc.transform(y_test)
y_under = enc.transform(y_under)
y_over = enc.transform(y_over)

# Scale embedding components
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_under = scaler.transform(X_under)
X_over = scaler.transform(X_over)

# Load trained model
with open('DNN_model.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights('DNN_checkpoint.hdf5')

print("Loaded model")
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# Compute and save softmax values
val_sm = model.predict(X_val,verbose=0)
test_sm = model.predict(X_test,verbose=0)
overrep_sm = model.predict(X_over,verbose=0)
underrep_sm = model.predict(X_under,verbose=0)

np.save('DNN_val_softmax', val_sm)
np.save('DNN_test_softmax', test_sm)
np.save('DNN_overrep_softmax', overrep_sm)
np.save('DNN_underrep_softmax', underrep_sm)

# Give scores
print("Test Data:")
calc_metrics(test_sm, y_test)
print("Underrepresented Genera:")
calc_metrics(underrep_sm, y_under)
print("Overrepresented Genera:")
calc_metrics(overrep_sm, y_over)
