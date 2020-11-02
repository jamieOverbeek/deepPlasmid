# deepPlasmid

### Environment

Building the environment with Anaconda3 is recommended. Once you have installed and initiated Anaconda3, run

    conda create --name py_3.7_tensorflow1.9 python=3.7
    conda activate py_3.7_tensorflow1.9
    conda install tensorflow-gpu=1.9
    conda install keras=2.2.4
    conda install scikit-learn=0.20.2

### Data generation (using pre-existing sequence data):

Note: the one-hot encoded sequence data for training and evaluating the CNN is ~15 Gb. 

Download the tarball of sequence files, and unpack:

    tar -xvf sequence_data.tar.gz

Generate the numpy arrays of data:

    python generate_data_files.py sequence_data/
  
### Training and Evaluating Networks
  
Train a deep neural network on an embedding pre-generated with dna2vec:

    python DNN_train.py sequence_data/
  
Evaluate the deep neural network:

    python DNN_CV.py sequence_data/
  
Train a convolutional neural network on one-hot encoded sequence data:

    python CNN_train.py sequence_data/
  
Evaluate the convolutional network:

    python CNN_CV.py sequence_data/

