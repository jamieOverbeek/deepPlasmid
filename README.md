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

Download the tarball of sequence files, sequence_data.tar.gz, and unpack:

    tar -xvzf sequence_data.tar.gz

The sequence_data/ directory will contain .fasta files for training, testing, and validation data; .plasclass files containing PlasClass predictions; and .plasflow_pred.tsv files containing PlasFlow predictions.

Generate the numpy arrays of data based on a pre-generated embedding (for the dense neural network) and one-hot encoding of sequences (for the convolutional neural network) with:

    python generate_data_files.py sequence_data/

Note: the one-hot encoded sequence data for training and evaluating the CNN is ~15 Gb.

### Training and Evaluating Networks
  
Train a dense neural network on the pre-generated 100 embedding features:

    python DNN_train.py sequence_data/
  
Evaluate the dense neural network performance:

    python DNN_CV.py sequence_data/
  
Train a convolutional neural network on one-hot encoded sequence data:

    python CNN_train.py sequence_data/
  
Evaluate the convolutional network performance:

    python CNN_CV.py sequence_data/

Compare the dense neural network accuracy by genus with PlasFlow and PlasClass classifications:

    python compare_genus_accuracy.py sequence_data/
    
All python scripts take as input the directory with sequence and array data, and have a --help option. The default network parameters in DNN_train.py and CNN_train.py are the best parameters selected via hyperparameter optimization.
