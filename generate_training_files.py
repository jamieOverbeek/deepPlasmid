from JCO_utils import *
import numpy as np
import time

data_dir = '../data/'
data_files = ['underrep', 'overrep', 'train', 'val', 'test']
dl_embedding = np.load(data_dir + 'dna2vec_train_emb.npy')
dl_kmers = np.load(data_dir + 'dna2vec_train_emb_kmers.npy')
time_mark = time.time()

for filename in data_files:
    chr_seqs, chr_names = sequence_list(data_dir + filename + '_chr.fasta')
    pl_seqs, pl_names = sequence_list(data_dir + filename + '_pl.fasta')

    class_labels = [0]*len(chr_seqs) + [1]*len(pl_seqs)
    seqs = chr_seqs + pl_seqs
    names = chr_names + pl_names
    
    seqs = remove_ambig_char(seqs)

    for i in range(len(seqs)):
        if np.random.choice([0,1])==1:
            seqs[i] = reverse_complement(seqs[i])

    np.save(data_dir + filename + '_y', np.asarray(class_labels))

    print('Sequences preprocessed in %0.1f seconds.' % (time.time()-time_mark))
    time_mark = time.time()

    kmer_counts = kmer_counts_split(seqs, dl_kmers)
    kmer_emb = np.dot(kmer_counts, dl_embedding)

    np.save(data_dir + filename + '_emb', kmer_emb)
    np.save(data_dir + filename + '_names', np.asarray(names))

    print('Embedding of k-mer counts generated in %0.1f seconds.' % (time.time()-time_mark))
    time_mark = time.time()

    seqs = pad_sequence(seqs)
    seqs = arr_onehot(seqs)

    np.save(data_dir + filename + '_onehot', seqs)
    print('One-hot files generated in %0.1f seconds.' % (time.time()-time_mark))
