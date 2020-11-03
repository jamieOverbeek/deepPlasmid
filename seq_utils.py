import numpy as np
import time, os
import gzip


def remove_ambig_char(seq_list):
    '''Replace nucleotide ambiguity codes with 'N' before processing'''
    ambig_char = ["R", "Y", "S", "W", "K", "M", "B", "V", "D", "H"]

    for i in range(len(seq_list)):
        for char in ambig_char:
            seq_list[i] = seq_list[i].replace(char, "N")

    return seq_list


def reverse_complement(sequence):
    '''Get the reverse complement of a sequence'''
    NT_dict = {"A":"T", "T":"A", "C":"G", "G":"C", "N":"N"}
    rev_compl = "".join([NT_dict[x] for x in reversed(sequence)])

    return rev_compl


def pad_sequence(seq_list, pad_length=48, max_len=10000):
    '''Make all sequences length 10,000 by truncating or adding 'N', and add reverse complements with padding'''
    seq_pad = []

    for i in range(len(seq_list)):
        curr_seq = seq_list[i]

        if max_len > 0:
            curr_seq = curr_seq[:max_len]
            curr_seq += "N"*(max_len - len(curr_seq))

        rev_compl = reverse_complement(curr_seq)
        seq_pad.append(curr_seq + "N"*pad_length + rev_compl)

    return seq_pad


def sequence_list(filename):
    '''Retrieve lists of sequences and sequence names from .fasta or .fna files'''
    seq_list  = []
    name_list = []

    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as read_file:
            read_lines = read_file.readlines()

    else:
        with open(filename, "r") as read_file:
            read_lines = read_file.readlines()


    if ".fna" in filename:
        fna_flag = True
    elif ".fasta" in filename:
        fna_flag = False
            
    else:
        print("'%s' file format not recognized." % (filename))
        exit()

    if fna_flag:
        read_lines_fasta = []

        for i in range(len(read_lines)):
            if read_lines[i].startswith('>'):
                read_lines_fasta.append(read_lines[i].rstrip())
                read_lines_fasta.append("")

            else:
                read_lines_fasta[-1] += read_lines[i].rstrip()

        read_lines = read_lines_reformat
        read_lines_reformat = []

    for i in range(0, len(read_lines), 2):
        name = read_lines[i].rstrip()
        seq  = read_lines[i+1].rstrip().upper()
        name_list.append(name)
        seq_list.append(seq)

    return seq_list, name_list



def trim_seq(seqs, names, min_len, max_len):
    '''Split long sequences into length 10,000 segments with randomized start points'''
    trim_seqs   = []
    trim_names  = []
    trim_labels = []

    for i in range(len(seqs)):
        if len(seqs[i]) < min_len:
            continue

        elif len(seqs[i]) > max_len:
            if len(seqs[i]) % max_len > 0:
                o = np.random.choice(range(len(seqs[i])%max_len))

            else:
                o = 0

            for j in range(max_len+o, len(seqs[i]), max_len):
                trim_names.append(names[i] + "_" + str(j-max_len) + "_" + str(j))
                trim_seqs.append(seqs[i][j-max_len:j])

        else:
            trim_names.append(names[i] + "_0_" + str(len(seqs[i])))
            trim_seqs.append(seqs[i])

    return trim_seqs, trim_names


def arr_onehot(seq_list):
    '''One-hot encode a sequence of [ACGT] (with N=(0,0,0,0))'''
    NTs = ["A","C","G","T"]
    seq_one_hot = np.zeros((len(seq_list),4,len(seq_list[0])), dtype=bool)
    seq_arr = np.asarray([list(x) for x in seq_list])

    for i in range(len(NTs)):
        ind = np.argwhere(seq_arr==NTs[i])
        seq_one_hot[ind[:,0],i,ind[:,1]] = True
    
    return seq_one_hot


def get_ftp_seq(genome_dir, ID, seq_type, min_len):
    '''Get contigs from genome .fna files, checking description for sequence type'''
    seq_file = genome_dir + ID + "/" + ID + ".fna"
    seqs = []
    names = []
    
    contig_seq, contig_names = sequence_list(seq_file)

    for i in range(len(contig_names)):
        contig_length = len(contig_seq[i])
        is_plasmid = "plasmid" in contig_names[i].lower()

        if (contig_length >= min_len) & (seq_type == is_plasmid):
            seqs.append(contig_seq[i])
            names.append(contig_names[i])

    return seqs, names


def kmer_counts_split(seq_list, kmer_list):
    '''Count k-mers in sequences, normalizing by sequence length'''
    k_start = min([len(x) for x in kmer_list])
    k_end = max([len(x) for x in kmer_list])
    kmer_ind_dict = {}
    kmer_counts = np.zeros((len(seq_list), kmer_list.shape[0]))
    
    for i in range(kmer_list.shape[0]):
        kmer_ind_dict[kmer_list[i]] = i

    for j in range(len(seq_list)):
        for seq in seq_list[j].split('N'):
            for k in range(k_start,k_end+1):
                kmers = [seq[x:x+k] for x in range(0,len(seq)-k+1)]
    
                for kmer in kmers:
                    kmer_counts[j,kmer_ind_dict[kmer]] += 1
        
        kmer_counts[j] = kmer_counts[j]/len(seq_list[j].replace('N', ''))

    return kmer_counts


def calc_metrics(softmax, labels):
    '''Calculate accuracy, precision, recall, and F1 score for NN predictions'''
    labl_cat = np.argmax(labels, axis=1)
    pred_cat = np.argmax(softmax, axis=1)

    tp = np.sum((labl_cat==1) & (pred_cat==1))
    fp = np.sum((labl_cat==0) & (pred_cat==1))
    fn = np.sum((labl_cat==1) & (pred_cat==0))
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)

    # Weight accuracy if plasmids and chromosomes are unbalanced
    acc = 0.5*(np.sum((labl_cat==0) & (pred_cat==0))/np.sum(labl_cat==0)) + 0.5*(np.sum((labl_cat==1) & (pred_cat==1))/np.sum(labl_cat==1))
    print("  Weighted Accuracy: %0.2f" % (acc))
    print("  Precision: %0.2f" % (precision))
    print("  Recall: %0.2f" % (recall))
    print("  F1 Score: %0.2f" % (2*(precision*recall)/(precision+recall)))
