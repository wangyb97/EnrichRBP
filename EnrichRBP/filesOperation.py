import numpy as np

def read_fasta_file(fasta_file=''):
    fp = open(fasta_file, 'r')
    seqslst = []
    while True:
        s = fp.readline()
        if not s:
            break
        else:
            if '>' not in s:
                seq = s.split('\n')[0].replace('T', 'U')
                seqslst.append(seq)
    return np.array(seqslst)


def read_label(label_file=''):
    label_ls = open(label_file).readlines()
    ls = []
    for item in label_ls:
        ls.append(int(item))
    return np.array(ls)
