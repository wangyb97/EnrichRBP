import itertools
import numpy as np


nucleotides = 'AGCU'

mer2 = list(itertools.product(nucleotides, repeat=2))
mer3 = list(itertools.product(nucleotides, repeat=3))
mer4 = list(itertools.product(nucleotides, repeat=4))


def kmers(seq, k):
    kmer_list = []
    for i in range(len(seq) - k + 1):
        kmer_list.append(seq[i:i + k])
    return kmer_list


def generate_pseudoKNC(seq, k):
    pseKNC_list = []
    for i in range(1, k + 1, 1):
        temp = list(itertools.product(nucleotides, repeat=i))
        for i in temp:
            pseKNC_list.append(seq.count(''.join(i)))
    return pseKNC_list


def generate_zCurve(seq):
    temp = []
    U = seq.count('U')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')
    x_axis = (A + G) - (C + U)
    y_axis = (A + C) - (G + U)
    z_axis = (A + U) - (C + G)
    temp.append(x_axis)
    temp.append(y_axis)
    temp.append(z_axis)
    return temp


def generate_gcContent(seq):
    temp = []
    U = seq.count('U')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')
    temp.append((G + C) / (A + C + G + U) * 100.0)
    return temp


def generate_GCAUSkew(seq):
    temp = []
    U = seq.count('U')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')

    GC_Skew = (G - C) / (G + C)
    AU_Skew = (A - U) / (A + U)

    temp.append(GC_Skew)
    temp.append(AU_Skew)
    return temp


def generate_GCAURatio(seq):
    temp = []
    U = seq.count('U')
    A = seq.count('A')
    C = seq.count('C')
    G = seq.count('G')

    temp.append((A + U) / (G + C))
    return temp


def generate_PGKM(seq, gapValue, kValue, mValue):
    PGKM_list = []
    for i in range(1, gapValue + 1, 1):
        for k in range(1, kValue + 1, 1):
            for m in range(1, mValue + 1, 1):
                mers = k + m
                if mers == 2:
                    mer = mer2
                elif mers == 3:
                    mer = mer3
                elif mers == 4:
                    mer = mer4
                mer_seq = kmers(seq, i + k + m)
                for gap in mer:
                    count = 0
                    for item in mer_seq:
                        if mers == 2:
                            if item[0] == gap[0] and item[-1] == gap[1]:
                                count = count + 1
                        elif mers == 3:
                            if k == 1:
                                if item[0] == gap[0] and item[-2] == gap[1] and item[-1] == gap[2]:
                                    count = count + 1
                            else:
                                if item[0] == gap[0] and item[1] == gap[1] and item[-1] == gap[2]:
                                    count = count + 1
                        else:
                            if item[0] == gap[0] and item[1] == gap[1] and item[-2] == gap[2] and item[-1] == gap[3]:
                                count = count + 1
                    PGKM_list.append(count)
    return PGKM_list


def generate_DPCP(seq):
    phys_dic = {
        'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
        'AU': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
        'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.14],
        'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.08],
        'UA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
        'UU': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
        'UC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
        'UG': [-0.01, -1.78, 3.32, 0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26, 0.17],
        'CA': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
        'CU': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
        'CC': [-0.01, -1.78, 3.32, 0.3, 8.7, 32, -11.1, -12.2, -29.7, -3.26, 0.49],
        'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
        'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
        'GU': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.44],
        'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
        'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34]}

    seqLength = len(seq)
    sequence_vector = np.zeros([101, 11])
    k = 2
    for i in range(0, seqLength - 1):
        sequence_vector[i, 0:11] = phys_dic[seq[i:i + k]]
    return sequence_vector





