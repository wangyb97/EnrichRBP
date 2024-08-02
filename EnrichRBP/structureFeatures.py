import argparse
import os
import sys
from keras.preprocessing.sequence import pad_sequences
import re
import linecache
import numpy as np
from functools import reduce
from collections import OrderedDict

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))



def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def list_to_str(lst):
    ''' Given a list, return the string of that list with tab separators
    '''
    return reduce( (lambda s, f: s + '\t' + str(f)), lst, '')


def concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region):
    combine_list = [pairedness.split(), hairpin_loop.split(), internal_loop.split(), multi_loop.split(), external_region.split()]
    return np.array(combine_list).T


def defineExperimentPaths(basic_path, name_id):
    path = basic_path + str(name_id) + '/'
    E_path = basic_path + str(name_id) + '/E/'
    H_path = basic_path + str(name_id) + '/H/'
    I_path = basic_path + str(name_id) + '/I/'
    M_path = basic_path + str(name_id) + '/M/'
    mk_dir(E_path)
    mk_dir(H_path)
    mk_dir(I_path)
    mk_dir(M_path)
    return path, E_path, H_path, I_path, M_path

def read_combined_profile(file_path):
    i = 0
    secondary_structure_list = []
    filelines = linecache.getlines(file_path)
    file_length = len(filelines)
    print(file_length)
    while i <= file_length - 1:
        pairedness = re.sub('[\s+]', ' ', filelines[i + 1].strip())
        hairpin_loop = re.sub('[\s+]', ' ', filelines[i + 2].strip())
        internal_loop = re.sub('[\s+]', ' ', filelines[i + 3].strip())
        multi_loop = re.sub('[\s+]', ' ', filelines[i + 4].strip())
        external_region = re.sub('[\s+]', ' ', filelines[i + 5].strip())
        combine_array = concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region)
        # X = pad_sequences(combine_array, maxlen=int(160), dtype=np.str, value=seq_encoding_keys.index('UNK'),
        #                   padding='post')
        secondary_structure_list.append(combine_array)
        i = i + 6

    return np.array(secondary_structure_list).astype(float)


def definecombinePaths(basic_path, name_id):
    path = basic_path + str(name_id) + '/'
    E_path = basic_path + str(name_id) + '/E/'
    H_path = basic_path + str(name_id) + '/H/'
    I_path = basic_path + str(name_id) + '/I/'
    M_path = basic_path + str(name_id) + '/M/'
    return path, E_path, H_path, I_path, M_path


def run_RNA(fasta_path, script_path, E_path, H_path, I_path, M_path, W, L, u):
    os.system(
        script_path + '/E_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        E_path + 'E_profile.txt')
    os.system(
        script_path + '/H_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        H_path + 'H_profile.txt')
    os.system(
        script_path + '/I_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        I_path + 'I_profile.txt')
    os.system(
        script_path + '/M_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        M_path + 'M_profile.txt')