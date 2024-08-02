import numpy as np
from EnrichRBP.structureFeatures import *
from EnrichRBP.biologicalFeatures import *
from EnrichRBP.languageFeatures import *

def generateStructureFeatures(dataset_path, script_path, basic_path, W, L, u, dataset_name=''):
    path, E_path, H_path, I_path, M_path = defineExperimentPaths(
        basic_path, dataset_name)
    run_RNA(dataset_path, script_path, E_path, H_path, I_path, M_path, W=W, L=L, u=u)
    fEprofile = open(E_path + 'E_profile.txt')
    Eprofiles = fEprofile.readlines()

    fHprofile = open(H_path + 'H_profile.txt')
    Hprofiles = fHprofile.readlines()

    fIprofile = open(I_path + 'I_profile.txt')
    Iprofiles = fIprofile.readlines()

    fMprofile = open(M_path + 'M_profile.txt')
    Mprofiles = fMprofile.readlines()

    mw = int(1)

    fhout = open(path + 'combined_profile.txt', 'w')

    for i in range(0, int(len(Eprofiles) / 2)):
        id = Eprofiles[i * 2].split()[0]
        print(id, file=fhout)
        E_prob = Eprofiles[i * 2 + 1].split()
        H_prob = Hprofiles[i * 2 + 1].split()
        I_prob = Iprofiles[i * 2 + 1].split()
        M_prob = Mprofiles[i * 2 + 1].split()
        P_prob = list(
            map((lambda a, b, c, d: 1 - float(a) - float(b) - float(c) - float(d)), E_prob, H_prob, I_prob, M_prob))
        print(list_to_str(P_prob[mw - 1:len(P_prob)]), file=fhout)
        print(list_to_str(E_prob[mw - 1:len(P_prob)]), file=fhout)
        print(list_to_str(H_prob[mw - 1:len(P_prob)]), file=fhout)
        print(list_to_str(I_prob[mw - 1:len(P_prob)]), file=fhout)
        print(list_to_str(M_prob[mw - 1:len(P_prob)]), file=fhout)
    fhout.close()

    features = read_combined_profile(path + 'combined_profile.txt')
    return features


def generateBPFeatures(sequences, pseudoKNC=False, ktuple=3, zigzag_coding=False, guanine_cytosine_Quantity=False, nucleotide_tilt=False, percentage_of_bases=False,
                     PGKM=False, gapValue=1, kValue=2, mValue=2, DPCP=False):
    features = []
    for seq in sequences:
        feature = []
        if pseudoKNC == True:
            if ktuple not in [3, 4, 5]:
                raise Exception('The value of ktuple should be in [3, 4, 5].')
            temp = generate_pseudoKNC(seq, ktuple)
            for item in temp:
                feature.append(item)
        if zigzag_coding == True:
            temp = generate_zCurve(seq)
            for item in temp:
                feature.append(item)
        if guanine_cytosine_Quantity == True:
            temp = generate_gcContent(seq)
            for item in temp:
                feature.append(item)
        if nucleotide_tilt == True:
            temp = generate_GCAUSkew(seq)
            for item in temp:
                feature.append(item)
        if percentage_of_bases == True:
            temp = generate_GCAURatio(seq)
            for item in temp:
                feature.append(item)
        if PGKM == True:
            if gapValue not in [1, 2, 3, 4, 5]:
                raise Exception('The value of gapValue should be in [1, 2, 3, 4, 5].')
            if kValue not in [1, 2]:
                raise Exception('The value of kValue should be in [1, 2].')
            if mValue not in [1, 2]:
                raise Exception('The value of mValue should be in [1, 2].')
            temp = generate_PGKM(seq, gapValue=gapValue, kValue=kValue, mValue=mValue)
            for item in temp:
                feature.append(item)
        if DPCP == True:
            temp = generate_DPCP(seq)
            for item in temp:
                feature.append(item)

        features.append(feature)
    features = np.array(features)
    return features


def generateStaticLMFeatures(sequences, kmer=3, model=''):
    if (kmer not in [3, 4, 5, 6]) or (str(kmer) != model.split('_')[-2][0].strip()):
        raise Exception('The value of kmer should match with model and be among [3, 4, 5, 6], please check your kmer value and model.')
    LM_type = model.split('_')[-1].strip()
    if LM_type == 'doc2vec':
        return RNAdoc2vec(kmer=kmer, model=model, sequence=sequences)

    elif LM_type == 'word2vec':
        return RNAword2vec(kmer=kmer, model=model, sequence=sequences)

    elif LM_type == 'GloVe':
        return RNAglove(kmer=kmer, model=model, sequence=sequences)

    elif LM_type == 'fasttext':
        return RNAfasttext(kmer=kmer, model=model, sequence=sequences)

    else:
        raise Exception("The LM_type should be in ['doc2vec', 'word2vec', 'GloVe', 'fasttext'].")



def generateDynamicLMFeatures(sequences, kmer=3, model=''):
    if (kmer not in [3, 4, 5, 6]) or (str(kmer) != model.split('_')[-1][0].strip()):
        raise Exception(
            'The value of kmer should match with model and be among [3, 4, 5, 6], please check your kmer value and model.')

    return RNABERT(kmer=kmer, model=model, sequence=sequences)
