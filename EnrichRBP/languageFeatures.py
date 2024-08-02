import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from glove import Glove
from gensim.models import FastText
import torch
from transformers import BertModel, BertTokenizer


def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


def convertKmer2glove(words, model):
    vector = []
    for word in words:
        vector.append(model.word_vectors[model.dictionary[word]])
    return vector


def seq2glove(seqs, kmer, model):
    vectors = []
    for num, seq in enumerate(seqs):
        seq = seq.strip()
        seq = seq.replace('T', 'U')
        l = len(seq)
        words = []
        for i in range(0, l, 1):
            if i + kmer >= l + 1:
                break
            words.append(seq[i:i + kmer])
        vectors.append(convertKmer2glove(words, model))
    vectors = np.array(vectors)
    return vectors


def convertKmer2fasttext(words, model):
    vector = []
    for word in words:
        vector.append(model.wv.get_vector(word))
    return vector


def convertKmer2BERT(dataloader, model):
    features = []
    seq = []
    tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BertModel.from_pretrained(model)

    model = model.to(device)
    model = model.eval()
    for sequences in dataloader:
        seq.append(sequences)
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedding = embedding.last_hidden_state.cpu().numpy()

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            features.append(seq_emd)
    return features


def seq2fasttext(seqs, kmer, model):
    vectors = []
    for num, seq in enumerate(seqs):
        seq = seq.strip()
        seq = seq.replace('T', 'U')
        l = len(seq)
        words = []
        for i in range(0, l, 1):
            if i + kmer >= l + 1:
                break
            words.append(seq[i:i + kmer])
        vectors.append(convertKmer2fasttext(words, model))
    vectors = np.array(vectors)
    return vectors


def convertWord2Vector(words, wv):
    vector = []
    for word in words:
        if word in wv:
            vector.append(wv.vectors[wv.vocab[word].index])
        else:
            continue
    return vector


def seq2ngram(seqs, kmer, wv):
    vectors = []
    for num, seq in enumerate(seqs):
        seq = seq.strip()
        seq = seq.replace('T', 'U')
        l = len(seq)
        words = []
        for i in range(0, l, 1):
            if i + kmer >= l + 1:
                break
            words.append(seq[i:i + kmer])
        vectors.append(convertWord2Vector(words, wv))
    vectors = np.array(vectors)
    return vectors


def RNAdoc2vec(kmer, model, sequence):
    model1 = Doc2Vec.load(model)
    seq_vector = seq2ngram(sequence, kmer, model1.wv)
    seq_vector = pad_sequences(seq_vector, padding='post')
    return seq_vector


def RNAword2vec(kmer, model, sequence):
    model1 = KeyedVectors.load(model)
    seq_vector = seq2ngram(sequence, kmer, model1.wv)
    seq_vector = pad_sequences(seq_vector, padding='post')
    return seq_vector

def RNAglove(kmer, model, sequence):
    glove = Glove.load(model)
    seq_vector = seq2glove(sequence, kmer, glove)
    seq_vector = pad_sequences(seq_vector, padding='post')
    return seq_vector

def RNAfasttext(kmer, model, sequence):
    fasttext = FastText.load(model)
    seq_vector = seq2fasttext(sequence, kmer, fasttext)
    seq_vector = pad_sequences(seq_vector, padding='post')
    return seq_vector


def RNABERT(kmer, model, sequence):
    Bert_Feature = []
    seqs = []
    for seq in sequence:
        seq = seq.strip().replace('U', 'T')
        ss = seq2kmer(seq, kmer)
        seqs.append(ss)
    sequences = torch.utils.data.DataLoader(seqs, batch_size=32, shuffle=False)
    Features = convertKmer2BERT(sequences, model)
    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature)
    return np.array(Bert_Feature)



