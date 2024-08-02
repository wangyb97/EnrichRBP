import numpy as np
from skfeature.utility.sparse_learning import *
from skfeature.function.information_theoretical_based import CIFE, CMIM, DISR, ICAP, JMI, MIFS, MIM, MRMR
from skfeature.function.similarity_based import fisher_score, reliefF, trace_ratio
from skfeature.function.sparse_learning_based import ll_l21, ls_l21
from skfeature.function.statistical_based import CFS, chi_square, f_score, gini_index, t_score
from tensorflow.keras.utils import to_categorical


def cife(features, labels, num_features=10):
    idx, _, _ = CIFE.cife(features, labels, n_selected_features=num_features)
    features = np.array(features[:, idx[0:num_features]])

    return features


def cmim(features, labels, num_features=10):
    idx, _, _ = CMIM.cmim(features, labels, n_selected_features=num_features)
    features = features[:, idx[0:num_features]]

    return features

def disr(features, labels, num_features=10):
    idx, _, _ = DISR.disr(features, labels, n_selected_features=num_features)
    features = features[:, idx[0:num_features]]

    return features


def icap(features, labels, num_features=10):
    idx, _, _ = ICAP.icap(features, labels, n_selected_features=num_features)
    features = features[:, idx[0:num_features]]

    return features


def jmi(features, labels, num_features=10):
    idx, _, _ = JMI.jmi(features, labels, n_selected_features=num_features)
    features = features[:, idx[0:num_features]]

    return features


def mifs(features, labels, num_features=10):
    idx, _, _ = MIFS.mifs(features, labels, n_selected_features=num_features)
    features = features[:, idx[0:num_features]]

    return features


def mim(features, labels, num_features=10):
    idx, _, _ = MIM.mim(features, labels, n_selected_features=num_features)
    features = features[:, idx[0:num_features]]

    return features


def mrmr(features, labels, num_features=10):
    idx, _, _ = MRMR.mrmr(features, labels, n_selected_features=num_features)
    features = features[:, idx[0:num_features]]

    return features


def fisherScore(features, labels, num_features=10):
    score = fisher_score.fisher_score(features, labels)
    idx = fisher_score.feature_ranking(score)
    features = features[:, idx[0:num_features]]

    return features


def relief_f(features, labels, num_features=10):
    score = reliefF.reliefF(features, labels)
    idx = reliefF.feature_ranking(score)
    features = features[:, idx[0:num_features]]

    return features


def traceRatio(features, labels, num_features=10):
    idx, feature_score, subset_score = trace_ratio.trace_ratio(features, labels, num_features, style='fisher')
    features = features[:, idx[0:num_features]]

    return features


def llL21(features, labels, num_features=10):
    labels = to_categorical(labels)
    Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(features, labels, 0.1, verbose=False)
    idx = feature_ranking(Weight)
    features = features[:, idx[0:num_features]]

    return features


def lsL21(features, labels, num_features=10):
    labels = to_categorical(labels)
    Weight, obj, value_gamma = ls_l21.proximal_gradient_descent(features, labels, 0.1, verbose=False)
    idx = feature_ranking(Weight)
    features = features[:, idx[0:num_features]]

    return features


def cfs(features, labels, num_features=10): # time-consuming
    idx = CFS.cfs(features, labels)
    features = features[:, idx[0:num_features]]

    return features


def chiSquare(features, labels, num_features=10):
    score = chi_square.chi_square(features, labels)
    idx = chi_square.feature_ranking(score)
    features = features[:, idx[0:num_features]]

    return features


def fScore(features, labels, num_features=10):
    score = f_score.f_score(features, labels)
    idx = f_score.feature_ranking(score)
    features = features[:, idx[0:num_features]]

    return features


def giniIndex(features, labels, num_features=10):
    label = []
    for item in labels:
        if item == '1':
            label.append(1)
        else:
            label.append(0)
    label = np.array(label)
    score = gini_index.gini_index(features, label)
    idx = gini_index.feature_ranking(score)
    features = features[:, idx[0:num_features]]

    return features


def tScore(features, labels, num_features=10):
    score = t_score.t_score(features, labels)
    idx = t_score.feature_ranking(score)
    features = features[:, idx[0:num_features]]

    return features


