# import tensorflow.python.keras.engine.functional
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.layers import Activation,\
    Concatenate, AveragePooling1D, Dropout, GRU, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Convolution1D, Dense, BatchNormalization, MaxPool1D, Flatten, Attention
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'



clf_names = ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'BaggingClassifier', 'RandomForestClassifier',
             'AdaBoostClassifier', 'GradientBoostingClassifier', 'SVM', 'LinearDiscriminantAnalysis', 'ExtraTreesClassifier']

ML_Classifiers = [
    LogisticRegression(max_iter=10000),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    BaggingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    SVC(probability=True),
    LinearDiscriminantAnalysis(),
    ExtraTreesClassifier()
]
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min', restore_best_weights=True)]

def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)
    input_at = Activation('relu')(input_bn)
    input_dp = Dropout(0.4)(input_at)
    return input_dp

def ConvolutionBlock(input, f, k):
    A1 = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    A1 = bn_activation_dropout(A1)
    return A1

def MultiScale(input):
    A = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 3)
    D = ConvolutionBlock(input, 64, 1)
    D = ConvolutionBlock(D, 64, 5)
    D = ConvolutionBlock(D, 64, 5)
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('relu')(result)
    return result


def createMLP(hidden_sizes):
    return MLPClassifier(hidden_layer_sizes=hidden_sizes, max_iter=300, learning_rate='adaptive', early_stopping=True,
                         verbose=True, n_iter_no_change=5)


def createCNN(shape1, shape2):
    CNN_input = Input(shape=(shape1, shape2))
    x = Convolution1D(filters=32, kernel_size=3, activation='relu')(CNN_input)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=32, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(padding='same')(x)
    x = Convolution1D(filters=64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(padding='same')(x)
    x = Flatten()(x)
    x = Dense(units=768, activation='relu')(x)
    output = Dense(units=2, activation='sigmoid')(x)

    model = Model(inputs=CNN_input, outputs=output)

    return model


def createRNN(shape1, shape2):
    RNN_input = Input(shape=(shape1, shape2))
    x = Bidirectional(LSTM(units=16, return_sequences=False))(RNN_input)
    x = Attention()([x, x])
    x = Dense(units=768, activation='relu')(x)
    output = Dense(units=2, activation='sigmoid')(x)

    model = Model(inputs=RNN_input, outputs=output)

    return model


def createResNet(shape1, shape2):
    sequence_input = Input(shape=(shape1, shape2), name='sequence_input')
    sequence = Convolution1D(filters=128, kernel_size=3, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('relu')(sequence)
    overallResult = MultiScale(sequence)
    overallResult = AveragePooling1D(pool_size=5)(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = Bidirectional(GRU(120, return_sequences=True))(overallResult)
    overallResult = Flatten()(overallResult)
    overallResult = Dense(768, activation='relu')(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)

    return Model(inputs=[sequence_input], outputs=[ss_output])



def evaluateMLclassifers(features, labels, file_path='', shuffle=True, folds=5):
    scale = StandardScaler()
    features = scale.fit_transform(features)

    cv = KFold(n_splits=folds, shuffle=shuffle)

    results_file = open(file_path + 'ML_evalution_metrics.csv', 'w')

    results_file.write('clf_name,metrics,metric_name\n')

    print('Starting runnning machine learning classifiers using ' + str(folds) + '-fold cross-validation, please be patient...')
    for clf_name, clf in zip(clf_names, ML_Classifiers):
        ACCs = []
        F1_Scores = []
        AUCs = []
        MCCs = []
        Recalls = []
        print('running ' + clf_name + '...')
        for train_index, test_index in cv.split(labels):
            train_features = features[train_index]
            train_labels = labels[train_index]

            test_features = features[test_index]
            test_labels = labels[test_index]
            clf.fit(train_features, train_labels)

            pre_proba = clf.predict_proba(test_features)[:, 1]
            pre_labels = clf.predict(test_features)


            auc = roc_auc_score(y_true=test_labels, y_score=pre_proba)
            acc = accuracy_score(y_pred=pre_labels, y_true=test_labels)
            f1 = f1_score(y_true=test_labels, y_pred=pre_labels)
            mcc = matthews_corrcoef(y_true=test_labels, y_pred=pre_labels)
            recall = recall_score(y_true=test_labels, y_pred=pre_labels)

            AUCs.append(auc)
            ACCs.append(acc)
            MCCs.append(mcc)
            Recalls.append(recall)
            F1_Scores.append(f1)
        print('finish')

        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(AUCs)) + ',' + 'AUC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(ACCs)) + ',' + 'ACC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(MCCs)) + ',' + 'MCC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(Recalls)) + ',' + 'Recall\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(F1_Scores)) + ',F1_Scores\n')

    print('All classifiers have finished running, the result file are locate in ' + file_path)

    results_file.close()




def evaluateDLclassifers(features, labels, file_path='', shuffle=True, folds=5):
    CNN = createCNN(features.shape[1], features.shape[2])
    RNN = createRNN(features.shape[1], features.shape[2])
    ResNet = createResNet(features.shape[1], features.shape[2])
    MLP = createMLP((5, 5, 5, 2, 2))
    DL_classifiers = {'CNN':CNN, 'RNN':RNN, 'ResNet-1D':ResNet, 'MLP':MLP}

    cv = KFold(n_splits=folds, shuffle=shuffle)
    results_file = open(file_path + 'DL_evalution_metrics.csv', 'w')

    results_file.write('clf_name,metrics,metric_name\n')

    labels_2D = to_categorical(labels)
    print('Starting runnning deep learning models using ' + str(folds) + '-fold cross-validation, please be patient...')
    for clf_name in DL_classifiers:
        print('running ' + clf_name + '...')
        ACCs = []
        F1_Scores = []
        AUCs = []
        MCCs = []
        Recalls = []
        for train_index, test_index in cv.split(labels):
            train_features = features[train_index]
            train_labels_2D = labels_2D[train_index]
            train_labels = labels[train_index]

            test_features = features[test_index]
            test_labels_2D = labels_2D[test_index]
            test_labels = labels[test_index]
            if clf_name == 'MLP':
                train_features = train_features.reshape(train_features.shape[0], -1)
                test_features = test_features.reshape(test_features.shape[0], -1)
                DL_classifiers[clf_name].fit(train_features, train_labels)
                pre_proba = DL_classifiers[clf_name].predict_proba(test_features)[:, 1]
                pre_labels = DL_classifiers[clf_name].predict(test_features)
                auc = roc_auc_score(y_true=test_labels, y_score=pre_proba)
                acc = accuracy_score(y_pred=pre_labels, y_true=test_labels)
                f1 = f1_score(y_true=test_labels, y_pred=pre_labels)
                mcc = matthews_corrcoef(y_true=test_labels, y_pred=pre_labels)
                recall = recall_score(y_true=test_labels, y_pred=pre_labels)

            else:
                DL_classifiers[clf_name].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                train_X, val_X, train_Y, val_Y = train_test_split(train_features, train_labels_2D, test_size=0.10, stratify=train_labels_2D)
                DL_classifiers[clf_name].fit(x=train_X, y=train_Y, epochs=30, batch_size=64, verbose=0, shuffle=True, callbacks=callbacks,
                          validation_data=(val_X, val_Y))
                pre_proba = DL_classifiers[clf_name].predict(test_features)
                pre_labels = np.argmax(pre_proba, axis=-1)
                pre_proba = pre_proba[:, 1]
                auc = roc_auc_score(y_true=test_labels, y_score=np.array(pre_proba))
                acc = accuracy_score(y_pred=np.array(pre_labels), y_true=np.array(test_labels))
                f1 = f1_score(y_true=np.array(test_labels), y_pred=np.array(pre_labels))
                mcc = matthews_corrcoef(y_true=np.array(test_labels), y_pred=np.array(pre_labels))
                recall = recall_score(y_true=np.array(test_labels), y_pred=np.array(pre_labels))

            AUCs.append(auc)
            ACCs.append(acc)
            MCCs.append(mcc)
            Recalls.append(recall)
            F1_Scores.append(f1)
        print('finish')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(AUCs)) + ',' + 'AUC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(ACCs)) + ',' + 'ACC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(MCCs)) + ',' + 'MCC\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(Recalls)) + ',' + 'Recall\n')
        results_file.write(clf_name + ',')
        results_file.write(str(np.mean(F1_Scores)) + ',F1_Scores\n')
    print('All models have finished running, the result file are locate in ' + file_path)

    results_file.close()


