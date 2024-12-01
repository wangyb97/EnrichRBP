Evaluating classifiers examples
==================================================================

In the EnrichRBP, we integrate several machine learning classifiers from sklearn and implement several classical deep learning models for users to perform performance tests, for which we provide two easy-to-use functions for machine learning classifiers and deep learning models respectively.

Importing related modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: py

    from EnrichRBP.evaluateClassifiers import evaluateDLclassifers, evaluateMLclassifers
    from EnrichRBP.Features import generateDynamicLMFeatures, generateStructureFeatures, generateBPFeatures
    from EnrichRBP.filesOperation import read_fasta_file, read_label


Evaluating various machine learning classifiers using biological features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A total of 11 machine learning classifiers are included in the ``EnrichRBP``. After the function finishes running, an ``ML_evalution_metrics.csv`` is generated, which contains the performance metrics of each classifier on the dataset.

.. code-block:: py

    fasta_path = '/home/wangyansong/wangyubo/EnrichRBP/src/RNA_datasets/circRNAdataset/AGO1/seq'
    label_path = '/home/wangyansong/wangyubo/EnrichRBP/src/RNA_datasets/circRNAdataset/AGO1/label'

    sequences = read_fasta_file(fasta_path)  # read sequences and labels from given path
    label = read_label(label_path)

    # Generating PGKM features for example.
    biological_features = generateBPFeatures(sequences, PGKM=True)

    # Perform feature selection to refine the biological features
    refined_biological_features = cife(biological_features, label, num_features=10)

    # Perform a 5-fold cross-validation of the machine learning classifier using biological features, and store the result file in the current folder.
    evaluateMLclassifers(refined_biological_features, folds=5, labels=label, file_path='./', shuffle=True)


output:

    ::

        Starting runnning machine learning classifiers using 5-fold cross-validation, please be patient...
        running LogisticRegression...
        finish
        running KNeighborsClassifier...
        finish
        running DecisionTreeClassifier...
        finish
        running GaussianNB...
        finish
        running BaggingClassifier...
        finish
        running RandomForestClassifier...
        finish
        running AdaBoostClassifier...
        finish
        running GradientBoostingClassifier...
        finish
        running SVM...
        finish
        running LinearDiscriminantAnalysis...
        finish
        running ExtraTreesClassifier...
        finish
        All classifiers have finished running, the result file are locate in ./


Evaluating various deep learning models using dynamic semantic information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``EnrichRBP`` we implement four classes of classical deep learning models, including ``CNN``, ``RNN``, ``ResNet-1D``, and ``MLP``. After the function finishes running, an ``DL_evalution_metrics.csv`` is generated, which contains the performance metrics of each model on the dataset.

We use the same dataset as previous example to evaluate deep learning models.

.. code-block:: py

    # Generating 4mer dynamic semantic information for evaluating models.
    dynamic_semantic_information = generateDynamicLMFeatures(sequences, kmer=4, model='/home/wangyansong/wangyubo/EnrichRBP/src/dynamicRNALM/circleRNA/pytorch_model_4mer')

    # Perform a 5-fold cross-validation of the machine learning classifier using biological features, and store the result file in the current folder.
    evaluateDLclassifers(dynamic_semantic_information, folds=5, labels=label, file_path='./', shuffle=True)

output:

    ::

        Starting runnning deep learning models using 5-fold cross-validation, please be patient...
        running CNN...
        (some log information)
        finish
        running RNN...
        (some log information)
        finish
        running ResNet-1D...
        (some log information)
        finish
        running MLP
        (some log information)
        finish
        All models have finished running, the result file are locate in ./


.. note:: The performance in the package is for reference only, and targeted hyperparameters need to be set for specific datasets to perform at their best.
