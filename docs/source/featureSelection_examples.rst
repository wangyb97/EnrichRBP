Feature selection examples
==============================================================

The EnrichRBP integrates several feature selection methods and provides a simple interface, which requires only the features to be selected, the dataset label, and the number of features you want to selected.


Importing related modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: py

    from EnrichRBP.filesOperation import read_fasta_file, read_label
    from EnrichRBP.Features import generateStaticLMFeatures, generateStructureFeatures, generateBPFeatures
    from EnrichRBP.featureSelection import cife # Here we use cife method as an example.


Prepare three types of features for feature selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AGO1 dataset is used to generate biological features, semantic information and secondary structure information respectively.

.. code-block:: py

    fasta_path = '/home/wangyansong/wangyubo/EnrichRBP/src/RNA_datasets/circRNAdataset/AGO1/seq'
    label_path = '/home/wangyansong/wangyubo/EnrichRBP/src/RNA_datasets/circRNAdataset/AGO1/label'

    sequences = read_fasta_file(fasta_path)  # read sequences and labels from given path
    label = read_label(label_path)

    # generate biological features
    biological_features = generateBPFeatures(sequences, PGKM=True)
    print(biological_features.shape)

    # generate static semantic information
    static_semantic_information = generateStaticLMFeatures(sequences, kmer=3, model='/home/wangyansong/wangyubo/EnrichRBP/src/staticRNALM/circleRNA/circRNA_3mer_fasttext')
    print(static_semantic_information.shape)

    # generate secondary structure information
    structure_features = generateStructureFeatures(fasta_path, script_path='/home/wangyansong/wangyubo/EnrichRBP/RNAplfold', basic_path='/home/wangyansong/wangyubo/EnrichRBP/src/circRNAdatasetAGO1', W=101, L=70, u=1)
    print(structure_features.shape)


output:

    ::

        (34636, 400)
        (34636, 99, 100)
        (34636, 101, 5)



Feature selection procedure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It should be noted that the feature dimension to be passed into the feature selection method needs to be two-dimensional, so features with semantic information and secondary structure information need to be downscaled for feature selection (the same applies to machine learning classifiers).

.. code-block:: py

    # The first method of dimensionality reduction: multiplying the last two dimensions.
    print('-------------------first method------------------------')
    static_semantic_information_1 = np.reshape(static_semantic_information, (static_semantic_information.shape[0], static_semantic_information.shape[1] * static_semantic_information.shape[2]))
    print(static_semantic_information_1.shape)

    structure_features_1 = np.reshape(structure_features, (structure_features.shape[0], structure_features.shape[1] * structure_features.shape[2]))
    print(structure_features_1.shape)


    # The second method of dimensionality reduction: sum operation according to one of the last two dimensions.
    print('------------------second method-----------------------')
    print('----------compress the third dimension----------------')
    static_semantic_information_2 = np.sum(static_semantic_information, axis=1)
    print(static_semantic_information_2.shape)
    static_semantic_information_3 = np.sum(static_semantic_information, axis=2)
    print(static_semantic_information_3.shape)

    print('---------compress the second dimension----------------')
    structure_features_2 = np.sum(structure_features, axis=1)
    print(structure_features_2.shape)
    structure_features_3 = np.sum(structure_features, axis=2)
    print(structure_features_3.shape)

output:

    ::

        -------------------first method------------------------
        (34636, 9900)
        (34636, 505)
        ------------------second method-----------------------
        ----------compress the third dimension----------------
        (34636, 100)
        (34636, 99)
        ---------compress the second dimension----------------
        (34636, 5)
        (34636, 101)

Input the three processed features into the feature selection method for refinement (here we use the ``CIFE`` method as an example).

.. code-block:: py

    refined_biological_features = cife(biological_features, label, num_features=10)
    print(refined_biological_features.shape)

    refined_static_semantic_information = cife(static_semantic_information_1, label, num_features=10)
    print(refined_static_semantic_information.shape)

    refined_structure_features = cife(structure_features_1, label, num_features=10)
    print(refined_structure_features.shape)

output:

    ::

        (34636, 10)
        (34636, 10)
        (34636, 10)


.. note:: The calculation process of some feature selection methods is more complicated, so the running time is longer, please be patient.