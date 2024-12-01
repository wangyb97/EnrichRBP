File operation examples
===========================

This page shows how to read a dataset using the ``fileOperation`` module of the ``EnrichRBP``

.. code-block:: py

    from EnrichRBP.filesOperation import read_fasta_file, read_label

Load the sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function ``read_fasta_file`` reads a ``.txt``, ``.fasta`` or ``.fa`` text file according to the path given in the parameters, and filters or replaces empty lines and 'T' characters, finally it returns a numpy array containing all sequences.

.. code-block:: py

    fasta_path = '/home/wangyansong/wangyubo/EnrichRBP/src/RNA_datasets/circRNAdataset/AGO1/seq' # Replace the path to load your own sequences of dataset

    sequences = read_fasta_file(fasta_path)
    print(type(sequences))
    print(sequences.shape)

output:
    ::

        <class 'numpy.ndarray'>
        (34636,)

Load the labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function ``read_label`` reads a text file and returns a numpy array containing labels corresponding to the sequences.

.. code-block:: py

    label_path = '/home/wangyansong/wangyubo/EnrichRBP/src/RNA_datasets/circRNAdataset/AGO1/label' # Replace the path to load your own labels of dataset

    label = read_label(label_path)
    print(type(label))
    print(label.shape)

output:
    ::

        <class 'numpy.ndarray'>
        (34636,)


