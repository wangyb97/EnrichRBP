EnrichRBP.filesOperation
===================================================

.. py:function:: EnrichRBP.filesOperation.read_fasta_file(fasta_file='')

    This function is used to read a sequence file in `fasta` format in the following form.
    ::

        >hsa_circ_0000005 start:16578,end:16712
        AGGCGTGGCTACTGCGGCTGGAGCTGCGATGAGACTCGGAACTCCTCGTCTTACTTTGTGCTCCATGTTTTGTTTTTGTATTTTGGTTTGTAAATTTGTAG
        >hsa_circ_0000023 start:50,end:174
        ACCCTTTCTGCCAGCCAGCTAGCCAGGGCCCAGAAACAAACACCGATGGCTTCTTCCCCACGTCCCAAGATGGATGCAATCTTAACTGAGGCCATTAAGGC
        >hsa_circ_0000038 start:189,end:318
        CCGTCCCCCCCACTGCCTACTCATATACCTCCAGAGCCTCCACGCACCCCTCCATTCCCTGCTAAGACTTTTCAAGTTGTGCCAGAAATTGAGTTTCCACC

    :Parameters: .. class:: fasta_file:str, default=''

                        Path to the fasta file (absolute path).

    :Attributes: .. class:: seqslst:list

                        The list used to store sequences from fasta file.

.. py:function:: EnrichRBP.fileOperation.read_label(label_file='')

    This function is used to read a label file in `txt` format in the following form.
    ::

        0
        1
        1
        0
        0

    :Parameters: .. class:: label_file:str, default=''

                        Path to the label file (absolute path).

    :Attributes: .. class:: label_ls:list

                        The list used to store labels according to the sequences. It will be transformed to array when the function returns.

