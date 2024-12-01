Installation
=======================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following packages are requirements:

- ``gensim``
- ``glove``
- ``glove_python_binary``
- ``Keras``
- ``matplotlib``
- ``numpy``
- ``scikit_learn``
- ``seaborn``
- ``shap``
- ``tensorflow`` or ``tensorflow_gpu``
- ``torch``
- ``transformers``
- ``yellowbrick``

Install EnrichRBP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n EnrichRBP python=3.7.6
    conda activate EnrichRBP
    git clone https://github.com/wangyb97/EnrichRBP.git
    cd EnrichRBP
    pip install -r requirement.txt

Choose the appropriate torch version for your own devices

::

    pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


skfeature installation

::

    git clone https://github.com/jundongl/scikit-feature.git
    cd scikit-feature
    python setup.py install

Note for OSX users: due to its use of OpenMP, glove-python-binary does not compile under Clang. To install it, you will need a reasonably recent version of gcc (from Homebrew for instance). This should be picked up by ``setup.py``.

::

    git clone https://github.com/maciejkula/glove-python.git
    cd glove-python
    python setup.py install