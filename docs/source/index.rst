.. EnrichRBP documentation master file, created by
   sphinx-quickstart on Thur Nov 28 20:12:13 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EnrichRBP's documentation!
=========================================================
**Date:** November 28, 2024. **Version:** 0.1.0

**paper:** EnrichRBP: an automated and interpretable computational platform for predicting and analyzing RNA-binding protein events  

**Citing Us:**

If you find EnrichRBP helpful in your work or research, we would greatly appreciate citations to the following paper
::
Unpublished yet.

EnrichRBP is a powerful web service that enables researchers to develop original deep learning and machine learning architectures to explore the complex dynamics of RNA-binding proteins.

**EnrichRBP supports:**

- 70 innovative deep learning algorithms, covering feature representation, selection, model training, comparison, optimization, and evaluation, all integrated within an automated pipeline.
- comprehensive visualizations, enhancing model interpretability, and facilitating the discovery of functionally significant sequence regions crucial for RBP interactions.
- ultra-fast predictions ranging from seconds to hours, applicable to both pre-trained and custom model scenarios, thus proving its utility in real-world applications.

**API Demo**

.. literalinclude:: ./API_Demo.py

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   ./getting_started.rst
   ./Installation.rst

.. toctree::
   :maxdepth: 2
   :caption: API

   ./file_operation.rst
   ./RNA_features.rst
   ./feature_selection_methods.rst
   ./evaluate_classifiers.rst
   ./analysis_plots.rst

.. toctree::
   :maxdepth: 2
   :caption: EXAMPLES

   ./basic_usage.rst
   ./fileOperation_examples.rst
   ./featureGeneration_examples.rst
   ./featureSelection_examples.rst
   ./evaluateClassifiers_examples.rst
   ./plotAnalysis_examples.rst


.. toctree::
   :maxdepth: 2
   :caption: HISTORY

   ./release_history.rst
