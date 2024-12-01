EnrichRBP.evaluateClassifiers
=============================================

This two functions are used to evaluate the validity of the sequence representations obtained in the preceding process.

.. py:function:: EnrichRBP.evaluateClassifiers.evaluateDLclassifers(features, labels, file_path='', shuffle=True, folds=5)

    ``EnrichRBP`` integrates four classical deep learning models (CNN, RNN, MLP and ResNet), cross-validates them using the representation matrix on the four classes of models, and stores the final performance metrics obtained for each model in ``DL_evalution_metrics.csv``.

    :Parameters:
                .. class:: features:numpy array, necessary parameters

                        Sequence feature matrix for training the four deep learning models.

                .. class:: labels:numpy array, necessary parameters

                        The label corresponding to each sequence (which indicates whether the corresponding sequence is the target sequence of the RBPs).

                .. class:: file_path:str, default=''

                        Path for storing cross-validation result files.

                .. class:: shuffle:bool, default=True

                        Whether to perform disorder when dividing sequence subsets used for cross-validation.

                .. class:: folds:int, default=5

                        Cross-validated folds, which divides the training set into 5 (or other values) subsets, where one subset is the validation set, and the other ``fold - 1`` subsets constitute the training set. Each subset needs to be performed once as a validation set.


.. py:function:: EnrichRBP.evaluateClassifiers.evaluateMLclassifers(features, labels, file_path='', shuffle=True, folds=5)

    ``EnrichRBP`` integrates eleven classical machine learning models (Logistic Regression, K-Nearest Neighbor, Decision Tree, GaussianNB, Bagging, Random Forest, AdaBoost, Gradient Boosting, SVM, LDA and ExtRa Trees), cross-validates them using the representation matrix on each model, and stores the final performance metrics obtained for each model in ``ML_evalution_metrics.csv``.

    :Parameters:
                .. class:: features:numpy array, necessary parameters

                        Sequence feature matrix for training the machine learning models.

                .. class:: labels:numpy array, necessary parameters

                        The label corresponding to each sequence (which indicates whether the corresponding sequence is the target sequence of the RBPs).

                .. class:: file_path:str, default=''

                        Path for storing cross-validation result files.

                .. class:: shuffle:bool, default=True

                        Whether to perform disorder when dividing sequence subsets used for cross-validation.

                .. class:: folds:int, default=5

                        Cross-validated folds, which divides the training set into 5 (or other values) subsets, where one subset is the validation set, and the other ``fold - 1`` subsets constitute the training set. Each subset needs to be performed once as a validation set.
