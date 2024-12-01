EnrichRBP.metricsPlot
================================
Many visualization functions are integrated in EnrichRBP for plotting different types of data or performance analysis, which requires some dependencies such as ``matplotlib``, ``sklearn``, ``seaborn``, ``shap`` and ``yellobrick``.

.. py:function:: EnrichRBP.metricsPlot.roc_curve_deeplearning(label_list, pred_proba_list, name_list, image_path='')

    :Parameters:
            .. class:: label_list:list

                    The list of label arrays corresponding to the sequences used to train each classifier, label value should be in {-1,1} or {0,1}.

            .. class:: pred_proba_list:list

                    The list of target score arrays corresponding to the sequences used to train each classifier, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).

            .. class:: name_list:list

                    The list of names corresponding to each classifier, the names in the list will be shown in final ``.png`` image file.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.

    :Attributes:
            .. class:: fpr:numpy array of shape (>2,)

                    False positive rate.

            .. class:: tpr:numpy array of shape (>2,)

                    True positive rate.

.. py:function:: EnrichRBP.metricsPlot.roc_curve_machinelearning(features, labels, clf_list, image_path='', test_size=0.25, random_state=0)

    :Parameters:
            .. class:: features:numpy array

                    Two-dimensional real number matrix used to fit each classifiers.

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels. The value of labels should be in {-1, 1} or {0, 1}

            .. class:: clf_list:list

                    The list of ``sklearn classifiers`` used to analyse roc curve.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.

            .. class:: test_size:float or int, default=0.25

                    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.

            .. class:: random_state:int, RandomState instance or None, default=0

                    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.



.. py:function:: EnrichRBP.metricsPlot.partial_dependence(features, labels, clf, feature_names, image_path='', subsample=50, n_jobs=3, random_state=0, grid_resolution=20)

    :Parameters:
            .. class:: features:{numpy array or dataframe} of shape (n_samples, n_features)

                    Features is used to generate a grid of values for the target features (where the partial dependence will be evaluated).

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: feature_names:array-like of shape (n_features,)

                    Name of each feature; feature_names[i] holds the name of the feature with index i.

            .. class:: clf:sklearn classifier

                    A fitted estimator object implementing predict, predict_proba, or decision_function. Multioutput-multiclass classifiers are ``not supported``.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.

            .. class:: subsample:float, int or None, default=50

                    Sampling for ICE curves. If ``float``, should be between 0.0 and 1.0 and represent the proportion of the dataset to be used to plot ICE curves. If ``int``, represents the absolute number samples to use.

            .. class:: n_jobs:int, default=3

                    The number of CPUs to use to compute the partial dependences.

            .. class:: random_state:int, RandomState instance or None, default=0

                    Controls the randomness of the selected samples when subsamples is not ``None``

            .. class:: grid_resolution:int, default=20

                    The number of equally spaced points on the axes of the plots, for each target feature.



.. py:function:: EnrichRBP.metricsPlot.confusion_matirx_deeplearning(test_labels, pred_labels, image_path='')

    :Parameters:
            .. class:: test_labels:numpy array of shape (n_samples,)

                    Ground truth labels corresponding to sequences in dataset.

            .. class:: pred_labels:numpy array of shape (n_samples,)

                    Estimated labels conducted by a deep learning model.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.

.. py:function:: EnrichRBP.metricsPlot.confusion_matrix_machinelearning(clf, features, labels, label_tags=None, test_size=0.25, normalize=None, random_state=0, image_path='')

    :Parameters:
            .. class:: clf:sklearn classifier

                    A sklearn classifier instance.

            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    Labels to index the matrix.

            .. class:: label_tags:list of names for different classes, default=None

                    Target names used for plotting. By default, ``labels`` will be used.

            .. class:: test_size:float or int, default=0.25

                    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.

            .. class:: normalize:{'true', 'pred', 'all'}, default=None

                    Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. If None, confusion matrix will not be normalized.

            .. class:: random_state:int, RandomState instance or None, default=0

                    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.

.. py:function:: EnrichRBP.metricsPlot.det_curve_machinelearning(features, labels, clf_list, image_path='', test_size=0.25, random_state=0)

    :Parameters:

            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: clf_list:list

                    List of classifiers used to draw det curve.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.

            .. class:: test_size:float or int, default=0.25

                    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.

            .. class:: random_state:int, RandomState instance or None, default=0

                    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.


.. py:function:: EnrichRBP.metricsPlot.det_curve_deeplearning(label_list, pred_proba_list, name_list, image_path='')

    :Parameters:
            .. class:: label_list:list

                    The list of label arrays corresponding to the sequences used to train each classifier, label value should be in {-1,1} or {0,1}.

            .. class:: pred_proba_list:list

                    The list of target score arrays corresponding to the sequences used to train each classifier, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).

            .. class:: name_list:list

                    The list of names corresponding to each classifier, the names in the list will be shown in final ``.png`` image file.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.



.. py:function:: EnrichRBP.metricsPlot.precision_recall_curve_machinelearning(features, labels, clf_list, image_path='', test_size=0.25, random_state=0)

    :Parameters:
            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.

            .. class:: test_size:float or int, default=0.25

                    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.

            .. class:: random_state:int, RandomState instance or None, default=0

                    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.



.. py:function:: EnrichRBP.metricsPlot.precision_recall_curve_deeplearning(label_list, pred_labels_list, name_list, image_path='')

    :Parameters:
            .. class:: label_list:list

                    The list of label arrays corresponding to the sequences used to train each classifier, label value should be in {-1,1} or {0,1}.

            .. class:: pred_proba_list:list

                    The list of target score arrays corresponding to the sequences used to train each classifier, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).

            .. class:: name_list:list

                    The list of names corresponding to each classifier, the names in the list will be shown in final ``.png`` image file.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.


.. py:function:: EnrichRBP.metricsPlot.shap_bar(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path='')

    :Parameters:
            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: clf:sklearn classifier

                    A fitted estimator object implementing predict, predict_proba, or decision_function. Multioutput-multiclass classifiers are ``not supported``.

            .. class:: sample_size:tuple, default=(0, 100)

                    Defines the number of samples used to perform the shap value calculation.

            .. class:: feature_size:tuple, default=(0, 10)

                    Defines the features for calculating shap values.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.


.. py:function:: EnrichRBP.metricsPlot.shap_scatter(features, labels, clf, feature_id, sample_size=(0, 100), feature_size=(0, 10), image_path='')

    :Parameters:
            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: clf:sklearn classifier

                    A fitted estimator object implementing predict, predict_proba, or decision_function. Multioutput-multiclass classifiers are ``not supported``.

            .. class:: feature_id:int

                    The feature id for visualization, which should be less than or equal to the difference - 1 between the two values in ``feature_size``

            .. class:: sample_size:tuple, default=(0, 100)

                    Defines the number of samples used to perform the shap value calculation.

            .. class:: feature_size:tuple, default=(0, 10)

                    Defines the features for calculating shap values.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.



.. py:function:: EnrichRBP.metricsPlot.shap_waterfall(features, labels, clf, feature_id, sample_size=(0, 100), feature_size=(0, 10), image_path='')

    :Parameters:
             .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: clf:sklearn classifier

                    A fitted estimator object implementing predict, predict_proba, or decision_function. Multioutput-multiclass classifiers are ``not supported``.

            .. class:: sample_size:tuple, default=(0, 100)

                    Defines the number of samples used to perform the shap value calculation.

            .. class:: feature_size:tuple, default=(0, 10)

                    Defines the features for calculating shap values.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.


.. py:function:: EnrichRBP.metricsPlot.shap_interaction_scatter(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path='')

    :Parameters:
            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: clf:sklearn classifier

                    A fitted estimator object implementing predict, predict_proba, or decision_function. Multioutput-multiclass classifiers are ``not supported``.

            .. class:: sample_size:tuple, default=(0, 100)

                    Defines the number of samples used to perform the shap value calculation.

            .. class:: feature_size:tuple, default=(0, 10)

                    Defines the features for calculating shap values.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.



.. py:function:: EnrichRBP.metricsPlot.shap_beeswarm(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path='')

    :Parameters:
            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: clf:sklearn classifier

                    A fitted estimator object implementing predict, predict_proba, or decision_function. Multioutput-multiclass classifiers are ``not supported``.

            .. class:: sample_size:tuple, default=(0, 100)

                    Defines the number of samples used to perform the shap value calculation.

            .. class:: feature_size:tuple, default=(0, 10)

                    Defines the features for calculating shap values.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.


.. py:function:: EnrichRBP.metricsPlot.shap_heatmap(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path='')

    :Parameters:
            .. class:: features:numpy array of shape (n_samples, n_features)

                    Input features corresponding to the sequences.

            .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

            .. class:: clf:sklearn classifier

                    A fitted estimator object implementing predict, predict_proba, or decision_function. Multioutput-multiclass classifiers are ``not supported``.

            .. class:: sample_size:tuple, default=(0, 100)

                    Defines the number of samples used to perform the shap value calculation.

            .. class:: feature_size:tuple, default=(0, 10)

                    Defines the features for calculating shap values.

            .. class:: image_path:str, default=''

                    The path used to store the final image file.



.. py:function:: EnrichRBP.metricsPlot.violinplot(features, x_id, y_id, image_path='')

    :Parameters:
        .. class:: features:dataframe of shape (n_samples, n_features)

                    Input features corresponding to the sequences.

        .. class:: x_id:str

                    Name of variables in ``data`` or vector data.

        .. class:: y_id:str

                    Name of variables in ``data`` or vector data.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.


.. py:function:: EnrichRBP.metricsPlot.boxplot(features, x_id, y_id, image_path='')

    :Parameters:
        .. class:: features:dataframe of shape (n_samples, n_features)

                    Input features corresponding to the sequences.

        .. class:: x_id:str

                    Name of variables in ``data`` or vector data.

        .. class:: y_id:str

                    Name of variables in ``data`` or vector data.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.




.. py:function:: EnrichRBP.metricsPlot.pointplot(features, x_id, y_id, image_path='')

    :Parameters:
        .. class:: features:dataframe of shape (n_samples, n_features)

                    Input features corresponding to the sequences.

        .. class:: x_id:str

                    Name of variables in ``features`` or vector data.

        .. class:: y_id:str

                    Name of variables in ``features`` or vector data.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.



.. py:function:: EnrichRBP.metricsPlot.barplot(features, x_id, y_id, image_path='')

    :Parameters:
        .. class:: features:dataframe of shape (n_samples, n_features)

                    Input features corresponding to the sequences.

        .. class:: x_id:str

                    Name of variables in ``features`` or vector data.

        .. class:: y_id:str

                    Name of variables in ``features`` or vector data.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.


.. py:function:: EnrichRBP.metricsPlot.sns_heatmap(features, sample_size=(0, 15), feature_size=(0, 15), image_path='')

    :Parameters:
        .. class:: features:numpy array of shape (n_samples, n_features)

                Input features corresponding to the sequences.

        .. class:: sample_size:tuple, default=(0, 15)

                The sample range used to plot the heatmap.

        .. class:: feature_size:tuple, default=(0, 15)

                The feature range used to plot the heatmap.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.

.. py:function:: EnrichRBP.metricsPlot.prediction_error(features, labels, classes, clf, test_size=0.25, random_state=0, image_path='')

    :Parameters:

        .. class:: features:numpy array of shape (n_samples, n_features)

                Input features corresponding to the sequences.

        .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

        .. class:: classes:list of str

                The class labels to use for the legend. Specifying classes in this manner is used to change the class names to a more specific format or to label encoded integer classes.

        .. class:: test_size:float or int, default=0.25

                    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.

        .. class:: random_state:int, RandomState instance or None, default=0

                    Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

        .. class:: clf: classifier

                A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.

.. py:function:: EnrichRBP.metricsPlot.descrimination_threshold(features, labels, clf, image_path='')

    :Parameters:

        .. class:: features:numpy array of shape (n_samples, n_features)

                Input features corresponding to the sequences.

        .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

        .. class:: clf: classifier

                A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.

.. py:function:: EnrichRBP.metricsPlot.learning_curve(features, labels, clf, folds=5, image_path='')

    :Parameters:

        .. class:: features:numpy array of shape (n_samples, n_features)

                Input features corresponding to the sequences.

        .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

        .. class:: folds:int, default=5

                Cross-validated folds, which divides the training set into 5 (or other values) subsets, where one subset is the validation set, and the other ``fold - 1`` subsets constitute the training set. Each subset needs to be performed once as a validation set.

        .. class:: clf: classifier

                A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.


.. py:function:: EnrichRBP.metricsPlot.cross_validation_score(clf, features, labels, folds=5, scoring=None, image_path='')

    :Parameters:

        .. class:: folds:int, default=5

                Cross-validated folds, which divides the training set into 5 (or other values) subsets, where one subset is the validation set, and the other ``fold - 1`` subsets constitute the training set. Each subset needs to be performed once as a validation set.

        .. class:: scoring:string, callable or None, optional, default: None

                A string or scorer callable object / function with signature ``scorer(estimator, features, labels)``

        .. class:: clf: classifier

                A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.

        .. class:: features:numpy array of shape (n_samples, n_features)

                Input features corresponding to the sequences.

        .. class:: labels:numpy array of shape (n_samples,)

                    True binary labels used to fit classifier. The value of labels should be in {-1, 1} or {0, 1}.

        .. class:: image_path:str, default=''

                    The path used to store the final image file.

