from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn import metrics
from sklearn.metrics import plot_precision_recall_curve
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_det_curve
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import DiscriminationThreshold
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import LearningCurve
from yellowbrick.model_selection import CVScores
import shap
import seaborn as sns
import numpy as np


def roc_curve_deeplearning(label_list, pred_proba_list, name_list, image_path=''):
    for test_label, pred_proba, name in zip(label_list, pred_proba_list, name_list):
        ax = plt.gca()
        fpr, tpr, threshold = metrics.roc_curve(test_label, pred_proba)
        roc_auc = auc(fpr, tpr)
        viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        viz.plot(ax=ax, name=name)
    plt.legend(prop={'size': 6})
    plt.savefig(image_path + 'roc_curve.png', dpi=150, bbox_inches='tight')


def roc_curve_machinelearning(features, labels, clf_list, image_path='', test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)
    for clf in clf_list:
        ax = plt.gca()
        clf.fit(X_train, y_train)
        plot_roc_curve(clf, X_test, y_test, ax=ax, alpha=0.8)

    plt.legend(prop={'size': 6})
    plt.savefig(image_path + 'roc_curve.png', dpi=150, bbox_inches='tight')


def partial_dependence(features, labels, clf, feature_names, image_path='', subsample=50, n_jobs=3, random_state=0,
                       grid_resolution=20):
    clf.fit(features, labels)
    display = plot_partial_dependence(clf, features, feature_names, kind="individual", subsample=subsample,
                                      n_jobs=n_jobs,
                                      random_state=random_state, grid_resolution=grid_resolution)
    display.figure_.suptitle('Partial dependence')
    display.figure_.subplots_adjust(hspace=0.3)
    plt.savefig(image_path + 'partial_dependence.png', bbox_inches='tight')


def confusion_matirx_deeplearning(test_labels, pred_labels, image_path=''):
    cm = metrics.confusion_matrix(test_labels, pred_labels)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(image_path + 'confusion_matrix.png', bbox_inches='tight')


def confusion_matrix_machinelearning(clf, features, labels, label_tags=None, test_size=0.25, normalize=None, random_state=0, image_path=''):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=random_state,
                                                        test_size=test_size)
    clf.fit(X_train, y_train)
    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=label_tags,
                                 normalize=normalize)
    if normalize == None:
        title = 'without_normalization'
    else:
        title = 'normalization'
    disp.ax_.set_title(title)
    plt.savefig(image_path + '/' + title + '_confusionMatrix.png', bbox_inches='tight')


def det_curve_machinelearning(features, labels, clf_list, image_path='', test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)
    for clf in clf_list:
        ax = plt.gca()
        clf.fit(X_train, y_train)
        plot_det_curve(clf, X_test, y_test, ax=ax, alpha=0.8)

    plt.legend(prop={'size': 6})
    plt.savefig(image_path + 'det_curve.png', dpi=150, bbox_inches='tight')


def det_curve_deeplearning(label_list, pred_proba_list, name_list, image_path=''):
    for test_label, pred_proba, name in zip(label_list, pred_proba_list, name_list):
        ax = plt.gca()
        fpr, fnr, threshold = metrics.det_curve(test_label, pred_proba)
        display = metrics.DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name='example estimator')
        display.plot(ax=ax,name=name)
    plt.legend(prop={'size': 6})
    plt.savefig(image_path + 'det_curve.png', dpi=150, bbox_inches='tight')


def precision_recall_curve_machinelearning(features, labels, clf_list, image_path='', test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)
    for clf in clf_list:
        ax = plt.gca()
        clf.fit(X_train, y_train)
        plot_precision_recall_curve(clf, X_test, y_test, ax=ax, alpha=0.8)

    plt.legend(prop={'size': 6})
    plt.savefig(image_path + 'precision_recall_curve.png', dpi=150, bbox_inches='tight')


def precision_recall_curve_deeplearning(label_list, pred_labels_list, name_list, image_path=''):
    for test_label, pred_label, name in zip(label_list, pred_labels_list, name_list):
        ax = plt.gca()
        precision, recall, _ = metrics.precision_recall_curve(test_label, pred_label)
        disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot(ax=ax, name=name)

    plt.legend(prop={'size': 6})
    plt.savefig(image_path + 'precision_recall_curve.png', dpi=150, bbox_inches='tight')


def shap_bar(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path=''):
    sample_size_start, sample_size_end = sample_size
    feature_size_start, feature_size_end = feature_size
    sample_slice = slice(sample_size_start, sample_size_end)
    feature_slice = slice(feature_size_start, feature_size_end)
    clf.fit(features[:, feature_slice], labels)
    explainer = shap.explainers.Exact(clf.predict_proba, features[:, feature_slice])
    shap_values = explainer(features[sample_slice, feature_slice])
    shap_values = shap_values[..., 1]
    shap.plots.bar(shap_values)
    plt.savefig(image_path + 'shap_bar.png', bbox_inches='tight')


def shap_scatter(features, labels, clf, feature_id, sample_size=(0, 100), feature_size=(0, 10), image_path=''):
    sample_size_start, sample_size_end = sample_size
    feature_size_start, feature_size_end = feature_size
    sample_slice = slice(sample_size_start, sample_size_end)
    feature_slice = slice(feature_size_start, feature_size_end)
    clf.fit(features[:, feature_slice], labels)
    explainer = shap.explainers.Exact(clf.predict_proba, features[:, feature_slice])
    shap_values = explainer(features[sample_slice, feature_slice])
    shap_values = shap_values[..., 1]
    shap.plots.scatter(shap_values[:, feature_id], color=shap_values)
    plt.savefig(image_path + 'shap_scatter.png', bbox_inches='tight')


def shap_waterfall(features, labels, clf, feature_id, sample_size=(0, 100), feature_size=(0, 10), image_path=''):
    sample_size_start, sample_size_end = sample_size
    feature_size_start, feature_size_end = feature_size
    sample_slice = slice(sample_size_start, sample_size_end)
    feature_slice = slice(feature_size_start, feature_size_end)
    clf.fit(features[:, feature_slice], labels)
    explainer = shap.explainers.Exact(clf.predict_proba, features[:, feature_slice])
    shap_values = explainer(features[sample_slice, feature_slice])
    shap_values = shap_values[..., 1]
    shap.plots.waterfall(shap_values[feature_id])
    plt.savefig(image_path + 'shap_waterfall.png', bbox_inches='tight')


def shap_interaction_scatter(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path=''):
    sample_size_start, sample_size_end = sample_size
    feature_size_start, feature_size_end = feature_size
    sample_slice = slice(sample_size_start, sample_size_end)
    feature_slice = slice(feature_size_start, feature_size_end)
    clf.fit(features[:, feature_slice], labels)
    explainer = shap.explainers.Exact(clf.predict_proba, features[:, feature_slice])
    interaction_shap_values = explainer(features[sample_slice, feature_slice], interactions=True)
    shap.plots.scatter(interaction_shap_values[:, :, 0])
    plt.savefig(image_path + 'shap_interaction_scatter.png', bbox_inches='tight')


def shap_beeswarm(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path=''):
    sample_size_start, sample_size_end = sample_size
    feature_size_start, feature_size_end = feature_size
    sample_slice = slice(sample_size_start, sample_size_end)
    feature_slice = slice(feature_size_start, feature_size_end)
    clf.fit(features[:, feature_slice], labels)
    explainer = shap.explainers.Exact(clf.predict_proba, features[:, feature_slice])
    shap_values = explainer(features[sample_slice, feature_slice])
    shap_values = shap_values[..., 1]
    shap.plots.beeswarm(shap_values)
    plt.savefig(image_path + 'beeswarm.png', bbox_inches='tight')


def shap_heatmap(features, labels, clf, sample_size=(0, 100), feature_size=(0, 10), image_path=''):
    sample_size_start, sample_size_end = sample_size
    feature_size_start, feature_size_end = feature_size
    sample_slice = slice(sample_size_start, sample_size_end)
    feature_slice = slice(feature_size_start, feature_size_end)
    clf.fit(features[:, feature_slice], labels)
    explainer = shap.explainers.Exact(clf.predict_proba, features[:, feature_slice])
    shap_values = explainer(features[sample_slice, feature_slice])
    shap_values = shap_values[..., 1]
    shap.plots.heatmap(shap_values)
    plt.savefig(image_path + 'shap_heatmap.png', bbox_inches='tight')


def violinplot(features, x_id, y_id, image_path=''):
    sns.violinplot(x=x_id, y=y_id, data=features,
                   inner=None)
    sns.swarmplot(x=x_id, y=y_id, data=features, color="white", edgecolor="gray")
    plt.savefig(image_path + 'violinplot.png', bbox_inches='tight')


def boxplot(features, x_id, y_id, image_path=''):
    sns.boxplot(x=x_id, y=y_id, data=features, whis=np.inf)
    sns.stripplot(x=x_id, y=y_id, data=features, color=".3")
    plt.savefig(image_path + 'boxplot.png', bbox_inches='tight')


def pointplot(features, x_id, y_id, image_path=''):
    sns.pointplot(x=x_id, y=y_id, data=features)
    plt.savefig(image_path + 'pointplot.png', bbox_inches='tight')


def barplot(features, x_id, y_id, image_path=''):
    sns.barplot(x=x_id, y=y_id, data=features)
    plt.savefig(image_path + 'barplot.png', bbox_inches='tight')


def sns_heatmap(features, sample_size=(0, 15), feature_size=(0, 15), image_path=''):
    sample_size_start, sample_size_end = sample_size
    feature_size_start, feature_size_end = feature_size
    sample_slice = slice(sample_size_start, sample_size_end)
    feature_slice = slice(feature_size_start, feature_size_end)
    features = features[sample_slice, feature_slice]
    print(features.shape)
    sns.heatmap(features)
    plt.savefig(image_path + 'sns_heatmap.png', bbox_inches='tight')


def prediction_error(features, labels, classes, clf, test_size=0.25, random_state=0, image_path=''):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)
    visualizer = ClassPredictionError(clf, classes=classes)

    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    plt.savefig(image_path + 'prediction_error.png', dpi=150, bbox_inches='tight')


def descrimination_threshold(features, labels, clf, image_path=''):
    visualizer = DiscriminationThreshold(clf)
    visualizer.fit(features, labels)
    visualizer.show()
    plt.savefig(image_path + 'descrimination_threshold.png', dpi=150, bbox_inches='tight')


def learning_curve(features, labels, clf, folds=5, image_path=''):
    cv = StratifiedKFold(n_splits=folds)
    visualizer = LearningCurve(
        clf, cv=cv, scoring='f1_weighted', n_jobs=4
    )
    visualizer.fit(features, labels)
    visualizer.show()
    plt.savefig(image_path + 'learning_curve.png', dpi=150, bbox_inches='tight')


def cross_validation_score(clf, features, labels, folds=5, scoring=None, image_path=''):
    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    visualizer = CVScores(clf, cv=cv, scoring=scoring)
    visualizer.fit(features, labels)
    visualizer.show()
    plt.savefig(image_path + 'cv_score.png', dpi=150, bbox_inches='tight')
