EnrichRBP.featureSelection
==============================================

The ``EnrichRBP`` integrates feature selection methods based on four different evaluation approaches (information theoretical based, similarity based, sparse learning based and statistical based).

Information theoretical based
----------------------------------------------

.. py:function:: EnrichRBP.featureSelection.cife(features, labels, num_features=10)

    This function uses Conditional Infomax Feature Extraction [CIFE]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [CIFE] Dahua Lin and Xiaoou Tang. 2006. Conditional infomax learning: An integrated framework for feature extraction and fusion. In ECCV. 68–82.

.. py:function:: EnrichRBP.featureSelection.cmim(features, labels, num_features=10)

    This function uses Conditional Mutual Information Maximization [CMIM]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [CMIM] François Fleuret. 2004. Fast binary feature selection with conditional mutual information. JMLR 5 (2004), 1531–1555.

.. py:function:: EnrichRBP.featureSelection.disr(features, labels, num_features=10)

    This function uses Double Input Symmetrical Relevance [DISR]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [DISR] Patrick Emmanuel Meyer, Colas Schretter, and Gianluca Bontempi. 2008. Information-theoretic feature selection in microarray data using variable complementarity. IEEE J. Select. Top. Sign. Process. 2, 3 (2008), 261–274.

.. py:function:: EnrichRBP.featureSelection.icap(features, labels, num_features=10)

    This function uses mutual information [ICAP]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [ICAP] Ali El Akadi, Abdeljalil El Ouardighi, and Driss Aboutajdine. 2008. A powerful feature selection approach based on mutual information. Int. J. Comput. Sci. Netw. Secur. 8, 4 (2008), 116.

.. py:function:: EnrichRBP.featureSelection.jmi(features, labels, num_features=10)

    This function uses Joint Mutual Information [JMI]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [JMI] Patrick Emmanuel Meyer, Colas Schretter, and Gianluca Bontempi. 2008. Information-theoretic feature selection in microarray data using variable complementarity. IEEE J. Select. Top. Sign. Process. 2, 3 (2008), 261–274.

.. py:function:: EnrichRBP.featureSelection.mifs(features, labels, num_features=10)

    This function uses Mutual Information Feature Selection [MIFS]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [MIFS] Roberto Battiti. 1994. Using mutual information for selecting features in supervised neural net learning. IEEE Trans. Neural Network. 5, 4 (1994), 537–550.

.. py:function:: EnrichRBP.featureSelection.mim(features, labels, num_features=10)

    This function uses Mutual Information Maximization [MIM]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [MIM] David D. Lewis. 1992. Feature selection and feature extraction for text categorization. In Proceedings of the Workshop on Speech and Natural Language. 212–217.

.. py:function:: EnrichRBP.featureSelection.mrmr(features, labels, num_features=10)

    This function uses Minimum Redundancy Maximum Relevance [MRMR]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [MRMR] Hanchuan Peng, Fuhui Long, and Chris Ding. 2005. Feature selection based on mutual information criteria of maxdependency, max-relevance, and min-redundancy. IEEE Trans. Pattern Anal. Mach. Intell. 27, 8 (2005), 1226–1238.

Similarity based
----------------------------------------------

.. py:function:: EnrichRBP.featureSelection.fisherScore(features, labels, num_features=10)

    This function uses  Fisher Score [fisherscore]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [fisherscore] Richard O. Duda, Peter E. Hart, and David G. Stork. 2012. Pattern Classification. John Wiley & Sons.

.. py:function:: EnrichRBP.featureSelection.relief_f(features, labels, num_features=10)

    This function uses ReliefF [reliefF]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [reliefF] Marko Robnik-Šikonja and Igor Kononenko. 2003. Theoretical and empirical analysis of relieff and rrelieff. Mach. Learn. 53, 1-2 (2003), 23–69.

.. py:function:: EnrichRBP.featureSelection.traceRatio(features, labels, num_features=10)

    This function uses Trace Ratio Criterion [traceratio]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [traceratio] Feiping Nie, Shiming Xiang, Yangqing Jia, Changshui Zhang, and Shuicheng Yan. 2008. Trace ratio criterion for feature selection. In AAAI. 671–676.

Sparse learning based
----------------------------------------------

.. py:function:: EnrichRBP.featureSelection.llL21(features, labels, num_features=10)

    This function uses l2,1-norm regularization-based feature selection method [lll21]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [lll21] Jiliang Tang, Salem Alelyani, and Huan Liu. 2014. Feature selection for classification: A review. Data Classification: Algorithms and Applications (2014), 37.

.. py:function:: EnrichRBP.featureSelection.lsL21(features, labels, num_features=10)

    This function uses l2,1-norm regularization-based feature selection method [lsl21]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [lsl21] Jiliang Tang, Salem Alelyani, and Huan Liu. 2014. Feature selection for classification: A review. Data Classification: Algorithms and Applications (2014), 37.

Statistical based
---------------------------

.. py:function:: EnrichRBP.featureSelection.cfs(features, labels, num_features=10)

    This function uses correlation-based filter approach [CFS]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [CFS] Mark A. Hall and Lloyd A. Smith. 1999. Feature selection for machine learning: Comparing a correlation-based filter approach to the wrapper. In FLAIRS. 235–239.

.. py:function:: EnrichRBP.featureSelection.chiSquare(features, labels, num_features=10)

    This function uses Chi-Square Score [chisquare]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [chisquare] Huan Liu and Rudy Setiono. 1995. Chi2: Feature selection and discretization of numeric attributes. In ICTAI. 388–391.

.. py:function:: EnrichRBP.featureSelection.fScore(features, labels, num_features=10)

    This function uses F-score [fscore]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [fscore] Wright, Sewall. “The Interpretation of Population Structure by F-Statistics with Special Regard to Systems of Mating.” Evolution, vol. 19, no. 3, 1965, pp. 395–420.

.. py:function:: EnrichRBP.featureSelection.giniIndex(features, labels, num_features=10)

    This function uses Gini Index [giniindex]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [giniindex] C. W. Gini. 1912. Variability and mutability, contribution to the study of statistical distribution and relaitons. Studi Economico-Giuricici Della R (1912).

.. py:function:: EnrichRBP.featureSelection.tScore(features, labels, num_features=10)

    This function uses T-score [tscore]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
                .. class:: features:numpy array, shape (n_samples, n_features)

                        Sample feature matrix to be processed.

                .. class:: labels:numpy array, shape (n_samples, )

                        Class labels according to the input samples.

                .. class:: num_features:int, default=10

                        Number of selected features.

.. [tscore] John C. Davis and Robert J. Sampson. 1986. Statistics and Data Analysis in Geology. Vol. 646. Wiley. New York.
