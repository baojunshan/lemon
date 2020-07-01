## 1.监督学习 supervised

- linear
    - Perceptron
    - LinearRegression
    - LinearRegressionGD
    - RidgeRegression
    - RidgeRegressionGD
    - LassoRegressionGD
    - ElasticNetGD
    - LogisticRegression
    
- svm
    - Kernel
    - SVC
    - SVM
    - LinearSVC
    - LinearSVR

- tree
    - DecisionTreeClassifier
    - DecisionTreeRegression
    
- naive_bayes
    - GaussianNB
    - BernoulliNB
    - CategoricalNB
    - MultinomialNB

- neighbors
    - DistanceMetric
    - KDTree
    - BallTree
    - NearestNeighbors
    - NearestCentroid
    - KNeighborsClassifier
    - KNeighborsRegression
    - RadiusNeighborsClassifier
    - RadiusNeighborsRegression
    - LocalOutlierFactor
    
- crf
- hmm
- ensemble
    - AdaBoostClassifier
    - AdaBoostRegression
    - BaggingClassifier
    - BaggingRegression
    - GradientBoostingClassifier
    - GradientBoostingRegression
    - RandomForestClassifier
    - RandomForestRegression
    - HistGradientBoostingClassifier
    - HistGradientBoostingRegression
    - IsolationForest
    - StackingClassifier
    - StackingRegression
    - VotingClassifier
    - VotingRegression
- net




## 2.无监督学习 unsupervised

- Birch
- DBSCAN
- KMeans
- AgglomerativeClustering
- pagerank



## 3.半监督 semi_supervised

- label_propagation
- label_spreading
- louvain



## 3.数据处理 processing

- decomposition
    - FactorAnalysis
    - PCA
    - SVD
- preprocessing
    - Binarizer
    - FunctionTransformer
    - KBinsDiscretizer
    - LabelBinarizer
    - LabelEncoder
    - MaxAbsScaler
    - MinMaxScaler
    - OnehotEncoder
    - OrdinalEncoder
    - PolynomialFeatures
    - RobustScaler
    - StandardScaler
- impute
    - SimpleImputer
    - KNNImputer


## 4.特征选择与评估 feature_utils

- feature_selection
    - VarianceThreshold
    - chi2
    - IV

## 5.模型选择与评估 model_utils

- metrics
    - (classification)
    - accuracy
    - auc
    - precision
    - recall
    - classification_report
    - cohen_cappa_score
    - confusion_matrix
    - dcg_score
    - f1_score
    - fbeta_score
    - jaccard_score
    - precision_recall_curve
    - roc_auc_score
    - roc_curve
    - hamming_loss
    - hinge_loss
    - zero_one_loss
    - (regression)
    - max_error
    - mean_absolute_error
    - mean_square_error
    - mean_square_log_error
    - median_absolute_error
    - r2_score
    - (cluster)
    - silhouette_score
- model_selection
    - train_test_split
    - KFold
    - LeaveOneOut
    - ShuffleSpilt
    - TimeSeriesSplit
    - GridSearchCV
    - RandomizedSearchCV
- pipeline
    - FeatureUnion
    - Pipeline

