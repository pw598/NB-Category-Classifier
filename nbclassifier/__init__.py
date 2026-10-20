"""nbclassifier - hierarchical multinomial naive Bayes for product categories.

Modules
    paths       repo folders, timestamps, finding the latest model/evaluation
    bootstrap   installs NLTK on demand
    config      HyperParams, grid expansion, CV settings
    cleaning    rule-set driven description cleaning (Cleaner)
    data        SQL/file loading, blacklist, rare-path filter, dataset save/load
    vectorizers TF-IDF / count vectorizers (incl. order-insensitive bigrams)
    model       SparseMultinomialNB and HierarchicalNB (3 prediction modes)
    folds       ID-hash folds (shared with the NN repo), stratified, holdout
    predict     the prediction table shared by OOF and batch outputs
    crossval    checkpointed cross-validation / grid search
    evaluation  confidence buckets, violin and coverage plots
    artifacts   saving / loading model folders
    tracking    optional MLflow logging

Submodules are imported explicitly (e.g. `from nbclassifier import data`),
so importing the package itself is cheap and has no heavy dependencies.
"""

__version__ = "1.0.0"
