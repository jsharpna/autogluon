class fairPareto:
    """Calculate the Pareto frontier for aggregated group-wise metrics.  This is done by changing the thresholds for
    each label class to achieve improved metrics in a Pareto optimal sense.

    Parameters
    ----------
    classifier: an AutoGluon Predictor
        the classifier for which we will be computing the Pareto frontier

    Methods
    -------
    fit: compute the Pareto frontier

    summarize: output the Pareto frontier as a table, json, or text

    plot: plot the Pareto frontier

    select: select the model by model id
    """

    def __init__(self, classifier):
        self._classifier = classifier

    def fit(self,
            test_data,
            protected_attribute,
            aggregatated_metrics):
        """Compute the Pareto frontier of protected group specific metrics for a classifier.

        Parameters
        ----------
        test_data: pd.DataFrame
            test data (what would be used in predict())

        protected_attribute: str or pd.Series
            the protected attribute (categorical) either as
            - a column name in test_data (str)
            - or the column variable with same index as test_data (pd.Series)

        aggregated_metrics: list of tuples
            list of (aggregation, metric) pairs
            - aggregation: method for combining a metric for each protected group into a single metric
            - metric: the metric used (must be valid classification metrics)
        """
        pass

    def summarize(self, how='table'):
        """Summarize the Pareto frontier

        Returns
        -------
        how = "table": a DataFrame with rows as models (indexed by model_id) and columns as aggregation, metric pair
        """
        pass

    def plot(self):
        """Plot the Pareto frontier for aggregation, metric pairs
        """

    def select(self, model):
        """Select a model by model_id

        Returns
        -------
        Classifier
        """