class Fairness:
    """Assess and mitigate the fairness of a predictor post-fit by computing and adjusting protected group specific metrics.

    Parameters
    ----------
    predictor: an AutoGluon predictor
        the predictor that we will be evaluating for fairness, it should have already been fit to
        training data

    Methods
    -------
    evaluate: evaluate and summarize fairness metrics

    mitigate: mitigate the unfairness of the predictor by adjusting the groupwise predictions

    """

    def __init__(self,
                 pred):
        self.pred = pred
        self._called_eval = False

    def evaluate(self,
                 validation_data,
                 protected_attribute,
                 return_data=False):
        """Evaluate fairness metrics and return summary

        Parameters
        ----------
        validation_data: pd.DataFrame
            validation data (what would be used in predict())

        protected_attribute: str or pd.Series
            the protected attribute (categorical) either as
            - a column name in validation_data (str)
            - or a categorical variable with same index as validation_data (pd.Series)

        return_data: bool
            if True then return the summary and the data in ??? format
        """
        self._called_eval = True

        pred_task = 'binary' #replace with task logic
        if pred_task == 'binary':
            fair_metrics = self._binary_metrics(validation_data, protected_attribute)

        if return_data = True:
            return fair_metrics

    def mitigate(self,
                 policy='pareto'):
        """
        Mitigate the unfairness of the predictor by adjusting the groupwise predictions.

        Pareto policy: Calculate the Pareto frontier for aggregated group-wise metrics.  This is done by changing the thresholds for
        each label class to achieve improved metrics in a Pareto optimal sense.

        Parameters
        ----------
        policy: str
            The policy used to mitigate, one of ['pareto']
        """
        pass

    def _binary_metrics(self,
                         validation_data,
                         protected_attribute):
        """Compute the fairness metrics for binary classification
        """
        pass
