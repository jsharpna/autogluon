## This is the public API that should be exposed to the general user

from autogluon.core.metrics import METRICS
import warnings
from ..utils import post_fit
from ..models.classifier2ST import Classifier2ST


class XShiftDetector:
    """Detect a change in covariate (X) distribution between training and test, which we call XShift.  It can tell you
    if your training set is not representative of your test set distribution.  This is done with a Classifier 2
    Sample Test.

    Parameters
    ----------
    classifier_class: an AutoGluon predictor, such as autogluon.tabular.TabularPredictor
        The predictor that will be fit on training set and predict the test set

    label: str
        the Y variable that is to be predicted (if it appears in the train/test data then it will be removed)

    eval_metric: str
        the metric used for the C2ST, it must be one of the binary metrics from autogluon.core.metrics

    Methods
    -------
    fit: fits the detector on training and test covariate data

    json, summary: outputs the results of XShift detection
    - test statistic
    - pvalue (optional, if compute_pvalue=True in .fit)
    - detector feature importances
    - top k anomalous samples

    Usage
    -----
    >>> xshiftd = XShiftDetector(TabularPredictor, label='class')
    Fit the detector...
    >>> xshiftd.fit(X, X_test)
    Output the decision...
    >>> xshiftd.decision()
    Output the summary...
    >>> xshiftd.summary()
    """

    def __init__(self,
                 classifier_class,
                 label=None,
                 eval_metric='balanced_accuracy',
                 sample_label = 'i2vkyc0p64'):
        named_metrics = METRICS['binary']
        assert eval_metric in named_metrics.keys(), \
            'eval_metric must be one of [' + ', '.join(named_metrics.keys()) + ']'
        self.eval_metric = named_metrics[eval_metric]  #is this necessary?
        self.C2ST = Classifier2ST(classifier_class,
                                  sample_label=sample_label,
                                  eval_metric=self.eval_metric)
        if not label:
            warnings.warn('label is not specified, please ensure that X, X_test do not have the Y (label) '
                          'variable')
        self.label = label
        self._is_fit = False
        self.anomalies = None
        self.fi_scores = None
        self.teststat_thresh = None

    def fit(self, X, X_test, compute_fi = True, compute_anomaly_score = True, **kwargs):
        """Fit the XShift detector.

        Parameters
        ----------
        X, X_test: pd.DataFrame
            training dataframe and test dataframe
        compute_fi:
        compute_anomaly_score:
        **kwargs (optional): keyword arguments to .fit() for the classifier_class
        """
        assert 'xshift_label' not in X.columns, 'your data columns contain "xshift_label" which is used internally'

        if self.label:
            if self.label in X.columns:
                X = X.drop(columns=[self.label])
            if self.label in X_test.columns:
                X_test = X_test.drop(columns=[self.label])

        self.C2ST.fit((X, X_test), **kwargs)

        # Sample anomalies
        if compute_anomaly_score:
            as_top = self.C2ST.sample_anomaly_scores(how='top')
            as_top = as_top[[1]].rename(columns={1: 'xshift_test_proba'})
            self.anomalies = as_top.join(X_test)

        # Feature importance
        if self.C2ST.has_fi and compute_fi:
            self.fi_scores = self.C2ST.feature_importance()

        self._is_fit = True

    @post_fit
    def decision(self, teststat_thresh=0.55):
        """Decision function for testing XShift.  Uncertainty quantification is currently not supported.

        Parameters
        ----------
        teststat_thresh: float
            the threshold for the test statistic

        Returns
        -------
        One of ['detected', 'not detected']
        """
        # default teststat_thresh by metric
        self.teststat_thresh = teststat_thresh
        if self.C2ST.test_stat > teststat_thresh:
            return 'detected'
        else:
            return 'not detected'

    @post_fit
    def json(self):
        """output the results in json format
        """
        res_json = {
            'detection status': self.decision(),
            'test statistic': self.C2ST.test_stat,
        }
        if self.fi_scores is not None:
            res_json['feature importance'] = self.fi_scores
        if self.anomalies is not None:
            res_json['sample anomalies'] = self.anomalies
        return res_json

    @post_fit
    def summary(self, format="markdown"):
        """print the results to screen
        """
        assert format == 'markdown', 'Only markdown format is supported'
        if self.decision() == 'not detected':
            ret_md = (
                f"# Detecting distribution shift\n"
                f"We did not detect a substantial difference between the training and test X distributions."
            )
            return ret_md
        else:
            ret_md = (
                f"# Detecting distribution shift\n"
                f"We detected a substantial difference between the training and test X distributions,\n"
                f"a type of distribution shift.\n"
                f"\n"
                f"## Test results\n"
                f"We can predict whether a sample is in the test vs. training set with a {self.eval_metric.name} of\n"
                f"{self.C2ST.test_stat} (larger than the threshold of {self.teststat_thresh}).\n"
                f"\n"
            )
            if self.fi_scores is not None:
                fi_md = (
                    f"## Feature importances\n"
                    f"The variables that are the most responsible for this shift are those with high feature importance:\n"
                    f"{self.fi_scores.to_markdown()}"
                )
                return ret_md + fi_md
        return ret_md
