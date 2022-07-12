import pandas as pd
import numpy as np
import metrics

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
                 return_data=True):
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
        from autogluon.core.utils import get_pred_from_proba_df
        labels=np.asarray(validation_data[self.pred.label])
        groups=validation_data[protected_attribute]
        y_pred_proba = self.pred.predict_proba(validation_data)
        y_pred = np.asarray(get_pred_from_proba_df(y_pred_proba, problem_type=self.pred.problem_type))

        pred_task = 'binary' #replace with task logic
        if pred_task == 'binary':
            out=self.evaluate_predictor_binary(labels==self.pred.positive_class,
                                              y_pred==self.pred.positive_class,y_pred_proba,groups)
            if return_data:
                return out

    

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

    def evaluate_predictor_binary(self,labels,y_pred, predictions_prob, groups):
        """Compute standard per-group  metrics for binary classification
        """
        predictions_prob = np.asarray(predictions_prob[self.pred.positive_class])
       
        collect= np.stack(( metrics.expected_label(labels,predictions_prob, groups),
                            metrics.expected_prediction(labels,predictions_prob, groups),
                            metrics.accuracy(labels, y_pred, groups),
                            metrics.balanced_accuracy(labels, y_pred, groups),
                            metrics.mcc(labels, y_pred, groups),
                            metrics.roc_auc(labels, predictions_prob,groups),
                            metrics.f1(labels, y_pred, groups), 
                            metrics.precision(labels, y_pred, groups),
                            metrics.recall(labels, y_pred, groups),
                        ))
        names=('Expected label','Expected Prediction','Accuracy','Balanced Accuracy', 'MCC', 'RoC AuC',
                 'F1', 'Precision', 'Recall'
                 )
        gap=(collect.max(-1)-collect.min(-1)).reshape(-1,1)
        #print(collect,gap)
        collect=np.hstack((collect,gap))
        out=pd.DataFrame(collect.T,index =groups.unique().tolist()+ ["Maximum difference",],columns=names)
        return out
        
