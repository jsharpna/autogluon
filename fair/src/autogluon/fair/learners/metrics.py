"""Per group variants of existing autogluon metrics, and fairness metrics
"""
import collections.abc
import autogluon.core.metrics as metrics
import numpy as np

def per_group(metric):
    """ Constructor for per group variants of existing metrics.
     The constructed metric takes the standard arguments y_true and y_est plus discrete group labels
     The groups can either be a vector of discrete labels or a sequence of binary masks.
     """
    def new_metric(y_true,y_est, groups):
        if isinstance(groups,collections.abc.Sequence):
            masks = groups
        else:
            groups = np.asarray(groups)
            unique = np.unique(groups)
            masks = list()
            for g in unique:
                masks.append((groups==g))

        out=np.zeros(len(masks))
        for i,m in enumerate(masks):
            out[i]=metric(y_true[m],y_est[m])
        return out
    return new_metric

def fix_groups(per_group_metric,groups):
    """fixes the choice of groups so that the metric can be passed to Pareto frontier algorithms"""
    if isinstance(groups,collections.abc.Sequence):
        masks = groups
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        masks = list()
        for g in unique:
            masks.append((groups==g))

    def new_metric(y_true,y_est):
        return per_group_metric(y_true,y_est,masks)
    return new_metric



def overall_expected_prediction(y_true,y_pred):
    """Reports the average prediction discarding y_true"""
    return y_pred.mean(-1)

def overall_expected_label(y_true,y_pred):
    """Reports the average label discarding y_pred"""
    return y_true.mean(-1)

expected_prediction = per_group(overall_expected_prediction)
expected_label = per_group(overall_expected_label)

##Classification metrics
accuracy = per_group(metrics.accuracy)
balanced_accuracy = per_group(metrics.balanced_accuracy)
mcc = per_group(metrics.mcc)
roc_auc = per_group(metrics.roc_auc)
roc_auc_ovo_macro = per_group(metrics.roc_auc_ovo_macro)
average_precision = per_group(metrics.average_precision)
log_loss = per_group(metrics.log_loss)
log_loss = per_group(metrics.log_loss)
pac_score = per_group(metrics.pac_score)
quadratic_kappa = per_group(metrics.quadratic_kappa)
precision = per_group(metrics.precision)
precision_macro = per_group(metrics.precision_macro)
precision_micro = per_group(metrics.precision_micro)
precision_samples = per_group(metrics.precision_samples)
precision_weighted = per_group(metrics.precision_weighted)
recall = per_group(metrics.recall)
recall_macro = per_group(metrics.recall_macro)
recall_micro = per_group(metrics.recall_micro)
recall_samples = per_group(metrics.recall_samples)
recall_weighted = per_group(metrics.recall_weighted)
f1 = per_group(metrics.f1)
f1_macro = per_group(metrics.f1_macro)
f1_micro = per_group(metrics.f1_micro)
f1_samples = per_group(metrics.f1_samples)
f1_weighted = per_group(metrics.f1_weighted)
	
