import random
import copy
import json
import logging

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, \
    r2_score, accuracy_score, mean_absolute_error
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from collections import OrderedDict
from utility.base_utility import BaseUtility
from sklearn.model_selection import GridSearchCV


def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    rae = np.sum([np.abs(y_hat[i] - y[i]) for i in range(len(y))]) / np.sum([np.abs(y_mean - y[i]) for i in range(len(y))])
    res = 1 - rae
    return res
# def rmspe(y, y_hat):
#     y = np.array(y).reshape(-1)
#     y_hat = np.array(y_hat).reshape(-1)
#     y_mean = np.mean(np.square((y-y_hat)/(y+0.0000001)))
#     return  np.sqrt(y_mean)

class ModelUtility:
    @classmethod
    def model_metrics(cls, y_pred, y_real, task_type, eval_method, f1_average=None):
        if task_type == 'classifier':
            if eval_method == 'f1_score':
                unique_target = np.unique(y_real.reshape(len(y_real)))
                if len(unique_target) > 2:
                    # average = 'macro'
                    average = 'micro'
                else:
                    if f1_average:
                        average = f1_average
                    else:
                        average = 'binary'
                score = f1_score(y_real, y_pred, average=average)
            elif eval_method == 'acc':
                score = accuracy_score(y_real, y_pred)
            elif eval_method == 'ks':
                # print('y_pred', y_pred)
                # print(pd.value_counts(y_real))
                fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
                score = max(abs(fpr-tpr))

            elif eval_method == 'auc':
                fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
                score = auc(fpr, tpr)
                # score = roc_auc_score(y_real, y_pred)
            elif eval_method == 'confusion_matrix':
                cm = confusion_matrix(y_real, y_pred)
                ac = (cm[0, 0] + cm[1, 1]) / \
                     (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
                sp = cm[1, 1] / (cm[1, 0] + cm[1, 1])
                score = 0.5 * ac + 0.5 * sp
                # print('cm', score, cm, ac, sp)
            else:
                logging.info(f'er')
                score = None
        elif task_type == 'regression':
            if eval_method == 'mse':
                score = mean_squared_error(y_real, y_pred)
            elif eval_method == 'r_squared':
                score = r2_score(y_real, y_pred)
            elif eval_method == 'mae':
                score = mean_absolute_error(y_real, y_pred)
            elif eval_method == 'sub_rae':
                score = sub_rae(y_real, y_pred)
            elif eval_method == 'rmse':
                score = np.sqrt(mean_squared_error(y_real, y_pred))
            else:
                logging.info(f'er')
                score = None
        else:
            logging.info(f'er')
            score = None
        return score



