# -*- coding: utf-8 -*-

import os
# import random
import time
import logging
import copy
import pandas as pd
import numpy as np
from pipline_thread_2N_batch_singlevalue import Pipline
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer
from pathlib import Path
# from constant import Operation
from feature_engineering.model_base import ModelBase
from utility.base_utility import BaseUtility
from utility.model_utility import ModelUtility, sub_rae
from dataset_split import DatasetSplit
import json
import random
import copy
from feature_engineering.feature_filter import FeatureFilterMath
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")
from xgboost.sklearn import XGBClassifier, XGBRegressor


class GetReward(object):
    def __init__(self, args, do_onehot=False, nlp_feature=None):
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = Path(self.dataset_path).stem
        self.target_col = args.target_col
        self.task_type = args.task_type
        self.eval_method = args.eval_method
        self.continuous_col = args.continuous_col
        self.discrete_col = args.discrete_col
        self.nlp_feature = nlp_feature

        #self.fe_num_limit = args.fe_num_limit
        if 'f1_average' in vars(args).keys():
            self.f1_average = args.f1_average
        else:
            self.f1_average = None
        self.do_onehot = do_onehot

        self.rep_num = None
        self.train_fe_params = None
        self.dataset_split = DatasetSplit(args)

        self.sample_path = Path('data/sample') / self.dataset_name

        self.pipline_ins = \
            Pipline(self.continuous_col, self.discrete_col, self.do_onehot)

    def get_model(self):
        if self.args.model == "rf":
            return self.get_rf_model(None)
        elif self.args.model == 'lr':
            return self.get_lr_model()
        elif self.args.model == 'xgb':
            return self.get_xgb_model()

    def filter_math_select(self, operation_idx_dict, x_train, y_train):

        ffm = FeatureFilterMath(operation_idx_dict)
        ffm.var_filter(x_train, threshold=0)

        ori_fe_len = x_train.shape[1]

        # ffm.columns_duplicates()

        # if self.task_type == 'classifier':
        #     ffm.chi2_filter(x_train, y_train, p_threshold=0.01)
        # ffm.mic_filter(x_train, y_train, task_type=self.task_type,
        #                mic_threshold=0)

        ffm.columns_duplicates(x_train)
        ffm.columns_na(x_train)

        ffm.update_delete_res()
        all_delete_idx = ffm.delete_idx_list
        new_train_fes = np.delete(x_train, all_delete_idx, axis=1)

        delete_idx_dict = ffm.delete_idx_dict
        rep_num = len(delete_idx_dict['delete_var_idx']) + \
                  len(delete_idx_dict['delete_duplicates_idx'])
        self.rep_num = rep_num / ori_fe_len

        return new_train_fes, all_delete_idx

    # @staticmethod
    # def remove_duplication_identical(data):
    #     # data = data[:, data.std(axis=0) != 0]
    #     _, idx = np.unique(data, axis=1, return_index=True)
    #     y = data[:, np.sort(idx)]
    #     return y, np.sort(idx)

    def feature_pipline_train(self, actions_trans, data):
        # Form a new dataset through actions on train
        new_fes_shape, operation_idx_dict = self.pipline_ins.calculate_shape(
            actions_trans, data, target_col=self.target_col)

        new_train_fes, new_train_columns, train_label = self.pipline_ins.create_action_fes(
            actions=actions_trans, ori_dataframe=data,
            task_type=self.task_type, target_col=self.target_col, train=True)

        fe_params = self.pipline_ins.fes_eng.get_train_params()
        # print('fe_params', fe_params)
        # print('new_train_fes', new_train_fes)
        # print('operation_idx_dict', operation_idx_dict)
        if len(operation_idx_dict['ori_continuous_idx']):
            max_con = -1
        else:
            if len(operation_idx_dict['ori_continuous_idx']) == 0:
                max_con = -1
            else:
                max_con = max(operation_idx_dict['ori_continuous_idx'])
        operation_idx_dict['ori_discrete_idx'] = list(range(max_con+1,new_train_fes.shape[1]))
        return new_train_fes, new_train_columns, train_label, fe_params, operation_idx_dict

    def feature_pipline_infer(self, fe_params, actions_trans, data):
        # Form a new dataset through actions on test
        new_test_fes, new_test_columns, test_label = self.pipline_ins.create_action_fes(
            actions=actions_trans, ori_dataframe=data,
            task_type=self.task_type, target_col=self.target_col,
            train=False, train_params=fe_params)
        self.pipline_ins.fes_eng.clear_train_params()
        return new_test_fes, new_test_columns, test_label

    def get_lr_model(self):
        #
        if self.task_type == 'classifier':
            model = ModelBase.lr_classify()
        elif self.task_type == 'regression':
            model = ModelBase.lr_regression()
        else:
            logging.info(f'er')
            model = None
        return model

    def get_svm_model(self):
        # choose model
        if self.task_type == 'classifier':
            model = ModelBase.svm_liner_svc()
        elif self.task_type == 'regression':
            model = ModelBase.svm_liner_svr()
        else:
            logging.info(f'er')
            model = None
        return model

    def get_rf_model(self, hyper_param):
        #
        if self.task_type == 'classifier':
            model = ModelBase.rf_classify(self.args.seed)
        elif self.task_type == 'regression':
            model = ModelBase.rf_regeression(self.args.seed)
        else:
            logging.info(f'er')
            model = None
        if hyper_param is not None and model is not None:
            model.set_params(**hyper_param)
        return model

    def get_xgb_model(self):
        #
        if self.task_type == 'classifier':
            model = ModelBase.xgb_classify()
        elif self.task_type == 'regression':
            model = ModelBase.xgb_regression()
        else:
            logging.info(f'er')
            model = None
        return model

    def train_test_infer(self, train_data, test_data, action_trans,
                     math_select=True):
        train_data_copy = copy.deepcopy(train_data)
        test_data_copy = copy.deepcopy(test_data)

        res_tuple = self.feature_pipline_train(action_trans, train_data_copy)
        if res_tuple is None:
            return None
        new_train_fes, new_train_columns, train_label, fe_params, operation_idx_dict = res_tuple
        #
        self.train_fe_params = fe_params

        new_test_fes, new_test_columns, test_label = \
            self.feature_pipline_infer(fe_params, action_trans, test_data_copy)

        if math_select:
            # math, mic
            new_train_fes, all_delete_idx = self.filter_math_select(
                operation_idx_dict, new_train_fes, train_label)
            new_test_fes = np.delete(new_test_fes, all_delete_idx, axis=1)

        #add txt(nlp) feature
        if not (self.nlp_feature is None):
            new_train_fes = np.hstack((new_train_fes, self.nlp_feature[:train_data.shape[0]]))
            new_test_fes = np.hstack((new_test_fes, self.nlp_feature[train_data.shape[0]:]))

        model = self.get_model()
        model.fit(new_train_fes, train_label)
        self.predict_score_ans(new_test_fes, model)


    def k_fold_score(self, search_data, action_trans, hp_action=None,
                     math_select=True, is_base=False):
        #
        search_data_copy = copy.deepcopy(search_data)
        res_tuple = self.feature_pipline_train(action_trans, search_data_copy)
        if res_tuple is None:
            return None
        search_fes, search_columns, search_label, fe_params, operation_idx_dict = res_tuple



        # simple feature_select
        if math_select and (not is_base):
            # math, mic
            search_fes, all_delete_idx = self.filter_math_select(
                operation_idx_dict, search_fes, search_label)

        # feature check
        # Num = search_fes.shape[1]
        # model.fit(search_fes, search_label)
        # entroys = model.feature_importances_
        # rank = [index for index, value in sorted(list(enumerate(entroys)), key=lambda x: x[1], reverse=True)]
        # print(rank[0:Num])
        # ans = []
        # for x in rank[0:Num]:
        #     print(entroys[x])
        #     ans.append(x)
        # print(ans)


        #add txt(nlp) feature
        if not (self.nlp_feature is None):
            search_fes = np.hstack((search_fes, self.nlp_feature))

        if self.task_type == 'classifier':
            if self.eval_method == 'auc':
                scoring_name = "accuracy"
            elif self.eval_method == 'f1_score':
                scoring_name = "f1_micro"

            score_list = cross_val_score(self.get_model(), search_fes, search_label, scoring=scoring_name, cv=5)
            # if not self.args.coreset:
            #     score_list = cross_val_score(self.model, search_fes, search_label, scoring=scoring_name, cv=5)
            # else:
            #     score_list = cross_val_score(self.model, search_fes, search_label, scoring=scoring_name, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10))
        else:
            if self.eval_method == 'sub_rae':
                subrae = make_scorer(sub_rae, greater_is_better=True)
                # score_list = cross_val_score(model, search_fes, search_label, scoring=subrae, cv=5)
                if not self.args.coreset:
                    score_list = cross_val_score(self.get_model(), search_fes, search_label, scoring=subrae, cv=5)
                else:
                    score_list = cross_val_score(self.get_model(), search_fes, search_label, scoring=subrae,
                                                 cv=KFold(n_splits=5, shuffle=True, random_state=10))
            elif self.eval_method == 'rmse':
                score_list = -cross_val_score(self.get_model(), search_fes, search_label,
                                              scoring="neg_mean_squared_error", cv=5)
                for i in range(len(score_list)):
                    score_list[i] = np.sqrt(score_list[i])
            elif self.eval_method == 'mae':
                score_list = -cross_val_score(self.get_model(), search_fes, search_label,
                                              scoring="neg_mean_absolute_error",
                                              cv=5)

        # model = self.get_model()
        # if self.task_type == 'classifier':
        #     skf = StratifiedKFold(n_splits=5)
        # elif self.task_type == 'regression':
        #     skf = KFold(n_splits=5)
        # else:
        #     print('task_type error')
        #     skf = None
        #
        # score_list = []
        # for trn_idx, val_idx in skf.split(search_fes, search_label):
        #     train_data = search_fes[trn_idx, :]
        #     train_label = search_label[trn_idx]
        #     valid_data = search_fes[val_idx, :]
        #     valid_label = search_label[val_idx]
        #
        #     # Built model
        #     model.fit(train_data, train_label)
        #
        #     # Validate score on validation set
        #     valid_score = self.predict_score(valid_data, valid_label, model)
        #     score_list.append(valid_score)

        if is_base:
            return score_list
        else:
            return score_list, search_columns, search_fes.shape[1]

    def predict_score(self, data, label, model):
        #
        if self.eval_method == 'ks' or self.eval_method == 'auc':
            y_pred = model.predict_proba(data)[:, 1]
        else:
            y_pred = model.predict(data)

        # sava the predict outcomes
        # np.set_printoptions(suppress=True)
        # np.savetxt('ans3.csv', y_pred, delimiter=',')

        score = ModelUtility.model_metrics(
            y_pred, label, self.task_type, self.eval_method, self.f1_average)

        return score

    def predict_score_ans(self, data, model):
        #
        # if self.eval_method == 'ks':
        #     y_pred = model.predict_proba(data)[:, 1]
        # else:
        y_pred = model.predict(data)
        # y_pred = np.exp(y_pred)
        if self.task_type == 'classifier':
            fe = self.train_fe_params[4]
            label_map = fe[self.target_col]
            inv_map = dict((v, k) for k, v in label_map.items())
            y_pred = pd.Series(y_pred).map(inv_map).astype(int).values
        np.set_printoptions(suppress=True)
        np.savetxt(f'ans_{self.args.model}_{self.dataset_name}.csv', y_pred, delimiter=',')


if __name__ == '__main__':
    pass
