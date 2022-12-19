# -*- coding: utf-8 -*-
import copy

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

import feature_type_recognition
from get_reward import GetReward
from constant import NO_ACTION
import numpy as np


def test_action(_args, action_trans, with_gbdt=False):
    # read
    search_data = pd.read_csv(_args.dataset_path)
    keep_data = pd.read_csv(f'data/test_{_args.data}.csv')

    if (not _args.continuous_col) and (not _args.discrete_col):
        T = feature_type_recognition.Feature_type_recognition()
        T.fit(search_data)
        T.num.remove(_args.target_col)
        _args.continuous_col = T.num
        _args.discrete_col = T.cat

    features = _args.continuous_col + _args.discrete_col
    label = _args.target_col
    search_data = search_data[features + [label]]

    all_data = pd.concat([search_data, keep_data])
    nlp = None
    #'txt' features-> nlp
    for x in _args.txt_col:
        tfv = TfidfVectorizer(min_df=3,  max_features=11000, strip_accents='unicode',
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

        tfv.fit(all_data[x].tolist())
        X_tfidf = tfv.transform(all_data[x].tolist()).toarray()
        #X_tfidf = tfv.fit_transform(all_data[x].astype(str))
        if nlp is None:
            nlp = X_tfidf
        else:
            nlp = np.hstack((nlp, X_tfidf))

    #Test sets
    keep_data = keep_data[_args.continuous_col + _args.discrete_col]
    search_data = search_data[_args.continuous_col + _args.discrete_col + [_args.target_col]]

    #fill num
    for x in _args.continuous_col:
        keep_data[x].replace('?', np.nan, inplace=True)
        keep_data[x].replace("NA", np.nan, inplace=True)
        keep_data[x] = keep_data[x].astype(float)
        mean = np.nanmean(keep_data[x])
        keep_data[x].fillna(mean, inplace=True)
    for x in _args.discrete_col:
        keep_data[x].replace('?', np.nan, inplace=True)
        keep_data[x].replace("NA", np.nan, inplace=True)
        keep_data[x].fillna("*Unique", inplace=True)

    get_reward_ins = GetReward(_args, nlp_feature=nlp)
    # action
    if with_gbdt:
        get_reward_ins.train_test_infer(search_data, keep_data, action_trans)


if __name__ == '__main__':
    from main import parse_args
    args_ = parse_args()
    args_.model = 'xgb'
    #house prices
    actions = [[['MSSubClass', 'GarageArea', 'multiply', 'concat'], ['LotFrontage', 'LowQualFinSF', 'add', 'concat'],
      ['LotArea', 'PoolArea', 'multiply', 'concat'], ['OverallQual', 'EnclosedPorch', 'add', 'replace'],
      ['OverallCond', 'OpenPorchSF', 'multiply', 'replace'], ['YearBuilt', 'log', 'concat'],
      ['YearRemodAdd', '3SsnPorch', 'subtract', 'concat'], ['MasVnrArea', 'LotArea', 'add', 'replace'],
      ['BsmtFinSF1', 'MoSold', 'subtract', 'concat'], ['BsmtFinSF2', 'LotArea', 'divide', 'concat'],
      ['BsmtUnfSF', 'YearRemodAdd', 'add', 'concat'], ['TotalBsmtSF', 'EnclosedPorch', 'multiply', 'concat'],
      ['1stFlrSF', 'OverallCond', 'multiply', 'replace'], ['2ndFlrSF', 'EnclosedPorch', 'subtract', 'concat'],
      ['LowQualFinSF', 'BsmtFinSF1', 'multiply', 'replace'], ['GrLivArea', 'ScreenPorch', 'multiply', 'replace'],
      ['TotRmsAbvGrd', 'GarageYrBlt', 'multiply', 'concat'], ['GarageYrBlt', 'LotFrontage', 'divide', 'concat'],
      ['GarageArea', '3SsnPorch', 'divide', 'replace'], ['WoodDeckSF', 'TotalBsmtSF', 'multiply', 'replace'],
      ['OpenPorchSF', '3SsnPorch', 'subtract', 'replace'], ['EnclosedPorch', 'BsmtFinSF2', 'multiply', 'replace'],
      ['3SsnPorch', 'TotalBsmtSF', 'divide', 'concat'], ['ScreenPorch', '1stFlrSF', 'add', 'replace'],
      ['PoolArea', 'YearRemodAdd', 'subtract', 'replace'], ['MiscVal', 'LotFrontage', 'multiply', 'replace'],
      ['MoSold', 'OpenPorchSF', 'subtract', 'replace'], ['YrSold', 'EnclosedPorch', 'multiply', 'replace'],
      ['MSZoning', 'MiscFeature', 'combine', 'concat'], ['Street', 'Electrical', 'combine', 'replace'],
      ['Alley', 'Condition2', 'combine', 'concat'], ['LotShape', 'Foundation', 'combine', 'concat'],
      ['LandContour', 'BsmtQual', 'combine', 'concat'], ['Utilities', 'Condition2', 'combine', 'replace'],
      ['LotConfig', 'HouseStyle', 'combine', 'concat'], ['LandSlope', 'ExterQual', 'combine', 'concat'],
      ['Neighborhood', 'Condition1', 'combine', 'replace'], ['Condition1', 'Condition2', 'combine', 'replace'],
      ['Condition2', 'BsmtCond', 'combine', 'concat'], ['BldgType', 'Street', 'combine', 'concat'],
      ['HouseStyle', 'Utilities', 'combine', 'replace'], ['RoofStyle', 'GarageCars', 'combine', 'concat'],
      ['RoofMatl', 'BedroomAbvGr', 'combine', 'concat'], ['Exterior1st', 'LandContour', 'combine', 'replace'],
      ['Exterior2nd', 'KitchenQual', 'combine', 'replace'], ['MasVnrType', 'Functional', 'combine', 'concat'],
      ['ExterQual', 'BsmtFullBath', 'combine', 'replace'], ['ExterCond', 'Exterior1st', 'combine', 'replace'],
      ['Foundation', 'GarageCars', 'combine', 'replace'], ['BsmtQual', 'BldgType', 'combine', 'replace'],
      ['BsmtCond', 'Fireplaces', 'combine', 'concat'], ['BsmtExposure', 'Condition1', 'combine', 'concat'],
      ['BsmtFinType1', 'MSZoning', 'combine', 'replace'], ['BsmtFinType2', 'Condition2', 'combine', 'concat'],
      ['Heating', 'LotConfig', 'combine', 'replace'], ['HeatingQC', 'ExterQual', 'combine', 'replace'],
      ['CentralAir', 'LotConfig', 'combine', 'replace'], ['Electrical', 'SaleCondition', 'combine', 'concat'],
      ['BsmtFullBath', 'BsmtQual', 'combine', 'concat'], ['BsmtHalfBath', 'Fence', 'combine', 'replace'],
      ['FullBath', 'PavedDrive', 'combine', 'replace'], ['HalfBath', 'PavedDrive', 'combine', 'replace'],
      ['BedroomAbvGr', 'FullBath', 'combine', 'replace'], ['KitchenAbvGr', 'PavedDrive', 'combine', 'concat'],
      ['KitchenQual', 'LotConfig', 'combine', 'replace'], ['Functional', 'RoofStyle', 'combine', 'concat'],
      ['Fireplaces', 'BsmtFinType1', 'combine', 'replace'], ['GarageType', 'MasVnrType', 'combine', 'concat'],
      ['GarageFinish', 'Condition2', 'combine', 'concat'], ['GarageCars', 'PavedDrive', 'combine', 'replace'],
      ['GarageQual', 'Utilities', 'combine', 'replace'], ['GarageCond', 'HouseStyle', 'combine', 'replace'],
      ['FireplaceQu', 'Street', 'combine', 'replace'], ['PavedDrive', 'Fence', 'combine', 'concat'],
      ['PoolQC', 'Exterior1st', 'combine', 'replace'], ['Fence', 'BsmtCond', 'combine', 'replace'],
      ['MiscFeature', 'HalfBath', 'combine', 'replace'], ['SaleType', 'MasVnrType', 'combine', 'replace'],
      ['SaleCondition', 'Foundation', 'combine', 'concat']],
     [['MSSubClass', '1stFlrSF', 'multiply', 'replace'], ['LotFrontage', '3SsnPorch', 'subtract', 'concat'],
      ['OverallQual', 'PoolArea', 'add', 'concat'], ['OverallCond', 'TotRmsAbvGrd', 'multiply', 'concat'],
      ['YearBuilt', 'WoodDeckSF', 'multiply', 'concat'], ['YearRemodAdd', '1stFlrSF', 'add', 'concat'],
      ['BsmtFinSF1', 'MoSold', 'subtract', 'concat'], ['BsmtFinSF2', 'MSSubClass', 'multiply', 'replace'],
      ['BsmtUnfSF', '2ndFlrSF', 'add', 'concat'], ['TotalBsmtSF', 'EnclosedPorch', 'subtract', 'concat'],
      ['1stFlrSF', 'LotArea', 'multiply', 'replace'], ['2ndFlrSF', 'MiscVal', 'subtract', 'concat'],
      ['LowQualFinSF', 'OverallQual', 'add', 'replace'], ['GrLivArea', 'LowQualFinSF', 'add', 'concat'],
      ['TotRmsAbvGrd', 'MasVnrArea', 'multiply', 'replace'], ['GarageYrBlt', 'LowQualFinSF', 'divide', 'concat'],
      ['GarageArea', 'EnclosedPorch', 'add', 'concat'], ['WoodDeckSF', 'YearRemodAdd', 'multiply', 'replace'],
      ['OpenPorchSF', 'LotArea', 'divide', 'concat'], ['EnclosedPorch', 'GrLivArea', 'subtract', 'concat'],
      ['3SsnPorch', '2ndFlrSF', 'divide', 'concat'], ['ScreenPorch', 'EnclosedPorch', 'add', 'concat'],
      ['PoolArea', 'LowQualFinSF', 'divide', 'concat'], ['MiscVal', 'LotFrontage', 'divide', 'concat'],
      ['MoSold', 'YearRemodAdd', 'multiply', 'replace'], ['YrSold', 'EnclosedPorch', 'multiply', 'concat'],
      ['MSZoning', 'BsmtHalfBath', 'combine', 'concat'], ['Street', 'FireplaceQu', 'combine', 'replace'],
      ['Alley', 'MSZoning', 'combine', 'replace'], ['LotShape', 'RoofStyle', 'combine', 'concat'],
      ['LandContour', 'LotConfig', 'combine', 'replace'], ['Utilities', 'ExterQual', 'combine', 'concat'],
      ['LotConfig', 'LandSlope', 'combine', 'replace'], ['LandSlope', 'Utilities', 'combine', 'concat'],
      ['Condition1', 'LandSlope', 'combine', 'concat'], ['Condition2', 'LandSlope', 'combine', 'concat'],
      ['BldgType', 'LotShape', 'combine', 'replace'], ['RoofStyle', 'KitchenQual', 'combine', 'concat'],
      ['RoofMatl', 'BldgType', 'combine', 'replace'], ['Exterior1st', 'SaleType', 'combine', 'concat'],
      ['Exterior2nd', 'Neighborhood', 'combine', 'replace'], ['ExterQual', 'LotShape', 'combine', 'replace'],
      ['ExterCond', 'RoofMatl', 'combine', 'replace'], ['Foundation', 'Exterior1st', 'combine', 'concat'],
      ['BsmtQual', 'Street', 'combine', 'concat'], ['BsmtCond', 'GarageType', 'combine', 'replace'],
      ['BsmtExposure', 'Condition1', 'combine', 'concat'], ['BsmtFinType1', 'BsmtFullBath', 'combine', 'concat'],
      ['BsmtFinType2', 'Utilities', 'combine', 'replace'], ['Heating', 'SaleCondition', 'combine', 'concat'],
      ['HeatingQC', 'CentralAir', 'combine', 'concat'], ['CentralAir', 'LotConfig', 'combine', 'replace'],
      ['Electrical', 'LandContour', 'combine', 'replace'], ['BsmtFullBath', 'HouseStyle', 'combine', 'replace'],
      ['BsmtHalfBath', 'Condition2', 'combine', 'concat'], ['FullBath', 'GarageFinish', 'combine', 'concat'],
      ['HalfBath', 'HouseStyle', 'combine', 'concat'], ['BedroomAbvGr', 'Foundation', 'combine', 'concat'],
      ['KitchenAbvGr', 'Functional', 'combine', 'replace'], ['KitchenQual', 'Condition2', 'combine', 'concat'],
      ['Functional', 'Utilities', 'combine', 'concat'], ['Fireplaces', 'GarageFinish', 'combine', 'replace'],
      ['GarageType', 'Electrical', 'combine', 'concat'], ['GarageFinish', 'Condition2', 'combine', 'concat'],
      ['GarageCars', 'SaleCondition', 'combine', 'replace'], ['GarageQual', 'MSZoning', 'combine', 'replace'],
      ['GarageCond', 'KitchenQual', 'combine', 'replace'], ['FireplaceQu', 'ExterCond', 'combine', 'replace'],
      ['PavedDrive', 'HouseStyle', 'combine', 'concat'], ['PoolQC', 'HalfBath', 'combine', 'concat'],
      ['Fence', 'LotConfig', 'combine', 'replace'], ['MiscFeature', 'HalfBath', 'combine', 'replace'],
      ['SaleType', 'Foundation', 'combine', 'replace'], ['SaleCondition', 'Utilities', 'combine', 'replace']]]
    all_data = pd.read_csv(args_.dataset_path)
    if (not args_.continuous_col) and (not args_.discrete_col):
        T = feature_type_recognition.Feature_type_recognition()
        T.fit(all_data)
        T.num.remove(args_.target_col)
        args_.continuous_col = T.num
        args_.discrete_col = T.cat

    features = args_.continuous_col + args_.discrete_col
    label = args_.target_col
    all_data = all_data[features + [label]]

    #fill num
    for x in args_.continuous_col:
        all_data[x].replace('?', np.nan, inplace=True)
        all_data[x].replace("NA", np.nan, inplace=True)
        all_data[x] = all_data[x].astype(float)
        mean = np.nanmean(all_data[x])
        all_data[x].fillna(mean, inplace=True)
    for x in args_.discrete_col:
        all_data[x].replace('?', np.nan, inplace=True)
        all_data[x].replace("NA", np.nan, inplace=True)
        all_data[x].fillna("*Unique", inplace=True)

    #all_data = shuffle(all_data)

    get_reward_ins = GetReward(args_)


    # base_score = get_reward_ins.k_fold_score(all_data, [], is_base=True)
    # print(f'base_score:{np.mean(base_score)}')
    # print(f'base_score:{base_score}')

    new_score, columns_name, fe_num = get_reward_ins.k_fold_score(all_data, actions)
    print(f'new_score:{np.mean(new_score)}')
    print(f'new_score:{new_score}')


    test_action(args_, actions, with_gbdt=True)
