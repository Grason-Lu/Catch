from feature_engineering.model_base import ModelBase
import pandas as pd


from get_reward import GetReward
from utility.model_utility import ModelUtility

def reduce_scale(all_data, args):

    #reduce sample size
    sample_size = min(args.core_size, all_data.shape[0])
    features = args.continuous_col + args.discrete_col
    label = args.target_col
    all_data = all_data[features + [label]]
    core_data = all_data.copy(deep = True)

    #preprocessing core_data
    get_reward_ins = GetReward(args)
    res_tuple = get_reward_ins.feature_pipline_train([], core_data)
    core_train, core_columns, core_label, fe_params, operation_idx_dict = res_tuple

    core_data = pd.concat([pd.DataFrame(core_train), pd.DataFrame(core_label)], axis=1)
    core_data.columns = features + [label]
    limit = 100

    ori_score = -100000
    current_size = 0
    step_size = sample_size // 10
    time = 0
    l1 = 0.3
    l2 = 0.7
    if args.eval_method == 'rmse' or args.eval_method == 'mse' or args.eval_method == 'mae':
        l1 = -l1
        l2 = -l2
    if args.task_type == 'regression':
        model = ModelBase.rf_regeression()
    else:
        model = ModelBase.rf_classify()
    temp = None
    new_data = None
    while (current_size < sample_size and time < limit):
        time += 1
        res = core_data.sample(step_size, axis = 0, replace = False)
        if new_data is None:
            new_data = res.copy(deep = True)
        else:
            temp = new_data.copy(deep = True)
            new_data = pd.concat([res, new_data], axis=0)

        # get score
        model.fit(new_data[features], new_data[label])
        y_pred = model.predict(core_data[features])
        y_pred_remain = model.predict(core_data.drop(new_data.index)[features])

        if 'f1_average' in vars(args).keys():
            f1_average = args.f1_average
        else:
            f1_average = None
        score_all = ModelUtility.model_metrics(
            y_pred, core_data[label].values, args.task_type, args.eval_method, f1_average)
        score_remain = ModelUtility.model_metrics(
            y_pred_remain, core_data.drop(new_data.index)[label].values, args.task_type, args.eval_method, f1_average)
        #print(str(score_all)+','+str(score_remain))
        score = score_all * l1 + score_remain * l2
        print(str(score)+','+str(ori_score))
        # bad re-back
        if score < ori_score:
            new_data = temp.copy(deep = True)
        else:
            current_size += step_size
            ori_score = score

    new_data.reset_index(drop=True, inplace=True)
    return new_data

