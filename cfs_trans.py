import multiprocessing

from get_reward import GetReward

class Multi_p(object):
    def __init__(self, args, search_data, get_reward_ins):
        self.search_data = search_data
        self.args = args
        self.get_reward_ins = get_reward_ins
    def pool_produce(self, tmp):
        actions_trans = tmp[0]
        score_list, columns_name, fe_num = self.get_reward_ins.k_fold_score(
            self.search_data, actions_trans)
        tmp[0] = score_list
        tmp.append(self.get_reward_ins.rep_num)
        tmp.append(columns_name)
        tmp.append(fe_num)
        return tmp

    def multi_c(self, process_pool_num, tmp):
        process_pool = multiprocessing.Pool(process_pool_num)
        ids = process_pool.map(self.pool_produce, tmp)
        process_pool.close()
        process_pool.join()
        return ids