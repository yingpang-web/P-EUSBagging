#! /usr/bin/python
# -*- coding: utf-8 -*-

import random
import functools

import numpy as np


def get_num(proba):
    """
    Returns 0 or 1 depending on proba value
    :param proba: num between (but including) [0, 1] - probability of 1
    """
    # 生成一个随机数，这个随机数小于这个概率就返回1，否则返回0。可见，概率越大，返回1的可能性就越大。
    # 这就是一个轮盘赌算法
    if random.random() < proba:
        return 1
    return 0


def optimize(learn_rate, neg_learn_rate, pop_size, num_best_vec_to_update_from, num_worst_vec_to_update_from, vec_len,
             optimisation_cycles, eval_f, subdataset_number, number_of_one, eps=0.01,vec_storage=None):
    """

    :param learn_rate: rate of pushing the population vector (vec) towards each of the best individuals
    :param neg_learn_rate: similar to learn rate, but pushes the vector away from the worst individuals
    :param pop_size: num of individuals in population
    :param num_best_vec_to_update_from: how many best individuals will be used to update population vector
    :param num_worst_vec_to_update_from: how many worst individuals will be used to update population vector
    :param vec_len: length of the population vector
    :param optimisation_cycles: num of optimisation cycles
    :param eval_f: function for individual's fitness evaluation
    :param eps: population vector will be pushed away eps from extreme values (0, 1)
    :param vec_storage: storage for population vectors from each turns, should implement "append" method
    :return: best binary vector. If many vectors have the same fitnesses, returns the one, that appeared most early
    """
    #保证array的元素全部打印出来，为了存日志用
    np.set_printoptions(threshold=np.inf)
    #启动日志文件
    # f_log = open("result/pbil.log", "a")
    # vector initialisation
    # 向量初始化，长度就是基因位数，即数据集个数的长度，每个基因位数概率复制为0.5
    vec = np.full(vec_len, 0.5, dtype=float)

    # initialise population
    # 初始化种群，即根据种群个数初始化N个向量长度为基金位数的向量
    population = np.empty((pop_size, vec_len), dtype=int)
    scores = [None for _ in range(pop_size)] # 给每一向量一个初始化得分为None

    # initialise best result
    # 初始化一个最佳的结果为 0.0
    best_of_all = [-float("inf"), None]

    # store vec?
    # 判断是否要存储向量 （就是把之前历史的向量变化规律存下来）
    if vec_storage is not None:
        vec_storage.append(list(vec))

    # 迭代优化，optimisation_cycles为迭代次数
    for i in range(optimisation_cycles):
        print "optimisation_cycles: %s" % i
        # f_log.write("\noptimisation_cycles : %s" % (i))
        # solution vectors generation
        # 解决方案向量的生成
        for j in range(pop_size):
            # 对于每一个种群个体，即每一个染色体
            # print "chromosome number: %s" % j
            for k in range(vec_len):
                population[j][k] = get_num(vec[k])  # 根据概率去给每一个种群的每一个基因位设置值，注意这个概率的向量只有一个，多样性决绝于get_num里面的随机数
            # 升级约束条件，每一段都要满足1的个数
            # subdataset_number = 4
            step = int(vec_len / subdataset_number)
            # number_of_one = 3
            for n in range(0, vec_len, step):
                # print population[j][n:n + step]
                if sum(population[j][n:n+step]) < number_of_one:
                    # 1.判断缺几个
                    # 2.while循环生成几个相应的1
                    reverse_flag = True
                    reverse_num = 0
                    difference_value = number_of_one - sum(population[j][n:n+step])
                    while reverse_flag:
                        random_index = random.randint(n, n+step-1)
                        if population[j][random_index] == 0:
                            population[j][random_index] = 1
                            reverse_num += 1
                        if reverse_num == difference_value:
                            reverse_flag = False
                if sum(population[j][n:n+step]) > number_of_one:
                    # print sum(sub_population)
                    # 1.判断缺几个
                    # 2.while循环生成几个相应的1
                    reverse_flag = True
                    reverse_num = 0
                    difference_value = sum(population[j][n:n+step]) - number_of_one
                    while reverse_flag:
                        random_index = random.randint(n, n+step-1)
                        if population[j][random_index] == 1:
                            population[j][random_index] = 0
                            reverse_num += 1
                        if reverse_num == difference_value:
                            reverse_flag = False
                # print population[j][n:n + step]
            # print " population[j]",  population[j]
            # f_log.write("\npopulation[%s]: " % (j))
            # f_log.write(str(population[j]))
            scores[j] = eval_f(population[j])
        # best vectors selection
        # 根据适应值函数的得分，对种群进行从大到小的排序
        sorted_res = sorted(zip(scores, population), key=lambda x:x[0], reverse=True)
        # 选出最好的几个
        best = sorted_res[:num_best_vec_to_update_from]
        # f_log.write("\nbest:\n ")
        # f_log.write(str(best))
        # 选出最坏的几个
        worst = sorted_res[-num_worst_vec_to_update_from:]
        # f_log.write("\nworst:\n ")
        # f_log.write(str(worst))

        # update best_of_all
        # 更新最好的解决方案 （是不是因为best_of_all 是一个tuple类型，所以要赋两个值）
        if best_of_all[0] < best[0][0]:
            best_of_all = (best[0][0], list(best[0][1]))

        # update vector
        # 更新向量 更新每个基因位被选择为1的概率 因为v是tuple类型，v[1]就是这个向量
        # 最好的n个向量的每一个位数都增加一个概率，最坏的n个向量的每一个位数都减少一个概率
        for v in best:
            vec += 2 * learn_rate * (v[1] - 0.5)
        for v in worst:
            vec -= 2 * neg_learn_rate * (v[1] - 0.5)

        # vector correction if elements outside [0, 1] range
        # 如果概率超出了[0,1]范围，就对它进行修正
        for j in range(vec_len):
            if vec[j] < 0:
                vec[j] = 0 + eps
            elif vec[j] > 1:
                vec[j] = 1 - eps
        # f_log.write("\nvec:\n ")
        # f_log.write(str(vec))

        # store vec?
        # 判断是否要存储这个向量，存储的意义是保留这个向量的历史更新记录
        if vec_storage is not None:
            vec_storage.append(list(vec))

    return best_of_all[1],sorted_res[0][0]


