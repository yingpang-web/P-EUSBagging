#! /usr/bin/python
# -*- coding: utf-8 -*-
import functools
import math

import numpy as np
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

import readData
from pbil.optimizer import optimize


def calculate_instance_diversity(selectedSamplesSet):
    totalDistancesMatrix = np.zeros((subdataset_number, subdataset_number))
    for i in range(subdataset_number):
        for j in range(i + 1, subdataset_number, 1):
            subSetI = selectedSamplesSet[i][:]
            subsetJ = selectedSamplesSet[j][:]
            nearest_neighborsJ = NearestNeighbors(n_neighbors=1)
            nearest_neighborsI = NearestNeighbors(n_neighbors=1)
            totalDistances = 0.0
            while (len(subSetI) != 0):
                k = 0
                nearest_neighborsJ.fit(subsetJ)
                temp = []
                temp.append(subSetI[k])
                knnJ = nearest_neighborsJ.kneighbors(temp, return_distance=True)
                _disI = knnJ[0][0][0]
                _neighborIndexI = knnJ[1][0][0]
                nearest_neighborsI.fit(subSetI)
                temp2 = []
                temp2.append(subsetJ[_neighborIndexI])
                knnI = nearest_neighborsI.kneighbors(temp2, return_distance=True)
                _disJ = knnI[0][0][0]
                _neighborIndexJ = knnI[1][0][0]
                if _neighborIndexJ == k:
                    del (subSetI[k])
                    del (subsetJ[k])
                    totalDistances += _disI
                else:
                    del (subSetI[_neighborIndexJ])
                    del (subsetJ[_neighborIndexI])
                    totalDistances += _disJ
            totalDistancesMatrix[i][j] = totalDistances
    return np.sum(totalDistancesMatrix) * (2.0 / (subdataset_number * (subdataset_number - 1)))


def get_MinMaj_dateset(minorityClassLabel, majorityClassLabel, trn_x, trn_y):
    majorityIndex = np.where(trn_y == majorityClassLabel)
    majInstances = trn_x[majorityIndex]
    minorityIndex = np.where(trn_y == minorityClassLabel)
    minInstances = trn_x[minorityIndex]

    return minInstances, majInstances


def eval_fun(nums, bits):
    assert len(nums) == len(bits)
    datasets = []
    vec_len = len(nums)
    step = int(vec_len / subdataset_number)

    for i in range(0, vec_len, step):
        subdataset = []
        for j in range(step):
            if bits[i + j] == 1:
                subdataset.append(nums[i + j])
        datasets.append(subdataset)

    return calculate_instance_diversity(datasets)


def decode_dataset(nums, bits):
    assert len(nums) == len(bits)
    datasets = []
    vec_len = len(nums)
    step = int(vec_len / subdataset_number)
    for i in range(0, vec_len, step):
        subdataset = []
        for j in range(step):
            if bits[i + j] == 1:
                subdataset.append(nums[i + j])
        datasets.append(subdataset)
    return datasets


def determine_MinMaj_class_label(trn_y):
    if np.sum(trn_y == 1.0) >= np.sum(trn_y == 0.0):
        minorityClassLabel = 0
        majorityClassLabel = 1
    else:
        minorityClassLabel = 1
        majorityClassLabel = 0

    return minorityClassLabel, majorityClassLabel


def get_train_dataset(minInstances, optimize_result_datasets, minorityClassLabel, majorityClassLabel):
    selectedSamplesSet = []
    selectedSamplesLabelSet = []

    minInstances_number = len(minInstances)
    classLabel = []
    for j in range(minInstances_number):
        classLabel.append(majorityClassLabel)
    for i in range(minInstances_number):
        classLabel.append(minorityClassLabel)

    for k in range(subdataset_number):
        temp_maj_dataset = list(optimize_result_datasets[k])
        temp_maj_dataset.extend(minInstances)
        selectedSamplesSet.append(temp_maj_dataset)
        selectedSamplesLabelSet.append(classLabel)

    return selectedSamplesSet, selectedSamplesLabelSet


def get_ML_models(selectedSamplesSet, attritbutesTrn, selectedSamplesLabelSet):
    models = []
    for classifierIndex in range(subdataset_number):
        trn_x_nomorlize = readData.minMaxNoralize(np.array(selectedSamplesSet[classifierIndex]), attritbutesTrn)
        base_estimator = DecisionTreeClassifier(criterion="gini")
        base_estimator.fit(trn_x_nomorlize, selectedSamplesLabelSet[classifierIndex])
        models.append(base_estimator)
    return models


def test_ML_models(tst_x, attritbutesTrn, models, minorityClassLabel, majorityClassLabel, vote_matrix_std):
    pred_tst_y = []
    _pred_tst_y = []
    tst_x_nomorlize = readData.minMaxNoralize(tst_x, attritbutesTrn)
    for classifierIndex in range(subdataset_number):
        _pred_tst_y.append(models[classifierIndex].predict(tst_x_nomorlize))
    #  get predicted results of each classifier
    for testInstanceIndex in range(np.shape(tst_x)[0]):
        temp_pred_y = 0.0
        for classifierIndex in range(subdataset_number):
            if _pred_tst_y[classifierIndex][testInstanceIndex] == majorityClassLabel:
                temp_pred_y += -1.0 * vote_matrix_std[classifierIndex]
            else:
                temp_pred_y += vote_matrix_std[classifierIndex]

        if temp_pred_y >= 0.0:
            pred_tst_y.append(minorityClassLabel)
        else:
            pred_tst_y.append(majorityClassLabel)

    return pred_tst_y


def get_voting_matrix(trn_x, trn_y, attritbutesTrn, models):
    _pred_tst_y = []
    weak_models_number = len(models)
    training_instances_number = np.shape(trn_x)[0]
    tst_x_nomorlize = readData.minMaxNoralize(trn_x, attritbutesTrn)
    for classifierIndex in range(subdataset_number):
        _pred_tst_y.append(models[classifierIndex].predict(tst_x_nomorlize))

    #  declare a matrix that store the penalty scores for each sample of each classifier
    vote_binary_matrix = np.ones(shape=(weak_models_number, training_instances_number))

    for instanceIndex in range(training_instances_number):
        incorrect_count = 0
        for modelIndex in range(weak_models_number):
            if _pred_tst_y[modelIndex][instanceIndex] != trn_y[instanceIndex]:
                incorrect_count += 1
        for modelIndex in range(weak_models_number):
            if _pred_tst_y[modelIndex][instanceIndex] != trn_y[instanceIndex]:
                vote_binary_matrix[modelIndex][instanceIndex] = -1.0 * incorrect_count / weak_models_number
            else:
                vote_binary_matrix[modelIndex][instanceIndex] = 1.0 * incorrect_count / weak_models_number

    vote_matrix = np.sum(vote_binary_matrix, axis=1)
    # normalize
    vote_matrix_std_temp = (vote_matrix - vote_matrix.min(axis=0)) / (vote_matrix.max(axis=0) - vote_matrix.min(axis=0))
    # sigmoid
    # vote_matrix_std_temp = 1/(1+math.e**(-vote_matrix))

    vote_matrix_std = vote_matrix_std_temp / np.sum(vote_matrix_std_temp, axis=0)

    return vote_matrix_std


def run(trainFilePath, testFilePath):
    # 1.1 read data
    trn_x, trn_y, attritbutesTrn = readData.getDataSet(trainFilePath)
    tst_x, tst_y, attritbutesTst = readData.getDataSet(testFilePath)

    # 1.2 split the data into majority and minority sets
    minorityClassLabel, majorityClassLabel = determine_MinMaj_class_label(trn_y)
    minInstances, majInstances = get_MinMaj_dateset(minorityClassLabel, majorityClassLabel, trn_x, trn_y)
    temp_data = list(majInstances)

    # 1.3 lengthen the length of the majarity class
    for i in range(subdataset_number - 1):
        temp_data.extend(majInstances)
    data = np.array(temp_data)

    # 2. PBIL
    # 2.1 select the optimal sub sets
    number_of_one = len(minInstances)  # as constraint condition
    l = []
    optimize_result, best_fitness_result = optimize(0.02, 0.02, pop_size, 2, 2, len(data), optimisation_cycles,
                                                    functools.partial(eval_fun, data), subdataset_number, number_of_one,
                                                    vec_storage=None)

    # 2.2 decode the dataset
    optimize_result_datasets = decode_dataset(data, optimize_result)

    # 2.3 combine with the majority set to form the train set
    selectedSamplesSet, selectedSamplesLabelSet = get_train_dataset(minInstances, optimize_result_datasets,
                                                                    minorityClassLabel, majorityClassLabel)

    # 2.4 train the base classifiers
    models = get_ML_models(selectedSamplesSet, attritbutesTrn, selectedSamplesLabelSet)

    # 2.5 get the weight of base calssifiers
    vote_matrix_std = get_voting_matrix(trn_x, trn_y, attritbutesTrn, models)

    # 2.6 test model
    pred_tst_y = test_ML_models(tst_x, attritbutesTrn, models, minorityClassLabel, majorityClassLabel, vote_matrix_std)

    return pred_tst_y, tst_y, best_fitness_result


def evaluate_model(pred_y, y):
    report = metrics.classification_report(y, pred_y, digits=6)
    accuracy = metrics.accuracy_score(y, pred_y)

    tn, fp, fn, tp = metrics.confusion_matrix(y, pred_y).ravel()
    tnr = 1.0 * tn / (tn + fp)
    tpr = 1.0 * tp / (tp + fn)
    Gmean = math.sqrt(tnr * tpr)
    auc = (tnr + tpr) / 2

    f_res.write(report)
    f_res.write("Accuracy: " + str(accuracy) + "\n")
    f_res.write("tn: %d, fp: %d, fn: %d, tp: %d \n " % (tn, fp, fn, tp))
    f_res.write("AUC: " + str(auc) + "\n\n")
    f_res.write("Gmean: " + str(Gmean) + "\n")

    print(report)
    print("Accuracy: ", accuracy)
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print("AUC: ", auc)
    print("Gmean: ", Gmean)


def cross_validation(crossValidationK, ds_name, totalRuntime):
    pred_tst_y = []
    tst_y = []
    total_best_fitness_result = 0.0
    print("\n------dataset_name: %s ------" % (ds_name))
    f_log.write("\ndata set: %s" % (ds_name))
    f_res.write("\ndata set: %s" % (ds_name))
    for runtime in range(totalRuntime):
        print("\n------runtime: %s ------" % (runtime))
        f_log.write("\nruntime : %s\n" % (runtime))

        for j in range(crossValidationK):
            print("Cross Validation: Folder:%d..........." % (j + 1))
            trainFilePath = "E:/KEEL/Software-2016-05-17/dist/" + "data_temp/" + ds_name + "/" + ds_name + "-5-" + str(
                j + 1) + "tra.dat"
            testFilePath = "E:/KEEL/Software-2016-05-17/dist/" + "data_temp/" + ds_name + "/" + ds_name + "-5-" + str(
                j + 1) + "tst.dat"
            temp_pred_tst_y, temp_tst_y, best_fitness_result = run(trainFilePath, testFilePath)
            pred_tst_y.extend(temp_pred_tst_y)
            tst_y.extend(temp_tst_y)
            total_best_fitness_result += best_fitness_result
    averageInstanceDiversity = total_best_fitness_result / (crossValidationK * totalRuntime)
    print("averageInstanceDiversity: ", averageInstanceDiversity)
    f_res.write("averageInstanceDiversity: " + str(averageInstanceDiversity) + "\n")
    evaluate_model(pred_tst_y, tst_y)


if __name__ == '__main__':
    # define the log file
    f_log = open("result/pbil105192013.log", "a")
    f_res = open("result/result105192013.txt", "a")
    # the number of base classifiers
    global subdataset_number
    subdataset_number = 10
    # the number of iterations
    global optimisation_cycles
    optimisation_cycles = 100
    pop_size = 50  # the number of population
    f_res.write("\nsubdataset_number: %s" % (subdataset_number))
    f_res.write("\noptimisation_cycles: %s" % (optimisation_cycles))
    f_res.write("\npop_size: %s" % (pop_size))
    print("\nsubdataset_number: %s" % (subdataset_number))
    print("\noptimisation_cycles: %s" % (optimisation_cycles))
    print("\npop_size: %s" % (pop_size))

    crossValidationK = 5
    totalRuntime = 1
    data_sets = ["abalone9-18"]

    for ds_name in data_sets:
        cross_validation(crossValidationK, ds_name, totalRuntime)

    f_res.close()
    f_log.close()
