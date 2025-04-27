# -*- coding:utf-8 -*-
import numpy as np


class Attr:
    def __init__(self):
        self.attrName = ""
        self.arrtType = ""
        self.isClass = False
        self.nVals = 0
        self.strVals = []
        self.vMax = None
        self.vMin = None


def getDataSet(csvFile_Name):
    file = open(csvFile_Name)
    reader = file.readlines()
    dataMat1 = []
    labelMat1 = []
    attributes = []
    className = ""
    for line in reader:
        if line.find('@attribute') != -1:
            attr = Attr()
            attr.attrName = line.split(' ')[1]
            if line.find('{') != -1:
                attr.arrtType = "enum"
                attr.strVals = line[line.find('{') + 1:line.find('}')].split(',')
                attr.nVals = len(attr.strVals)
            elif line.find('[') != -1:
                attr.arrtType = "number"
                minMax = line[line.find('[') + 1:line.find(']')].split(',')
                attr.vMin = float(minMax[0])
                attr.vMax = float(minMax[1])
            attributes.append(attr)
        elif line.find('@output') != -1:
            className = line.split(' ')[1][:-1]
        elif line.find('@') == -1:
            break

    for i in range(len(attributes)):
        if attributes[i].attrName == className:
            attributes[i].isClass = True

    for lineTotal in reader:
        line = lineTotal.split(',')
        curLine = []
        numAttributes = len(attributes)
        if line[0].find('@') == -1:
            if line[-1].find('\n') != -1:
                line[-1] = line[-1][:-1]
            for i in range(numAttributes):
                if attributes[i].isClass != True:
                    if attributes[i].arrtType != "enum":
                        curLine.append(float(line[i]))
                    else:
                        for k in range(attributes[i].nVals):
                            if line[i].strip() == attributes[i].strVals[k].strip():
                                curLine.append(float(k))
                                break
                else:
                    for j in range(attributes[i].nVals):
                        if line[-1].strip() == attributes[i].strVals[
                            j].strip():
                            labelMat1.append(j)
                            break
            dataMat1.append(curLine)

    sampleCount = np.shape(labelMat1)[0]
    labels = np.zeros(sampleCount)
    for i in range(sampleCount):
        labels[i] = labelMat1[i]
    file.close()

    return np.array(dataMat1), labels, attributes


def minMaxNoralize(dataMat1, attributes):
    n_inst = np.shape(dataMat1)[0]
    n_dim = np.shape(dataMat1)[1]
    x = np.zeros((n_inst, n_dim))
    for i in range(len(attributes) - 1):
        if attributes[i].arrtType == "enum":
            attributes[i].vMax = float(attributes[i].nVals - 1)
            attributes[i].vMin = 0.0
        if attributes[i].vMax == None:
            attributes[i].vMax = np.min(dataMat1, axis=0)[i]
        if attributes[i].vMin == None:
            attributes[i].vMin = np.min(dataMat1, axis=0)[i]

        if (attributes[i].vMax - attributes[i].vMin) != 0.0:
            x[:, i] = (dataMat1[:, i] - attributes[i].vMin) / (attributes[i].vMax - attributes[i].vMin)
        else:
            x[:, i] = dataMat1[:, i]
    return x
