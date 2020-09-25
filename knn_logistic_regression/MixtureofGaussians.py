import pandas as pd
import scipy
from scipy.spatial import distance
import numpy as np
from numpy.random import rand
import random
import matplotlib.pyplot as plt
import math

train_data_addr="./knn-dataset/trainData"
train_labels_addr="./knn-dataset/trainLabels"
test_data_addr="./knn-dataset/testData.csv"
test_labels_addr="./knn-dataset/testLabels.csv"
fold=10

def read_data():
    train_data_lst=[]
    train_label_lst=[]
    test_data=pd.read_csv(test_data_addr, header=None)
    test_label=pd.read_csv(test_labels_addr, header=None)-5
    for i in range(fold):
        data=pd.read_csv(train_data_addr+str(i+1)+".csv", header=None)
        label=pd.read_csv(train_labels_addr+str(i+1)+".csv", header=None)-5
        train_data_lst.append(data)
        #ina listi az dataframe hast
        train_label_lst.append(label)
    return train_data_lst, train_label_lst, test_data, test_label

def unify(traindata, trainlabel):
    data=pd.concat(traindata, ignore_index=True)
    label=pd.concat(trainlabel, ignore_index=True)
    return [data, label]

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def calc_sigma(dat, labl, y_not, miu1, miu2):
    class1=np.zeros((1000,64))
    class2=np.zeros((1000,64))
    for j in range(1000):
        class1[j,:]=labl[j,:]*(dat[j,:]-miu1)
        class2[j,:]=y_not[j,:]*(dat[j,:]-miu2)
    s1=np.matmul(np.transpose(class1),class1)
    s2=np.matmul(np.transpose(class2),class2)
    return s1, s2

def parameters(dataset, total_data):
    data, label=dataset
    dat=np.array(data)
    labl=np.array(label)
    sum_of_labels=labl.sum(axis=0)
    dat_T=np.transpose(dat)
    x_Ty=np.transpose(np.matmul(dat_T,labl))
    temp=np.ones((1000,1))
    y_not=temp-labl
    x_Tynot=np.transpose(np.matmul(dat_T,y_not))
    pi=sum_of_labels/total_data
    N1=pi*total_data
    N2=total_data-N1
    miu1=x_Ty/N1
    miu2=x_Tynot/N2
    s1, s2=calc_sigma(dat, labl, y_not, miu1, miu2)
    sigma=(s1+s2)/total_data
    return pi, miu1, miu2, N1, N2, sigma

def MG(data, label):
    total_data=1000
    dataset_folded=[]
    dataset_folded=unify(data, label)
    pi, miu1, miu2, N1, N2, sigma=parameters(dataset_folded, total_data)
    print("\npi is :{0}".format(pi))
    print("\nu1 is :\n{0}".format(miu1))
    print("\nu2 is :\n{0}".format(miu2))
    print("\nsigma size:{0}".format(sigma.shape))
    l = len(sigma[0])
    print("\nDiagonal of sigma is :\n")
    print ([sigma[i][i] for i in range(l)])
    return pi, miu1, miu2, sigma

def label(a):
    if a>=0.5:
        return 1
    else:
        return 0

def acc(w, w_0, x, y):
    counter=0
    for i in range (x.shape[0]):
        a=w_0 + np.matmul(x[i,:], w)
        if (label(sigmoid(a))) != y[i,:]:
            counter=counter+1
    accuracy=1-(counter/x.shape[0])
    return accuracy

def calc_w_0(pi, miu1, sigma_inv, miu2):
    frst_term=-0.5*(np.matmul(np.matmul(miu1,sigma_inv),np.transpose(miu1)))
    scnd_term=0.5*(np.matmul(np.matmul(miu2,sigma_inv),np.transpose(miu2)))
    thrd_term=np.log(pi/(1-pi))
    w_0=frst_term + scnd_term+ thrd_term
    return w_0

def test(test_data, test_label, pi, miu1, miu2, sigma):
    data=np.array(test_data)
    label=np.array(test_label)
    sigma_inv=np.linalg.inv(sigma)
    w=np.matmul(sigma_inv,np.transpose(np.subtract(miu1,miu2)))
    w_0=calc_w_0(pi, miu1, sigma_inv, miu2)
    accuracy=acc(w, w_0, data, label)
    print("\ntest accuracy: {0}".format(accuracy))
    return

def main():
    train_data, train_label, test_data, test_label=read_data()
    pi, miu1, miu2, sigma = MG(train_data, train_label)
    test(test_data, test_label,  pi, miu1, miu2, sigma)
    return

if __name__=="__main__":
    main()
