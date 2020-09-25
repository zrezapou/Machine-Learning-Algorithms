import pandas as pd
import scipy
from scipy.spatial import distance
import numpy as np
from numpy.random import rand
import random
import matplotlib.pyplot as plt
random.seed(100)


train_data_addr="./knn-dataset/trainData"
train_labels_addr="./knn-dataset/trainLabels"
test_data_addr="./knn-dataset/testData.csv"
test_labels_addr="./knn-dataset/testLabels.csv"
fold=10
# initial=320000
# lambdaa=350000
# step=1000
initial=1
lambdaa=17500
step=100
iterations=10

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

def create_fold(traindata, trainlabel, fold_indx):
    temp_data=traindata.copy()
    temp_label=trainlabel.copy()
    valid_dat=traindata[fold_indx]
    valid_labl=trainlabel[fold_indx]
    del(temp_data[fold_indx])
    del(temp_label[fold_indx])
    train_dat=pd.concat(temp_data, ignore_index=True)
    train_labl=pd.concat(temp_label, ignore_index=True)
    return [valid_dat, valid_labl, train_dat, train_labl]


def add_dim(X):
    temp=np.ones(((X.shape)[0], (X.shape)[1]+1))
    temp[:, 1:]=X
    X=temp
    return X

def sigmoid_np(x):
  return np.divide(1, (np.add(np.exp(-x), 1)))

def find_label(a):
    if a>=0.5:
        return 1
    else:
        return 0

def find_acc(w, data, label):
    counter=0
    x=np.array(data)
    x=add_dim(x)
    y=np.array(label)
    prob=sigmoid_np(np.matmul(x, w))
    for i in range (x.shape[0]):
        if (find_label(prob[i])) != y[i,0]:
            counter=counter+1
    accuracy=1-(counter/x.shape[0])
    return accuracy

def gradient(w, x, y, lamb):
    w_t=np.transpose(w)
    temp=np.subtract(sigmoid_np(np.matmul(w_t, x)), y)
    return np.add(np.matmul(x, np.transpose(temp)), lamb*w)


def hessian(w, x, lamb):
    w_t=np.transpose(w)
    sigma=sigmoid_np(np.matmul(w_t, x))
    sigma2=np.subtract(1, sigma)
    r=np.diag((np.multiply(sigma, sigma2))[0])
    h=np.add(np.matmul(np.matmul(x, r), np.transpose(x)), lamb*np.identity((w.shape)[0]))
    return h

def newtons_method(x, y, w0, iteration, lamb):
    w=w0
    for n in range (iteration):
        w = np.subtract(w, np.matmul(np.linalg.inv(hessian(w, x, lamb)), gradient(w, x, y, lamb)))
    return w


def find_w(data, label, lamb):
    x=np.array(data)
    x=add_dim(x)
    x=np.transpose(x)
    y=np.array(label)
    y=np.transpose(y)
    w0=np.random.rand(65,1)*0.02-0.01
    w=newtons_method(x, y, w0, iterations, lamb)
    return w

def logistic_regression2(dataset):
    valid_dat, valid_labl, train_dat, train_labl=dataset
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=100, solver='newton-cg').fit(train_dat, train_labl)
    acc=clf.score(valid_dat, valid_labl)
    return acc

def logistic_regression(dataset, fixed_lamb, mode):
    valid_dat, valid_labl, train_dat, train_labl=dataset
    sweep=np.linspace(initial, lambdaa, int((lambdaa-initial)/step+1))
    if mode=='train':
        acc=np.zeros((1,int((lambdaa-initial)/step+1)))
        j=0
        for lamb in sweep:
            w=find_w(train_dat, train_labl, lamb)
            accuracy=find_acc(w, valid_dat, valid_labl)
            acc[0,j]=accuracy
            j=j+1
    elif mode=='test':
        w=find_w(train_dat, train_labl, fixed_lamb)
        print("\n\n")
        print(np.transpose(w))
        print("\n\n")
        print(w.shape)
        acc=find_acc(w, valid_dat, valid_labl)
    return acc

def kfold(data, label):
    acc=np.zeros((fold,int((lambdaa-initial)/step+1)))
    # acc=np.zeros((fold,1))
    dataset_folded=[]
    for fold_indx in range(fold):
        dataset_folded=create_fold(data, label, fold_indx)
        acc[fold_indx,:]=logistic_regression(dataset_folded, 0, 'train')
        # acc[fold_indx,:]=logistic_regression2(dataset_folded)
    result=acc.mean(axis=0)
    idx=np.argmax(result)
    print(result)
    print("best lambda: {0}".format(initial+idx*step))
    print("Validation Accuracy: {0}".format(result[np.argmax(result)]))
    #plot_acc(result)
    return

def test(train_data, train_label, test_data, test_label):
    train_data=pd.concat(train_data, ignore_index=True)
    train_label=pd.concat(train_label, ignore_index=True)
    dataset=[pd.DataFrame(test_data), pd.DataFrame(test_label), pd.DataFrame(train_data), pd.DataFrame(train_label)]
    test_acc=logistic_regression(dataset, 6401,'test')
    # test_acc=logistic_regression2(dataset)
    print("test accuracy:{0}".format(test_acc))
    return

def plot_acc(avg_err):
    x=np.linspace(initial, lambdaa, num=int((lambdaa-initial)/step+1))
    fig=plt.figure(1)
    ax = fig.gca()
    #ax.set_xticks(np.arange(initial, lambdaa , 10))
    ax.set_xlim([initial, lambdaa])
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Lambda')
    plt.plot(x, avg_err, 'r')
    plt.grid()
    plt.show()

def main():
    train_data, train_label, test_data, test_label=read_data()
    kfold(train_data, train_label)
    test(train_data, train_label, test_data, test_label)
    return

if __name__=="__main__":
    main()
