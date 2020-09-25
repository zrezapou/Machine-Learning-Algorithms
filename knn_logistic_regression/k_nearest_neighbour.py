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
k_range=30

def break_tie():
    rdm=random.random()
    if rdm<=0.5:
        return -1
    elif rdm>0.5 and rdm<1:
        return +1
    else:
        print("error")

def decide_label(sum_of_labels):
    if sum_of_labels==0:
        predicted_label=break_tie()
    else:
        predicted_label=np.sign(sum_of_labels)
    return predicted_label

def read_data():
    train_data_lst=[]
    train_label_lst=[]
    test_data=pd.read_csv(test_data_addr, header=None)
    test_label=(pd.read_csv(test_labels_addr, header=None)-5.5)*2
    for i in range(fold):
        data=pd.read_csv(train_data_addr+str(i+1)+".csv", header=None)
        label=(pd.read_csv(train_labels_addr+str(i+1)+".csv", header=None)-5.5)*2
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

def find_distance(valid_dat, train_dat):
    return pd.DataFrame(distance.cdist(valid_dat.iloc[:,:],
     train_dat.iloc[:,:], metric='euclidean'))

def find_acc(size, valid_labl, train_labl, ary, k):
    counter=0
    for m in range((size)[0]):
        dp=ary.iloc[[m]]
        sorteddp=dp.sort_values(by=m, ascending=True, axis=1)
        header=sorteddp.columns.values
        labels9=(train_labl.iloc[header[0:k]]).values
        # print(labels9)
        newlabel=sum(labels9)
        # print(newlabel)
        predicted_label=decide_label(newlabel)
        if predicted_label != np.array(valid_labl.iloc[m]):
            counter=counter+1
    accuracy=((size)[0]-counter)/(size)[0]
    return accuracy

def knn(dataset, K, mode):
    valid_dat, valid_labl, train_dat, train_labl=dataset
    ary=find_distance(valid_dat, train_dat)
    if mode=='train':
        acc=np.zeros((1,K))
        for j in range(1, K+1):
            accuracy=find_acc(valid_dat.shape, valid_labl, train_labl, ary, j)
            acc[0,j-1]=accuracy
    elif mode=='test':
        acc=find_acc(valid_dat.shape, valid_labl, train_labl, ary, K)
    return acc

def kfold(data, label):
    acc=np.zeros((10,30))
    dataset_folded=[]
    for fold_indx in range(fold):
        dataset_folded=create_fold(data, label, fold_indx)
        # print(dataset_folded[1])
        acc[fold_indx,:]=knn(dataset_folded, k_range, 'train')
    # print(acc)
    result=acc.mean(axis=0)
    idx=np.argmax(result)
    print(result)
    print("best K:{0}".format(idx+1))
    print("Validation Accuracy: {0}".format(result[np.argmax(result)]))
    plot_acc_k(result, 30)
    return

def test(train_data, train_label, test_data, test_label):
    train_data=pd.concat(train_data, ignore_index=True)
    train_label=pd.concat(train_label, ignore_index=True)
    dataset=[pd.DataFrame(test_data), pd.DataFrame(test_label), pd.DataFrame(train_data), pd.DataFrame(train_label)]
    # print(dataset[3].shape)
    test_acc=knn(dataset, 19, 'test')
    print("test accuracy:{0}".format(test_acc))
    return
def plot_acc_k(avg_err, k):
    x=np.linspace(1, k, num=k)
    fig=plt.figure(1)
    ax = fig.gca()
    ax.set_xticks(np.arange(1, k , 3))
    ax.set_xlim([1, k])
    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs K')
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
