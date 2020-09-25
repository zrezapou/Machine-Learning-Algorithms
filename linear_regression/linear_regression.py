import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt


train_input_addr="./regression-dataset/trainInput"
train_targets_addr="./regression-dataset/trainTarget"
test_input_addr="./regression-dataset/testInput.csv"
test_targets_addr="./regression-dataset/testTarget.csv"
fold=10
lambda_range=41


def read_data():
    train_input_lst=[]
    train_target_lst=[]
    test_input=pd.read_csv(test_input_addr, header=None)
    test_target=pd.read_csv(test_targets_addr, header=None)
    for i in range(fold):
        input=pd.read_csv(train_input_addr+str(i+1)+".csv", header=None)
        target=pd.read_csv(train_targets_addr+str(i+1)+".csv", header=None)
        train_input_lst.append(input)
        #ina listi az dataframe hast
        train_target_lst.append(target)
    return train_input_lst, train_target_lst, test_input, test_target

def create_fold(traininput, traintarget, fold_indx):
    temp_input=traininput.copy()
    temp_target=traintarget.copy()
    valid_input=traininput[fold_indx]
    valid_target=traintarget[fold_indx]
    del(temp_input[fold_indx])
    del(temp_target[fold_indx])
    train_input=pd.concat(temp_input, ignore_index=True)
    # print(valid_input.shape)
    train_target=pd.concat(temp_target, ignore_index=True)
    return [valid_input, valid_target, train_input, train_target]
def add_dim(X):
    temp=np.ones(((X.shape)[0], (X.shape)[1]+1))
    temp[:, 1:]=X
    X=temp
    return X
def find_w(X, Y, lambdaa):
    X_T=np.transpose(X)
    A=np.add(np.matmul(X_T, X), lambdaa*(np.identity((X.shape)[1])))
    # print(X_T.shape)
    # print(Y.shape)
    b=np.matmul(X_T, Y)
    # print("hello")
    w = np.linalg.solve(A, b)
    return w

def find_loss(w, X, Y):
    first=np.subtract(np.matmul(X, w), Y)
    loss=np.matmul(np.transpose(first),first)
    return loss[0]

def regression(dataset, lambdaa, mode):
    valid_input, valid_target, train_input, train_target=dataset
    valid_input=np.array(add_dim(valid_input))
    train_input=np.array(add_dim(train_input))
    valid_target=np.array(valid_target)
    train_target=np.array(train_target)
    if mode=='train':
        loss=np.zeros((1,lambdaa))
        # print(loss.shape)
        count=0
        for j in range(lambdaa):
            count=j/10
            w=find_w(train_input, train_target, count)
            lossval=find_loss(w, valid_input, valid_target)
            loss[0,j]=lossval
    elif mode=='test':
        w=find_w(train_input, train_target, lambdaa)
        loss=find_loss(w, valid_input, valid_target)
    return loss

def kfold(input, target):
    loss=np.zeros((10,lambda_range))
    dataset_folded=[]
    for fold_indx in range(fold):
        dataset_folded=create_fold(input, target, fold_indx)
        # print(dataset_folded[1])
        loss[fold_indx,:]=regression(dataset_folded, lambda_range, 'train')
    # print(acc)
    result=loss.mean(axis=0)
    idx=np.argmin(result)
    print(result)
    print("best lambda:{0}".format(idx/10))
    print("Validation Loss: {0}".format(result[np.argmin(result)]))
    plot_loss(result)
    return

def test(train_input, train_target, test_input, test_target):
    train_input=pd.concat(train_input, ignore_index=True)
    train_target=pd.concat(train_target, ignore_index=True)
    dataset=[pd.DataFrame(test_input), pd.DataFrame(test_target), pd.DataFrame(train_input), pd.DataFrame(train_target)]
    loss=regression(dataset, 1.3, 'test')
    return loss

def plot_loss(avg_err):
    x=np.around(np.linspace(0,4,41), decimals=1)
    fig=plt.figure(1)
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 4.01 , 0.5))
    ax.set_xlim([0, 4])
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Error')
    # ax.set_title('Lambda vs Error')
    plt.plot(x, avg_err, 'r')
    plt.grid()
    plt.show()

def main():
    train_input, train_target, test_input, test_target=read_data()
    kfold(train_input, train_target)

    loss=test(train_input, train_target, test_input, test_target)
    print(loss[0])
    #print(train_target[9])
    #print(train_target[9].shape)
    return

if __name__=="__main__":
    main()
