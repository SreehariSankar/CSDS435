import libsvmw.python.svm as svm
import libsvmw.python.svmutil as svmu
import libsvmw.python.commonutil as util

import sklearn.svm.svc as svc

import scipy as sp
import scipy.sparse as spr
import numpy as np
import matplotlib.pyplot as plt
import math


# LOAD INITIAL FULL TRAINING DATA AND TESTING DATA------------------------

# TRAINING DATA
y,x=util.svm_read_problem("DogsVsCats/DogsVsCats.train",return_scipy=True)
X=spr.csr_matrix.toarray(x)
# TEST DATA
y1,x1=util.svm_read_problem("DogsVsCats/DogsVsCats.test",return_scipy=True)
X1=spr.csr_matrix.toarray(x1)
print("Completed loading data")
# LOAD ENDS HERE----------------------------------------------------------


# ADABOOST
'''
:param 
number of iterations: K (Default is 10)
'''
print("Adaboost experiment")
w=np.zeros(X.shape[0])
w[int((X.shape[0])/2):]+=np.double(1/X.shape[0])
# print(w)
alpha_r=0
classifiers=[]
alpha=[]
for i in range(0,10):
    eps = np.double(0)
    alph = 0
    vector_machine=svmu.svm_train(w,y,X,'-h 0 -q')
    classifiers.append(vector_machine)
    p_labs, p_acc, p_vals = svmu.svm_predict(y, X, vector_machine)
    print(p_labs-y)
    acc=p_acc[0]/100
    # print("Accuracy for the ",i,"th iteration:",acc)
    for j in range(len(p_labs)):
        if(p_labs[j]!=y[j]):
            eps+=w[j]
    print("Epsilon value:",eps)
    alph=np.double(0.5*math.log((1-eps)/eps))
    Z=0
    for j in range(len(p_labs)):
        Z += w[j]*math.exp(-1*alph*p_labs[j]*y[j])
    for j in range(len(w)):
        w[j]=w[j]*math.exp(-1*alph*p_labs[j]*y[j])/Z
    print("Weights sum:",np.sum(w))
    # plt.plot(w,'r+')
    # plt.show()
    alpha.append(alph)
    alpha_r+=alph

i=classifiers[0]
p_labs, p_acc, p_vals = svmu.svm_predict(y1, x1, i,'-q')
t_labs=np.zeros(shape=np.array(p_labs).shape)
for i in range(len(classifiers)):
    p_labs, p_acc, p_vals = svmu.svm_predict(y1, x1, classifiers[i])
    t_labs+=np.array(p_labs)*(alpha[i]/alpha_r)

# t_labs/=len(classifiers)
acc=0
for i in range(len(t_labs)):
    if ((t_labs[i]>0) and (y1[i]==1)) or ((t_labs[i]<=0)and(y1[i]==-1)):
        acc+=1
print(acc/len(y1))