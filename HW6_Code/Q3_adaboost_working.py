import sklearn.svm as svm
import scipy as sp
import scipy.sparse as spr
import numpy as np
import matplotlib.pyplot as plt
import math
import libsvm.commonutil as util
import warnings
warnings.filterwarnings("ignore")


# LOAD INITIAL FULL TRAINING DATA AND TESTING DATA------------------------

# TRAINING DATA
y,x=util.svm_read_problem("DogsVsCats/DogsVsCats.train",return_scipy=True)
X=spr.csr_matrix.toarray(x)
print(X.shape,y.shape,"Train shapes")
# TEST DATA
y1,x1=util.svm_read_problem("DogsVsCats/DogsVsCats.test",return_scipy=True)
X1=spr.csr_matrix.toarray(x1)
print(X1.shape,y1.shape,"Test shapes")
print("Completed loading data")
# LOAD ENDS HERE----------------------------------------------------------

# ADABOOST
'''
:param 
number of iterations: K (Default is 10)
'''
print("Adaboost experiment")
w=np.zeros(X.shape[0])
w+=(1/X.shape[0])
# print(w)
alpha_r=0
classifiers=[]
alpha=[]

# NOTE: THIS CAN EVEN BE PLACED INSIDE THE LOOP. IT MAKES NO DIFFERENCE
vector_machine=svm.LinearSVC()

for i in range(0,10):
    eps = 0
    alph = 0
    #TRAIN
    vector_machine.fit(X,y,w)
    p_labs= vector_machine.predict(X)

    # UNCOMMENT TO SEE DIFFERENCE IN PREDICTION AND ACTUAL
    # print(p_labs-y)
    p_acc=vector_machine.score(X,y)
    classifiers.append(vector_machine)
    print("Accuracy for the ",i,"th iteration:",p_acc)

    # EPSILON

    for j in range(len(p_labs)):
        if(p_labs[j]!=y[j]):
            eps+=w[j]
    print("Epsilon value:",eps)

    # ALPHA
    alph=(0.5*math.log((1-eps)/eps))

    # Z
    Z=0
    for j in range(len(p_labs)):
        Z += w[j]*math.exp(-1*alph*p_labs[j]*y[j])

    # WEIGHT UPDATE
    for j in range(len(w)):
        w[j]=w[j]*math.exp(-1*alph*p_labs[j]*y[j])/Z

    print("Weights sum:",np.sum(w))
    # plt.plot(w,'r+')
    # plt.show()

    #STORE ALPHAS
    alpha.append(alph)
    alpha_r+=alph


# TESTING THE CLASSIFIERS =======================================================
i=classifiers[0]
p_labs = i.predict(X1)
t_labs=np.zeros(shape=np.array(p_labs).shape)
# t_labs is the VOTING ARRAY

for i in range(len(classifiers)):
    p_labs  = classifiers[i].predict(X1)
    t_labs+=np.array(p_labs)*(alpha[i]/alpha_r)
acc=0
print(t_labs)
for i in range(len(t_labs)):
    if ((t_labs[i]>=0) and (y1[i]==1)) or ((t_labs[i]<0)and(y1[i]==-1)):
        acc+=1
print("Accuracy_Test",acc/len(y1))


i=classifiers[0]
p_labs = i.predict(X)
t_labs=np.zeros(shape=np.array(p_labs).shape)
for i in range(len(classifiers)):
    p_labs  = classifiers[i].predict(X)
    t_labs+=np.array(p_labs)*(alpha[i]/alpha_r)
# t_labs/=len(classifiers)
acc=0
print(t_labs)
for i in range(len(t_labs)):
    if ((t_labs[i]>=0) and (y[i]==1)) or ((t_labs[i]<0)and(y[i]==-1)):
        acc+=1
print("Accuracy_Train",acc/len(y1))