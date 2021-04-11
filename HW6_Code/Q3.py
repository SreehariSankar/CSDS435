import libsvm.svm as svm
import libsvm.svmutil as svmu
import libsvm.commonutil as util
import scipy as sp
import scipy.sparse as spr
import matplotlib.pyplot as plt
import numpy as np
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



# CREATE A RANDOM DISTRIBUTION OF +VES AND -VES---------------------------
full=np.column_stack((X,y))
np.random.shuffle(full)
np.random.shuffle(full)
np.random.shuffle(full)
X=full[:,:(full.shape[1]-1)]
y=full[:,(full.shape[1]-1):]
y=np.ndarray.flatten(y)
print("Completed shuffling")
# RANDOMIZER ENDS HERE----------------------------------------------------


# 10-FOLD CV PART==================================================================

# I MANUALLY CODED THE 10FOLD CV BEFORE REALIZING THAT CV WAS A LIBRARY OPTION
# THIS WORKS BUT THE OTHER WAY IS EASIER
'''
print("10 fold CV experiment")
k=int(X.shape[0]/10)
avg=0
for i in range(0,10):
    test_x = X[i * k:(i + 1) * k, :]
    lst = np.arange(i * k, (i + 1) * k)
    train_x = np.delete(X, lst, axis=0)
    test_y = y[i * k:(i + 1) * k]
    lst = np.arange(i * k, (i + 1) * k)
    train_y = np.delete(y, lst)
    vector_machine = svmu.svm_train(train_y, train_x, '-t 1 -h 0')
    p_labs, p_acc, p_vals = svmu.svm_predict(test_y, test_x, vector_machine)
    avg+=p_acc[0]
    print(i)
print("Average accuracy:",avg/10)
'''
# MANUAL CV 10 FOLD ENDS HERE --------------------------------------------------

# 10 FOLD CV USING NORMAL OPTIONS------------------------------------ RUN THIS

print("10 fold CV experiment")
acc=svmu.svm_train(y,X,'-h 0 -v 10 -t 0 ')
print("Accuracy:",acc)

# 10 FOLD CV USING NORMAL OPTIONS------------------------------------ RUN THIS

# CV ENDS HERE======================================================================


# FULL DATA PREDICTION

print("Full data prediciton experiment")
vector_machine=svmu.svm_train(y,X,'-t 1 -h 0 -d 5')
p_labs, p_acc, p_vals = svmu.svm_predict(y1, X1, vector_machine)
print(p_acc)