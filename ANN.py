import numpy as np
import pickle
import time
"""
PLEASE REFER TO THE BACKPROP ALGORITHM GIVEN IN THE ANN CLASS TO SEE HOW I HANDLE THE BACKPROP.
THIS IS A VERY RUDIMENTARY ANN PROGRAM THAN CAN ONLY HANDLE TWO LAYERS.
"""


def load_dataset():
    with open("data", 'rb') as f:
        data = pickle.load(f)
    return data["training_images"], data["training_labels"], data["test_images"], data["test_labels"]

def test(ann, test_x):
    LL = [test_x, 0, 0]
    for l in range(len(ann)):
        LL[l + 1] = ann[l].FFD(LL[l])
    return LL[-1]

def SIG(x):
    return 1 / (1 + np.exp(-x))

def SIG__(x):
    return x * (1 - x)


class ANN:
    def __init__(self, NL, L_Num):
        self.l = L_Num
        self.ww = np.random.uniform(size=(NL[self.l + 1], NL[self.l]))
        self.bias = np.random.uniform(size=(NL[self.l + 1], 1))

        self.z = None


    def FFD(self, ip, BSZ=1):
        self.z = np.dot(self.ww, ip) / BSZ + self.bias
        return SIG(self.z)

    def Back_Prop(self, LL, cost, lr, BSZ):
        prev_cost = np.dot(self.ww.T, cost) * SIG__(LL[self.l]) 
        self.db = np.sum(cost, axis=1, keepdims=True)
        self.dw = np.dot(cost, LL[self.l].T) 
        self.ww += lr * self.dw / BSZ
        self.bias += lr * self.db / BSZ
        return prev_cost


def MAKE(layer_structure):

    ann = [0] * (len(layer_structure) - 1)
    for l in range(len(ann)):
        ann[l] = ANN(layer_structure, l)
    return ann


def train(ann, train_x, train_y, eps, LR, BSZ):
    n_train = len(train_x.T)
    x_BSZ = [train_x[:, n:n + BSZ] for n in np.arange(n_train // BSZ) * BSZ]
    op = np.zeros((len(ann[-1].bias), n_train))
    op[train_y, np.arange(n_train)] = 1
    op1 = [op[:, n:n + BSZ] for n in np.arange(n_train // BSZ) * BSZ]
    e = 0
    while e < eps:
        for xb, op2 in zip(x_BSZ, op1):
            LL = [xb, 0, 0]
            for l in range(len(ann)):
                LL[l + 1] = ann[l].FFD(LL[l], BSZ)
            cost = op2 - LL[-1]
            d_cost = cost * SIG__(LL[-1])
            for l in np.flip(np.arange(len(ann))):
                d_cost = ann[l].Back_Prop(LL, d_cost, LR, BSZ)
        e += 1


if __name__=='__main__':
    """
    README:
    THIS IS A VERY BASIC ANN PROGRAM THAT CAN ONLY HANDLE TWO LAYERS.
    DUE TO THE LACK OF NUMBER OF LAYERS AND NEURONS WITHIN THOSE LAYERS, 
    PLEASE DON'T EXPECT MUCH GROUNDBREAKING PERFORMANCE.
    THIS IS FOR THE EDIFICATION OF THE BACKPROP ALGORITHM
    """

    train_x, train_y, test_x, test_y = load_dataset()
    print("Done loading Data !!")
    print("NOTE:  Please modify the count of training and testing carefully.")

    train_count = 60000
    test_count = 10000
    train_x = (train_x[:train_count] / (np.max(train_x) - np.min(train_x))).T
    train_y = (train_y[:train_count]).T
    test_x = (test_x[:test_count] / (np.max(test_x) - np.min(test_x))).T

    architecture = [len(train_x), 40, 10]
    net = MAKE(architecture)

    print("Starting training")
    start = time.time()
    train(net, train_x, train_y, eps=150, LR=1, BSZ=15)
    end = time.time()
    print("Done training in: ", end - start // 60000, "mins")
    tests = np.zeros((10, test_count))
    tests[test_y, np.arange(test_count)] = 1

    OP = test(net, test_x)
    OP[OP > 0.75] = 1
    OP[OP <= 0.75] = 0

    print("Output vectors: ", OP)
    print("Actual OP vecs:", tests)

    # out = np.abs(OP - tests)
    # err = 0
    # for i in range(out.shape[1]):
    #     sm = np.sum(out[:,i])
    #     if sm!=0:
    #         err+=1
    # print(err/out.shape[1])
