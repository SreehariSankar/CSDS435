from __future__ import print_function
import numpy as np
import pandas as pd
import copy

# def entropy(ff):
#     # INFORMATION GAIN
#     print("Using IG")
#     freq_ = ff[np.array(ff).nonzero()[0]]
#     prob_ = freq_ / float(freq_.sum())
#     return -np.sum(prob_ * np.log(prob_))

def entropy(ff):
    # GINI INDEX
    print("Using GINI index")
    freq_ = ff[np.array(ff).nonzero()[0]]
    prob_ = (freq_ / float(freq_.sum()))**2
    return 1 - np.sum(prob_)

# def entropy(ff):
#     # GAIN RATIO
#     print("Using Gain ratio")
#     freq_ = ff[np.array(ff).nonzero()[0]]
#     prob_ = (freq_ / float(freq_.sum())) ** 2
#     return -1*(np.sum(prob_ * np.log(prob_)))

def convert_(x,y):
    #  Try out a binary split for the continuous attr
    mn = min(x)
    mx = max(x)
    entropy_val = 100
    mark = 0
    for thresh in range((mn*10)+1, mx*10, (mx-mn)):
        thresh = thresh/10
        x0 = copy.deepcopy(x)
        x0[x0<=thresh]=0
        x0[x0>thresh]=1
        ppx1, ppxn, pnx1, pnxn = 0,0,0,0
        x1, xn = 0,0
        for j in range(len(x)):
            if y[j] == 'no' and x0[j]==1:
                pnx1+=1
                x1+=1
            elif y[j] == 'no' and x0[j]==0:
                pnxn+=1
                xn+=1
            elif y[j] == 'yes' and x0[j]==0:
                ppxn+=1
                xn+=1
            elif y[j] == 'yes' and x0[j]==1:
                ppx1+=1
                x1+=1
        if x1 == 0:
            x1+=1
        if xn==0:
            xn+=1
        ppx1=ppx1/x1
        ppxn=ppxn/xn
        pnx1=pnx1/x1
        pnxn=pnxn/xn
        H = ((x1/len(x)) * (-1 * ((ppx1 * np.log(ppx1) + pnx1 * np.log(pnx1))))) + ((xn / len(x)) * (-1 * ((ppxn * np.log(ppxn) + pnxn * np.log(pnxn)))))
        if H<entropy_val:
            entropy_val=H
            mark = thresh
    print(entropy_val, mark)
    return mark

class TNode(object):
    def __init__(self, dst=0, idx=None, entropy=0,chl=[]):
        self.idx = idx
        self.entropy = entropy
        self.dst = dst
        self.spt = None
        self.chl = chl
        self.order = None
        self.label = None

    def set_pr(self, spt, order):
        self.spt = spt
        self.order = order

    def setl(self, label):
        self.label = label


class DTree(object):
    def __init__(self, max_depth=100):
        self.root = None
        self.max_depth = max_depth
        self.count = 0

    def fit(self, data, target):
        self.count = data.count()[0]
        self.data = data
        self.attr = list(data)
        self.target = target
        self.L = target.unique()

        idx = range(self.count)
        self.root = TNode(idx=idx, entropy=self.ETR(idx), dst=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.dst < self.max_depth or node.entropy < self.mg:
                node.chl = self.SPLT(node)
                if not node.chl:
                    self.SL(node)
                queue += node.chl
            else:
                self.SL(node)

    def ETR(self, idx):
        if len(idx) == 0:
            return 0
        idx = [i + 1 for i in idx]
        ff = np.array(self.target[idx].value_counts())
        return entropy(ff)


    def SL(self, node):
        tid = [i + 1 for i in node.idx]
        node.setl(self.target[tid].mode()[0])
        
    def pre(self, new_data):
        N = new_data.count()[0]
        L = [None] * N
        for n in range(N):
            x = new_data.iloc[n, :]
            node = self.root
            while node.chl:
                node = node.chl[node.order.index(x[node.spt])]
            L[n] = node.label

        return L

    def SPLT(self, node):
        idx = node.idx
        bg = 0
        bs = []
        ba = None
        order = None
        sub_data = self.data.iloc[idx, :]
        for i, att in enumerate(self.attr):
            values = self.data.iloc[idx, i].unique().tolist()
            if len(values) == 1:
                continue
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id - 1 for sub_id in sub_ids])

            if min(map(len, splits)) < 2: 
                continue

            I_res = 0
            for split in splits:
                I_res += len(split) * self.ETR(split) / len(idx)
            gain = node.entropy - I_res
            if gain < 1e-3:
                continue
            if gain > bg:
                bg = gain
                bs = splits
                ba = att
                order = values
        node.set_pr(ba, order)
        child_nodes = [TNode(idx=split,
                                entropy=self.ETR(split), dst=node.dst + 1) for split in bs]
        return child_nodes

    



if __name__ == "__main__":
    df = pd.read_csv('data.csv')

    m = convert_(df.iloc[:,2], df.iloc[:,-1])
    for i in df.index:
        if df.at[i, 'temperature']>m:
            df.at[i, 'temperature']=1
        else:
            df.at[i, 'temperature'] = 0

    m1 = convert_(df.iloc[:,3], df.iloc[:,-1])
    for i in df.index:
        if df.at[i, 'humidity']>m1:
            df.at[i, 'humidity']=1
        else:
            df.at[i, 'humidity'] = 0

    X = df.iloc[:-1, 1:-1]
    y = df.iloc[:-1, -1]
    tree = DTree(max_depth=3)
    tree.fit(X, y)
    out = tree.pre(df.iloc[-1:,1:-1])
    print(out)
    # out = tree.pre(X)
    # acc = 0
    # for i in range(len(out)):
    #     if out[i]==y[i]:
    #         acc+=1
    # print("Accuracy: ",acc/len(out))