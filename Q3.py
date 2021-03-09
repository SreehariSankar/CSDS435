from collections import defaultdict
from itertools import chain, combinations

def traceback(node, path):
    if node.parent is not None:
        path.append(node.name)
        traceback(node.parent, path)

def traceback_find(base, head):
    node = head[base][1]
    cond = []
    freq = []
    while node is not None:
        path = []
        traceback(node, path)
        if len(path) > 1:
            cond.append(path[1:])
            freq.append(node.count)
        node = node.next
    return cond, freq

class Node:
    def __init__(self, name, freq, parent):
        self.name = name
        self.count = freq
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, freq):
        self.count += freq

    def display(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in list(self.children.values()):
            child.display(ind + 1)


def create(lst, freq, min_sup):
    head = defaultdict(int)
    for idx, item in enumerate(lst):
        for item1 in item:
            head[item1] += freq[idx]
    head = dict((item, sup) for item, sup in head.items() if sup >= min_sup)
    for item in head:
        head[item] = [head[item], None]
    fp = Node('Null', 1, None)
    for idx, item in enumerate(lst):
        item = [item for item in item if item in head]
        item.sort(key=lambda itm: head[itm][0], reverse=True)
        currentNode = fp
        for item1 in item:
            currentNode = update_tree(item1, currentNode, head, freq[idx])
    return fp, head



def update(current, target, head):
    # /////////// HEAD UPDATER //////////
    if head[current][1] is None:
        head[current][1] = target
    else:
        currentNode = head[current][1]
        while currentNode.next is not None:
            currentNode = currentNode.next
        currentNode.next = target


def update_tree(current, node, head, freq):
    # ------ UPDATE FUNCTION ------------
    if current in node.children:
        node.children[current].increment(freq)
    else:
        new = Node(current, freq, node)
        node.children[current] = new
        update(current, new, head)

    return node.children[current]



def supp(test, lst):
    count = 0
    for item in lst:
        if set(test).issubset(item):
            count += 1
    return count


def get_rules(items, lst, min_confidence):
    rules = []
    for item in items:
        subsets = chain.from_iterable(combinations(item, r) for r in range(1, len(item)))
        itemSup = supp(item, lst)
        for s in subsets:
            confidence = float(itemSup / supp(s, lst))
            if confidence > min_confidence:
                rules.append([set(s), set(item.difference(s)), confidence])
    return rules


def algorithm(head, min_sup, pre, lst):
    # ======== SORT =================
    srt = [item[0] for item in sorted(list(head.items()), key=lambda p: p[1][0])]

    # ======= START =================
    for item in srt:
        NFS = pre.copy()
        NFS.add(item)
        lst.append(NFS)
        # ******** PATH FINDER *************
        cond_pat, freq = traceback_find(item, head)
        cond_tree, have_new = create(cond_pat, freq, min_sup)
        if have_new is not None:
            algorithm(have_new, min_sup, NFS, lst)


def FP(min_support=0.4, min_confidence=0.6):
    freq = [1, 1, 1, 1, 1]
    lst = [['Br', 'M'], ['Br', 'D', 'Be', 'E'], ['M', 'D', 'Be', 'C'], ['Br', 'M', 'D', 'Be'], ['Br', 'M', 'D', 'C']]
    min_sup = len(lst) * min_support
    fp, head = create(lst, freq, min_sup)
    f_items = []
    algorithm(head, min_sup, set(), f_items)
    rules = get_rules(f_items, lst, min_confidence)
    print(rules)


if __name__ == "__main__":
    FP()