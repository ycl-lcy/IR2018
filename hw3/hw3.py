import os, sys
import numpy as np
import operator
from collections import defaultdict
from nltk.stem.porter import *
from math import log

pc = [log(1.0/13.0) for i in range(13)]
with open("training.txt", "r") as f:
    train = f.read()
    train = train.split("\r\n") 
    train = [i[2:].strip(" ").split(" ") for i in train]

stemmer = PorterStemmer()
tfv = {}
tf = {}
train_docs = []
zero = [0]*13
for c,clist in enumerate(train):
    token_n = 0.0
    ctoken_n = 0.0
    for doc in clist:
        train_docs.append(int(doc))
        with open("IRTM/" + doc + ".txt") as f:
            f = f.read().lower()
            words = re.findall(r'[a-z]+', f)
	    words = [stemmer.stem(i) for i in words]
            ctoken_n += len(set(words))
            for word in words:
                token_n += 1.0
                try:
                    tf[word] += 1.0
                except KeyError:
                    tf[word] = 0.0
                tf[word] += 1.0
                try:
                    tfv[word][c] += 1.0
                except KeyError:
                    tfv[word] = [0.0]*13
                tfv[word][c] += 1.0
    zero[c] = log(1/(token_n+ctoken_n))
    tfv = {k: [(vv+1.0)/(token_n+ctoken_n) if ii == c else vv for ii, vv in enumerate(v)] for k, v in tfv.iteritems()}
features = [i[0] for i in sorted(tf.items(), key=operator.itemgetter(1))[-500:]]
print(train_docs)
with open('ans.csv', 'w') as f:
    f.write('id,Value\n')
    for i in range(1, 1096):
        if i not in train_docs:
            clf = np.array(pc)
            with open('IRTM/'+str(i)+'.txt') as ff:
                ff = ff.read().lower()
                words = re.findall(r'[a-z]+', ff)
                words = [stemmer.stem(ii) for ii in words]
                for word in words:
                    if word in features:
                        tmp = [log(j) if j != 0.0 else zero[ii] for ii,j in enumerate(tfv[word])]
                        clf += np.array(tmp) 
                m = np.argsort(clf)[12]
                f.write(str(i)+','+str(m+1)+'\n')
                
