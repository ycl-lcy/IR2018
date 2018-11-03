import re
import numpy as np
from stemming.porter2 import stem
from stop_words import get_stop_words
from math import log10


N = 1095
dic = {}
tfs = [{}]
docs = [[[],[]]]
words = []
for i in range(1, N+1):
    with open('IRTM/'+ str(i) +'.txt', 'r') as input_file:
        input = input_file.read()
        input = input.lower()
        input = re.findall(r'[a-z]+', input)

    stop_words = get_stop_words('english')

    tf = {}
    words_tmp = []
    for word in input:
        word = stem(word)
        if word not in stop_words and len(word) > 1:
            words.append(word)
            words_tmp.append(word)
            if word in tf:
                tf[word][1] += 1
            else:
                tf[word] = [0, 1]
    
    tfs.append(tf)

    for word in set(words_tmp):
        if word in dic:
            dic[word][1] += 1
        else:
            dic[word] = [0, 1]

words = set(words)
df = sorted(dic.items())

dic = {}
for i,e in enumerate(df):
    dic[e[0]] = [i+1, e[1][1]]

with open('dictionary.txt', 'w') as f:
    f.write("t_index\tterm\tdf\n")
    for i in df:
        f.write(str(dic[i[0]][0])+"\t"+i[0]+"\t"+str(i[1][1])+"\n")

for tf in tfs:
    for word, i in tf.items():
        tf[word][0] = dic[word][0]

for i in range(1, N+1):
    tfs[i] = sorted(tfs[i].items())

# for ii, tf in enumerate(tfs):
    # doc = [[],[]]
    # length = 0
    # for i in tf:
        # doc[0].append(i[1][0])
        # doc[1].append(i[1][1]*log10(N/dic[i[0]][1]))
        # length += (i[1][1]*log10(N/dic[i[0]][1]))**2
    # print("jizz")
    # print(doc)
    # length = length**(1/2)
    # #tmp = np.array(doc[1])/length
    # docs.append(doc)

# for i in range(1, N+1):
    # with open('vector/'+str(i), 'w') as f:
        # f.write("t_index\ttf-idf\n")
        # for j in range(docs[i][0]):
            # f.write(str(docs[i][0][j])+"\t"+str(docs[i][1][j])+"\n")
for i in range(1, N+1):
    tf = tfs[i]
    doc_index = []
    doc_tfidf = []
    length = 0
    for j in tf:
        doc_index.append(dic[j[0]][0])
        doc_tfidf.append(j[1][1]*log10(N/dic[j[0]][1]))
        length += (j[1][1]*log10(N/dic[j[0]][1]))**2
    length = length**(1/2)
    doc_tfidf = np.array(doc_tfidf)
    doc_tfidf = doc_tfidf/length
    doc_tfidf = doc_tfidf.tolist()
    doc = list(zip(doc_index, doc_tfidf))
    docs.append(doc)
    with open(str(i)+".txt", 'w') as f:
        f.write("t_index\ttf-idf\n")
        for j in doc:
            f.write(str(j[0])+"\t"+str(j[1])+"\n")

#docs = [sorted(i, key=lambda x: x[0]) for i in docs]


# for i in range(1, N+1):
    # with open('vector/'+str(i), 'w') as f:
        # f.write("t_index\ttf-idf\n")

def consine(Doc_x, Doc_y):
    cos = 0
    x = docs[Doc_x]
    y = docs[Doc_y]
    for p in x:
        for q in y:
            if p[0] == q[0]:
                cos += float(p[1])*float(q[1])

    #cos = cos/length_x/length_y
    return cos


print(consine(1,2))
