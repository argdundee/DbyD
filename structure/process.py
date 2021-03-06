#process.py textfile.txt stc.pickle endc.pickle corpus.npz model.npz
import nltk
import pickle
import string
import math
import textwrap
import sys, os

from nltk.corpus import stopwords

from vsm.model.ldagibbs import LDAGibbs as LDA
from vsm.viewer.ldagibbsviewer import LDAGibbsViewer as LDAViewer
from vsm.corpus import Corpus

def features(wordarray):
    after = wordarray + ['EOF']
    before = ['BOF'] + wordarray
    pos = nltk.pos_tag(wordarray)
    ft = zip(wordarray,pos,before[:-1],after[1:])
    return [{'word':wrd[0], 'before':wrd[2], 'after':wrd[3], 'pos':wrd[1][1]} for wrd in ft]

def vdifference(v1,v2):
    u = zip(v1,v2)
    d = [i[0][1]-i[1][1] for i in u]
    ss = sum(x*x for x in d)
    return math.sqrt(ss)

#Open file and split into words
exepath = os.path.dirname(os.path.abspath(sys.argv[0])) + '/'
print "Opening file"
file = open(sys.argv[1])
raw = file.read()
rawwords = nltk.word_tokenize(raw)

#Compute features for each word
print "Computing features for each word"
fts = features(rawwords)

#Classify starts and ends
f = open(exepath+sys.argv[2])
stclassifier = pickle.load(f)
f.close()

f = open(exepath+sys.argv[3])
endclassifier = pickle.load(f)
f.close()

print "Classifying words"
sttest = [(f['word'], stclassifier.classify(f)) for f in fts]
endtest = [(f['word'], endclassifier.classify(f)) for f in fts]

#Extract props from text
print "Identifying propositions"
inprop = False
props = []
prop = []
for i in range(0, len(sttest)):
    if sttest[i][1] == 'start':
        prop = []
        inprop = True
    if endtest[i][1] == 'end':
        if prop != []:
            prop.append(sttest[i][0])
            props.append(prop)
            prop = []
        inprop = False
    if inprop:
        prop.append(sttest[i][0])

#Get topics for each prop
c = Corpus.load(exepath+sys.argv[4])
m = LDA.load(exepath+sys.argv[5])

v = LDAViewer(c,m)
stopwords = stopwords.words('english')

allowed_chars=string.ascii_letters
trans_table = string.maketrans('','')

print "Applying topic model"
#Remove props with only words in stoplist
vsmprops = []
np = []
for p in props:
    np = [w.lower() for w in p if w.lower() not in stopwords and not w.translate(trans_table,allowed_chars)]
    if len(np) > 0:
        vsmprops.append(p)

props = vsmprops

proptopics = [sorted(v.sim_word_top([w.lower() for w in p if w.lower() not in stopwords and not w.translate(trans_table,allowed_chars)], show_topics=False), key=lambda t: t[0]) for p in props]

edges = []
for i in range(1, len(proptopics)):
    if vdifference(proptopics[i], proptopics[i-1]) < 0.5:
        edges.append((i, i-1))
    else:
        mindistance = 1
        for j in range(0, i-1):
            d = vdifference(proptopics[i], proptopics[j])
            if d < mindistance:
                mindistance = d
                link = j
        if mindistance < 0.5:
            edges.append((i, j))

#Generate DOT
print "Generating diagram"
dot = 'graph pgraph {'

for i in range (0,len(props)):
    dot = dot + str(i+1) + ' [label="' + '\\n'.join(textwrap.wrap(' '.join(props[i]), 40)) + '", shape="box", style=filled, fontname="FreeSans", color="#6666cc", fillcolor="#ebf3ff"];'

for edge in edges:
    dot = dot + str(edge[0]+1) + ' -- ' + str(edge[1]+1) + ';'

dot = dot + '}' 

f = open(sys.argv[6], 'w')
f.write(dot)
f.close()
