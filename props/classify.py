import nltk
import pickle
import string
import sys, os

from nltk.corpus import stopwords

def features(wordarray):
    after = wordarray + ['EOF']
    before = ['BOF'] + wordarray
    pos = nltk.pos_tag(wordarray)
    ft = zip(wordarray,pos,before[:-1],after[1:])
    return [{'word':wrd[0], 'before':wrd[2], 'after':wrd[3], 'pos':wrd[1][1]} for wrd in ft]

#Open file and split into words
exepath = os.path.dirname(os.path.abspath(sys.argv[0])) + '/'
print "Opening file"
file = open(sys.argv[1])
raw = file.read()
rawwords = nltk.word_tokenize(raw)

#Compute features for each word
print "Computing features for each word"
fts = features(rawwords)

clroot = sys.argv[2]
#Classify starts and ends
f = open(exepath+clroot+'-st.pickle')
stclassifier = pickle.load(f)
f.close()

f = open(exepath+clroot+'-end.pickle')
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

dot = ''
for i in range (0,len(props)):
    dot = dot + str(i+1) + ' [label="' + ' '.join(props[i]) + '"];' + "\n"

print dot
