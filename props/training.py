import os,sys
import nltk
import pickle

def ftstart(word):
    if "PROPENDW" in word:
        return (word[:-8], '')
    elif "PROPSTARTW" in word:
        return (word[10:], 'start')
    else:
        return (word, '')

def ftend(word):
    if "PROPENDW" in word:
        return (word[:-8], 'end')
    elif "PROPSTARTW" in word:
        return (word[10:], '')
    else:
        return (word, '')


def features(wordarray):
    after = wordarray + ['EOF']
    before = ['BOF'] + wordarray
    pos = nltk.pos_tag(wordarray)
    ft = zip(wordarray,pos,before[:-1],after[1:])
    return [{'word':wrd[0], 'before':wrd[2], 'after':wrd[3], 'pos':wrd[1][1]} for wrd in ft]

print "Reading file"
file = open(sys.argv[1])
raw = file.read()
rawwords = nltk.word_tokenize(raw)
slblwords = [ftstart(w) for w in rawwords]
elblwords = [ftend(w) for w in rawwords]

print "Computing features for each word"
fts = features([l[0] for l in slblwords])

sttrain = [(t[1], t[0][1]) for t in zip(slblwords,fts)]
endtrain = [(t[1], t[0][1]) for t in zip(elblwords,fts)]

print "Training start classifier"
stclassifier = nltk.NaiveBayesClassifier.train(sttrain)
sttest = [(f['word'], stclassifier.classify(f)) for f in fts]

print "Training end classifier"
endclassifier = nltk.NaiveBayesClassifier.train(endtrain)
endtest = [(f['word'], endclassifier.classify(f)) for f in fts]

outroot = sys.argv[2]
f = open(outroot+'-st.pickle', 'wb')
pickle.dump(stclassifier, f)
f.close()
f = open(outroot+'-end.pickle', 'wb')
pickle.dump(endclassifier, f)
f.close()

