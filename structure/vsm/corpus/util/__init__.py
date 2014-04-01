import re

import numpy as np

import nltk

import corpusbuilders
from corpusbuilders import *

__all__ = ['strip_punc', 'rem_num', 'rehyph', 'add_metadata',
         'apply_stoplist', 'filter_by_suffix', 'word_tokenize',
         'sentence_tokenize', 'paragraph_tokenize']
__all__ += corpusbuilders.__all__



def strip_punc(tsent):
    """
    """
    p1 = re.compile(r'^(\W*)')
    p2 = re.compile(r'(\W*)$')

    out = []
    for word in tsent:
        w = re.sub(p2, '', re.sub(p1, '', word))
        if w:
            out.append(w)

    return out


def rem_num(tsent):
    """
    """
    p = re.compile(r'(^\D+$)|(^\D*[1-2]\d\D*$|^\D*\d\D*$)')

    return [word for word in tsent if re.search(p, word)]


def rehyph(sent):
    """
    """
    return re.sub(r'(?P<x1>.)--(?P<x2>.)', '\g<x1> - \g<x2>', sent)


def add_metadata(corpus, ctx_type, new_field, metadata):
    """
    """
    from vsm import arr_add_field
    i = corpus.context_types.index(ctx_type)
    md = corpus.context_data[i]
    corpus.context_data[i] = arr_add_field(md, new_field, metadata)

    return corpus


def apply_stoplist(corp, nltk_stop=True, add_stop=None, freq=0):
    """
    """
    stoplist = set()
    if nltk_stop:
        for w in nltk.corpus.stopwords.words('english'):
            stoplist.add(w)
    if add_stop:
        for w in add_stop:
            stoplist.add(w)

    return corp.apply_stoplist(stoplist=stoplist, freq=freq)


def filter_by_suffix(l, ignore):
    """
    """
    return [e for e in l if not sum([e.endswith(s) for s in ignore])]


def word_tokenize(text):
    """Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    lower-case words in this text with numbers and punctuation, except
    for hyphens, removed.

    The core work is done by NLTK's Treebank Word Tokenizer.
    """

    text = rehyph(text)
    text = nltk.TreebankWordTokenizer().tokenize(text)

    tokens = [word.lower() for word in text]
    tokens = strip_punc(tokens)
    tokens = rem_num(tokens)
    
    return tokens


def sentence_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    sentences in this text.

    This is a wrapper for NLTK's pre-trained Punkt Tokenizer.
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return tokenizer.tokenize(text)


def paragraph_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    paragraphs in this text. It's expected that the text marks
    paragraphs with two consecutive line breaks.
    """

    par_break = re.compile(r'[\r\n]{2,}')
    
    return par_break.split(text)


