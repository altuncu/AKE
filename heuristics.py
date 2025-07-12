import csv
from langdetect import detect
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions
from collections import Counter
import nltk

def extract_noun_phrases(text):
        grammar = r"""
            NBAR:
                {<NN.*|JJ|VBG|VBN|CD>*<NN.*|VBG>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR|JJ>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        noun_phrases = set()
        sentences_str = [ [w for w in split_contractions(web_tokenizer(s)) if not (w.startswith("'") and len(w) > 1) and len(w) > 0] for s in list(split_multi(text)) if len(s.strip()) > 0]
        sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
        chunker = nltk.RegexpParser(grammar)
        toks = nltk.regexp_tokenize(text, sentence_re)
        postoks = nltk.tag.pos_tag(toks)
        tree = chunker.parse(postoks)
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            noun_phrases.add(" ".join([w[0] for w in subtree.leaves()]).lower())
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NBAR'):
            string = " ".join([w[0] for w in subtree.leaves()])
            noun_phrases.add(string.split(" ")[-1].lower())
            while len(string.split(" ")) > 1:
                string = string.split(" ", 1)[1]
                noun_phrases.add(string.lower())
        return noun_phrases

def second(kw):
    return kw[1]

def apply_heuristics(input, keywords, thes, wiki, pos, lan):
    if lan == "auto":
        language = detect(input)
    else:
        language = lan    
    
    if pos:
        noun_phrases = extract_noun_phrases(input)
        for kw in keywords:
            if kw[0].lower() not in noun_phrases:
                keywords.remove(kw)
    if thes:   
        for kw in keywords:
            if kw[0].lower() in thes:
                kw[1] = kw[1] / 2
                keywords.sort(key=second, reverse=False)
    if wiki:
        for kw in keywords:
            if kw[0].lower() in wiki:
                kw[1] = kw[1] / 2
                keywords.sort(key=second, reverse=False)

    return keywords    


def pos_distribution(dataset, inputs, goldens):
    distribution = Counter()
    patterns = []
    for i in range(len(inputs)):
        sentences_str = [ [w for w in split_contractions(web_tokenizer(s)) if not (w.startswith("'") and len(w) > 1) and len(w) > 0] for s in list(split_multi(inputs[i])) if len(s.strip()) > 0]
        sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
        toks = nltk.regexp_tokenize(inputs[i], sentence_re)
        postoks = dict(nltk.tag.pos_tag(toks))
        for g in goldens[i]:
            pattern = ""
            for w in g.split():
                try:
                    pattern = pattern + postoks[w] + " " 
                except:
                    continue    
            patterns.append(pattern)
    distribution += Counter(patterns)
    total = sum(distribution.values()) - distribution[""]
    
    with open('results/goldens-analysis/pos-distribution-'+ dataset + '.csv','w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pattern", "Count"])
        for key, value in dict(distribution.most_common()).items():
            if key != "":
                writer.writerow([key, value/float(total)])

