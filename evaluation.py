import os
import json
import tablib
import collections
from context_classifier import context_classifier as cc
from nltk import stem
from nltk.corpus import stopwords
from tqdm import tqdm
from datetime import date
from modified_methods import m_mrakun
from modified_methods.m_yake import m_yake

CS_DICT = "data/dictionaries/CSO.3.3.txt"
ECON_DICT = "data/dictionaries/stw.txt"
MED_DICT = "data/dictionaries/mesh2021.txt"
AGRO_DICT = "data/dictionaries/agrovoc_2021-07-02_core.txt"
WIKI_ENTRIES = "data/dictionaries/enwiki-latest-all-titles-in-ns0-cleaned.txt"

class Evaluator():
    def __init__(self, dictPath=None, wiki=False, pos=False):
        self.goldKPs = []
        self.inputs = []
        self.extractedKPs = []
        self.dict = set()
        self.dicts = {}
        self.wikis = set()
        self.pos = pos
        self.wiki = wiki
        self.dictPath = dictPath

        self.microRecall = 0.0
        self.microPrecision = 0.0
        self.microRPrecision = 0.0
        self.microF1 = 0.0
        self.macroRecall = 0.0
        self.macroPrecision = 0.0
        self.macroRPrecision = 0.0
        self.macroF1 = 0.0

        if dictPath and dictPath != "auto":
            with open(dictPath, "r", encoding="utf8") as f:
                for l in f.read().splitlines():
                    self.dict.add(l.lower())
        elif dictPath == "auto":
            with open(CS_DICT, "r", encoding="utf8") as f:
                dict_cs = set()
                for l in f.read().splitlines():
                    dict_cs.add(l.lower())
                self.dicts["cs"] = dict_cs
            with open(ECON_DICT, "r", encoding="utf8") as f:
                dict_econ = set()
                for l in f.read().splitlines():
                    dict_econ.add(l.lower())
                self.dicts["fin"] = dict_econ
            with open(MED_DICT, "r", encoding="utf8") as f:
                dict_med = set()
                for l in f.read().splitlines():
                    dict_med.add(l.lower())
                self.dicts["bio"] = dict_med

        if wiki:
            with open(WIKI_ENTRIES, "r", encoding="utf8") as f:
                for l in f.read().splitlines():
                    self.wikis.add(l.lower())

        print("Evaluator initialised...")

    def __loadDataset__(self, dataset):
        print("Dataset loading started...")
        self.dataset = dataset
        self.inputs = []
        self.goldKPs = []

        if dataset in ["kpcrowd", "citeulike180", "fao30", "fao780", "inspec", "kdd", "krapivin2009",
                       "nguyen2007", "pubmed", "schutz2008", "semeval2010", "semeval2017", "theses100",
                       "wiki20", "www", "inspec-s", "kdd-s", "www-s", "kptimes-s"]:
            path = 'data/datasets/' + dataset + '/'
            goldKPs = [f[:-4] for f in os.listdir(path + 'keys/')]
            inputs = [f[:-4] for f in os.listdir(path + 'docsutf8/')]

            for i in range(len(inputs)):
                with open(path + 'docsutf8/' + inputs[i] + '.txt', 'r', encoding='utf-8') as f:
                    self.inputs.append(f.read())

            for i in range(len(goldKPs)):
                keywords = []
                with open(path + 'keys/' + goldKPs[i] + '.key', 'r', encoding='utf-8') as f:
                    for line in f:
                        keywords.append(line.strip('\n'))
                keywords = [k.lower() for k in keywords]
                self.goldKPs.append(keywords)

        elif dataset == "kptimes":
            with open('data/datasets/KPTimes/KPTimes.test.jsonl', encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
                for d in data:
                    self.inputs.append(d["abstract"])
                    self.goldKPs.append([kw.lower() for kw in d["keyword"].split(';')])

        elif dataset == "kptimes-econ":
            with open('data/datasets/KPTimes/KPTimes_economics.jsonl', encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
                for d in data:
                    self.inputs.append(d["abstract"])
                    self.goldKPs.append([kw.lower() for kw in d["keyword"].split(';')])

        elif dataset == "duc":
            path = 'data/datasets/duc-2001-pre-master/'
            inputs = [f for f in os.listdir(path + 'src/txt/')]

            with open(path + 'references/test.reader.json') as f:
                data = json.loads(f.read())
                for doc, kps in data.items():
                    self.goldKPs.append([kw[0].lower() for kw in kps])

            for i in range(len(inputs)):
                with open(path + 'src/txt/' + inputs[i], 'r', encoding='utf-8') as f:
                    self.inputs.append(f.read())

        print("Dataset " + dataset + " loaded...")

    def __extractKeyphrases__(self, method, ngram, stoplist, threshold, number):
        print("Keyphrase extraction with " + method + " started...")
        self.extractedKPs = []

        for i in tqdm(range(len(self.inputs))):
            # Choose thesaurus...
            if not self.dictPath:
                dictionary = None
            elif self.dictPath == "auto":
                context = cc.predict_tags(self.inputs[i])
                if 'cs' in context:
                    dictionary = self.dicts['cs']
                elif 'fin' in context:
                    dictionary = self.dicts['fin']
                elif 'bio' in context:
                    dictionary = self.dicts['bio']
                else:
                    dictionary = None
            else:
                dictionary = self.dict

            # Load wiki titles...
            if self.wiki:
                titles = self.wikis
            else:
                titles = None    
                
            if method == "yake":
                extractor = m_yake.KeywordExtractor(lan='en', n=ngram, dedupLim=threshold,
                                                    windowsSize=1, top=number, stopwords=stoplist,
                                                    thes=dictionary, wiki=titles, pos=self.pos)
                output = extractor.extract_keywords(self.inputs[i])                                                        
            elif method == "rakun":
                hyperparameters = {"distance_threshold":2,
                                   "distance_method": "editdistance",
                                   "num_keywords" : number,
                                   "pair_diff_length":2,
                                   "stopwords" : stoplist,
                                   "bigram_count_threshold":2,
                                   "num_tokens":list(range(1,ngram+1)),
		                           "max_similar" : 3, ## n most similar can show up n times
		                           "max_occurrence" : 3} ## maximum frequency overall
                extractor = m_mrakun.RakunDetector(hyperparameters, thes=dictionary, wiki=titles, pos=self.pos)
                output = extractor.find_keywords(self.inputs[i], input_type = "text")  

            self.extractedKPs.append(list(list(dict(output).keys())))      

            with open("./results/experiments/logs/{}_{}.log".format(method, date.today()), "a", encoding='utf-8') as f:
                f.write("Extracted keyphrases: " + str(self.extractedKPs[-1]) + "\n")
                f.write("Gold standard keyphrases: " + str(self.goldKPs[i]) + "\n")

    def __calculateScore__(self, average, rprecision=False, k=10, matching="exact"):
        stemmer = stem.PorterStemmer()
        cumTruePositives = 0
        cumGolds = 0
        cumExts = 0
        precisions = []
        recalls = []
        self.nonexacts = [[]] * len(self.extractedKPs)

        for i in range(len(self.extractedKPs)):
            stems_gold = []
            stems_ext = []

            if rprecision:
                n = len(self.goldKPs[i])
            else:
                n = k

            self.extractedKPs[i] = self.extractedKPs[i][:n]

            for kw in self.goldKPs[i]:
                kw = " ".join([stemmer.stem(word) for word in kw.split()])
                kw = kw.lower()
                stems_gold.append(kw)

            for kw in self.extractedKPs[i]:
                kw = " ".join([stemmer.stem(word) for word in kw.split()])  
                kw = kw.lower() 
                stems_ext.append(kw) 

            if matching == "exact":
                truePositives = 0

            for s in stems_ext:
                if s in stems_gold:
                    truePositives += 1

            if average == "micro":
                cumTruePositives += truePositives
                cumGolds += len(self.goldKPs[i])
                cumExts += n
            if average == "macro":
                try:
                    recalls.append(truePositives / float(len(stems_gold)))
                except:
                    recalls.append(0)
                try:
                    precisions.append(truePositives / float(len(stems_ext)))
                except:
                    precisions.append(0)

        if average == "micro":
            try:
                recall = cumTruePositives / float(cumGolds)
            except:
                recall = 0
            try:
                precision = cumTruePositives / float(cumExts)
            except:
                precision = 0
        if average == "macro":
            try:
                recall = sum(recalls) / float(len(recalls))
            except:
                recall = 0
            try:
                precision = sum(precisions) / float(len(precisions))
            except:
                precision = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / float(precision + recall)
        else:
            f1 = 0

        if rprecision:
            return precision
        else:
            return precision, recall, f1

    def __computeScores__(self, k=10, matching="exact"):
        print("Score calculation started...")
        self.microPrecision, self.microRecall, self.microF1 = self.__calculateScore__("micro", False, k, matching)
        self.macroPrecision, self.macroRecall, self.macroF1 = self.__calculateScore__("macro", False, k, matching)
        self.macroRPrecision = self.__calculateScore__("macro", True, k, matching)
        self.microRPrecision = self.__calculateScore__("micro", True, k, matching)
        print("Scores calculated...")

    def __saveNonExactMatchings__(self, dataset):
        data = tablib.Dataset(headers=["Dataset", "Input", "Extracted", "Gold"])
        for i in self.nonexacts:
            if i:
                for k in i:
                    data.append((dataset, self.inputs[self.nonexacts.index(i)], k[0], k[1]))

        with open('results/nonexact-matchings/' + dataset + '_nonexact-matchings.xlsx', 'wb') as f:
            f.write(data.export('xlsx'))

    def evaluate(self, dataset="kpcrowd", method="yake", matching="exact", k=10, n=3, ngram_analysis=False):
        self.__loadDataset__(dataset)
        self.__extractKeyphrases__(method, n, stopwords.words('english'), threshold=0.9, number=10)
        self.__computeScores__(k, matching)
        if matching != "exact":
            self.__saveNonExactMatchings__(dataset)
        if ngram_analysis:
            self.ngramAnalysis(dataset, method, k)

    def ngramAnalysis(self, dataset, method=None, k=10):
        ngram = {}
        if method:
            for exts in self.extractedKPs:
                counts = [len(str.split(i)) for i in exts[:k]]
                for c in counts:
                    if c > 4:
                        c = 4
                    if not c in ngram:
                        ngram[c] = 1
                    else:
                        ngram[c] += 1
        else:
            for goldens in self.goldKPs:
                counts = [len(str.split(i)) for i in goldens]
                for c in counts:
                    if c > 4:
                        c = 4
                    if not c in ngram:
                        ngram[c] = 1
                    else:
                        ngram[c] += 1
        total = sum(ngram.values())
        for key in ngram:
            ngram[key] = round(ngram[key] / total, 2)

        self.ngrams = collections.OrderedDict(sorted(ngram.items()))
        with open("./results/experiments/logs/{}_{}_ngram.json".format(method, date.today()), "a", encoding='utf-8') as f:
            json.dump(self.ngrams, f)