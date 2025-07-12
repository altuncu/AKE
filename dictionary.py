import rdflib
import jsonlines
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def convert_to_txt():
    agrovoc = open("data/dictionaries/agrovoc_2021-07-02_core.txt","w", encoding="utf-8")
    mesh = open("data/dictionaries/mesh2021.txt", "w", encoding="utf-8")
    stw = open("data/dictionaries/stw.txt", "w", encoding="utf-8")
    cso = open("data/dictionaries/CSO.3.3.txt", "w", encoding="utf-8")

    g = rdflib.Graph()
    g.parse("data/dictionaries/agrovoc_2021-07-02_core.nt")
    for a, b, c in g:
        if b.endswith("literalForm"):
            if c.language == "en":
                agrovoc.write(c.value + "\n")
    agrovoc.close()

    g.parse("data/dictionaries/mesh2021.nt")
    for a, b, c in g:
        if b.endswith("rdf-schema#label"):
            if c.language == "en":
                mesh.write(c.value + "\n")
    mesh.close()

    g.parse("data/dictionaries/stw.nt")
    for a, b, c in g:
        if b.endswith("prefLabel"):
            if c.language == "en":
                stw.write(c.value + "\n")
    stw.close()

    g.parse("data/dictionaries/CSO.3.3.nt")
    for a, b, c in g:
        if b.endswith("rdf-schema#label"):
            if c.language == "en":
                cso.write(c.value + "\n")
    cso.close()


def extract_category(category):
    data = []
    with jsonlines.open('data/datasets/KPTimes/KPTimes.test.jsonl') as f:
        data += [line for line in f if category in line["keyword"].split(";") or category in line["categories"]]
    with jsonlines.open('data/datasets/KPTimes/KPTimes.valid.jsonl') as f:
        data += [line for line in f if category in line["keyword"].split(";") or category in line["categories"]]
    with jsonlines.open('data/datasets/KPTimes/KPTimes.train.jsonl') as f:
        data += [line for line in f if category in line["keyword"].split(";") or category in line["categories"]]
    with jsonlines.open('data/datasets/KPTimes/KPTimes_economics.jsonl', "w") as o:
        o.write_all(data)
    print(str(len(data)) + " items extracted from the category '" + category + "'...")

def clean_wiki_data(path):
    with open(path, "r", encoding="utf8") as inFile, \
    open(path + "-cleaned.txt", 'w', encoding="utf8") as outFile:
        for l in inFile.read().splitlines():
            outFile.write(re.sub(r"\([^()]*\)", "", lemmatizer.lemmatize(l.replace("_", " ")).lower()) + "\n")
