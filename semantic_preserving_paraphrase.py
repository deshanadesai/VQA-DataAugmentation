from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import spacy
import json
from pprint import pprint
nlp = spacy.load('en')
import textacy
import csv

def get_VP(sentence):
    pattern = r'<VERB>?<ADV>*<VERB>+'
    doc = textacy.Doc(sentence, lang='en')
    lists = textacy.extract.pos_regex_matches(doc, pattern)
    return lists

def get_triplet(sentence):
    doc = nlp(sentence)
    triplet = []
    for chunk in doc.noun_chunks:
        #print(chunk.text, chunk.root.text, chunk.root.dep_)
              #chunk.root.head.text)
        triplet.append({chunk.root.dep_: chunk.root.text})
    #print(triplet)
    return triplet

data = json.load(open('/Users/deshanadesai/Code/COCO/v2_OpenEnded_mscoco_train2014_questions.json'))

f = open('store_triplets.csv','w')
writer = csv.writer(f)

for question in data['questions'][:1000]:
    sentence = question['question']
    triplet = get_triplet(sentence)
    VPs = get_VP(sentence)

    if (triplet is not None) and (VPs is not None):
        verbs = []
        for vp in VPs:
            verbs.append(vp.text)
        triplet.append({'verb phrase':verbs})
        tagged = pos_tag(word_tokenize(sentence))
        verbs = []
        for token in tagged:
            if token[1].startswith('V'):
                verbs.append(token[0])
        triplet.append(verbs)
        writer.writerow([sentence, triplet])

f.close()
    
    