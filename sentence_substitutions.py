from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
import json
from pprint import pprint
import argparse

def get_synsets(word, pos_tag):
    synsets = wordnet.synsets(word, pos = pos_tag)
    return synsets
    
def get_synonyms(word, pos_tag):
    synsets = get_synsets(word, pos_tag)[0]
    return synsets.lemma_names()

def get_entailments(word, pos_tag):
    synsets = get_synsets(word, pos_tag)
    if synsets:
        return synsets[0].entailments()
    else:
        return []

def get_definition(syn):
    return syn.definition()

def get_hypernyms(word, pos_tag):
    synsets = get_synsets(word, pos_tag)[0]
    return synsets.hypernyms()

def get_similar(syn):
    return syn.similar_tos()
    
def generate_ques(sentence, word, synonym):
    return sentence.replace(word, synonym)
    
def synonym_substitution(sentence):
    tagged = pos_tag(word_tokenize(sentence))
    questions = []
    for token in tagged:
        if token[1].startswith('N'):
            syns = get_synonyms(token[0], wordnet.NOUN)
            for synonyms in syns:
                if not(str.lower(str(synonyms)) == str.lower(str(token[0]))):
                    questions.append(generate_ques(sentence, token[0], synonyms))
    #print(sentence)
    print(questions)

def hypernym_substitution(sentence):
    tagged = pos_tag(word_tokenize(sentence))
    questions = []
    for token in tagged:
        if token[1].startswith('N'):
            syns = get_hypernyms(token[0], wordnet.NOUN)
            for hyper in syns:
                nms = hyper.lemma_names()
                for hypernyms in nms:
                    if not(str(hypernyms) == str(token[0])):
                        questions.append(generate_ques(sentence, token[0], hypernyms))
    #print(sentence)
    print(questions)    

def entailment_substitution(sentence):
    tagged = pos_tag(word_tokenize(sentence))
    questions = []
    for token in tagged:
        if token[1].startswith('V'):
            syns = get_entailments(token[0], wordnet.VERB)
            for entail in syns:
                nms = entail.lemma_names()
                for entailments in nms:
                    if not(str(entailments) == str(token[0])):
                        questions.append(generate_ques(sentence, token[0], entailments))
    #print(sentence)
    print(questions)  
    
def substitutions(sentence):
    print(sentence)
    print("substitutions:")
    entailment_substitution(sentence)
    hypernym_substitution(sentence)
    synonym_substitution(sentence)
    
    
    
if __name__ == '__main__':
    data = json.load(open('/Users/deshanadesai/Code/COCO/v2_OpenEnded_mscoco_train2014_questions.json'))


    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help="path to annotations file")
    args = parser.parse_args()
    row = data['questions'][args.index]
    sentence = row['question']
    substitutions(sentence)

'''
tagged = pos_tag(word_tokenize(sentence))
print(tagged)
for token in tagged:
    if token[1].startswith('N'):
        synsets = wordnet.synsets(token[0], pos=wordnet.NOUN)
        print("**")
        print(token[0])
        print(synsets)
        print("Name: ",synsets[0].name())
        print("Def: ",synsets[0].definition())
        print("EG. : ",synsets[0].examples())
        print("Similar: ",synsets[0].similar_tos())
        print("Hypernym: ",synsets[0].hypernyms())
        print("Hyponym: ",synsets[0].hyponyms())
        print("Synonyms: ",synsets[0].lemma_names())
        print("Entailments: ", synsets[0].entailments())
'''
