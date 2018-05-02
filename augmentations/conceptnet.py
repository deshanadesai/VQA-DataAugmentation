import requests
import json
import csv
import nltk
import inflect
import argparse
import re
from tqdm import tqdm
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')


class conceptNet:

    def __init__(self):
        self.url = 'http://api.conceptnet.io/c/en/'
    
    def lookup(self,word):
        obj = requests.get(self.url+word).json()
        obj = obj['edges']
        return obj
        
    def find_relations(self,obj, targets):
        subgraph = []
        for i, node in enumerate(obj):
            if node["weight"] > 2.0:# and node["surfaceText"] != None:
                relation = node['rel']['label']
                if relation in targets:
                    subgraph.append(node)
        return subgraph
                
    def get_end_nodes(self, nodes, word = ''):
        ends = []
        pattern = '/c/en/(.*)'
        for n in nodes:
            #print(n['start']['@id'], re.match(pattern, n['start']['@id']))
            if (not bool(re.search(pattern, n['start']['@id']))) or (not bool(re.search(pattern, n['end']['@id']))):
                #print(n['start']['@id'])
                continue 
            if n['start']['label'] == word:
                ends.append((n['end']['label'],"start",n['weight']))
            if n['end']['label'] == word:
                ends.append((n['start']['label'],"end",n['weight']))
        #return list(set(ends))
        return ends
        
    def get_end_node(self, nodes, word = ''):
        pattern = r'/c/en/(.*)'
        
        if len(nodes)==0:
            return None
        for n in nodes:
            #print(n['start']['@id'], re.match(pattern, n['start']['@id']))
            if (not bool(re.search(pattern, n['start']['@id']))) or (not bool(re.search(pattern, n['end']['@id']))):
                #print(n['start']['@id'])
                continue            
            if n['start']['label'] == word:
                return [(n['end']['label'],"start",n['weight'])]
            if n['end']['label'] == word:
                return [(n['start']['label'],"end",n['weight'])]
            break

            
def pluralize(word, pos_tag_orig, target):
    target = target.replace("a ","")
    target = target.replace("the", "")
    
    if pos_tag_orig == 'NNS' and p.singular_noun(target):
        return p.plural(target)
    else:
        return target
    
def generate_ques(sentence, word, synonym):
    return sentence.replace(word, synonym)

def antonym(obj, word):
    nodes = c.find_relations(obj, ["Antonym"])
    return c.get_end_node(nodes, word)   

def is_a_word(obj, word):
    nodes = c.find_relations(obj, ["IsA"])
    return c.get_end_node(nodes, word) 

def has_a(obj,word):
    nodes = c.find_relations(obj, ["HasA"])
    return c.get_end_node(nodes, word)

def part_of(obj,word):
    nodes = c.find_relations(obj, ["PartOf"])
    return c.get_end_node(nodes, word) 

def made_of(obj,word):
    nodes = c.find_relations(obj, ["MadeOf"])
    return c.get_end_node(nodes, word)   

def used_for(obj,word):
    nodes = c.find_relations(obj, ["UsedFor"])
    return c.get_end_node(nodes, word)   

def has_property(obj,word):
    nodes = c.find_relations(obj, ["HasProperty"])
    return c.get_end_node(nodes, word)

def get_causes(obj,word):
    nodes = c.find_relations(obj, ["Causes"])
    return c.get_end_node(nodes, word)   

def get_synonyms(obj,word):
    nodes = c.find_relations(obj, ["Synonym"])
    return c.get_end_nodes(nodes, word)   

def get_entailments(obj,word):
    nodes = c.find_relations(obj, ["HasPrerequisite","HasSubEvent","HasFirstSubevent","Entails","HasLastSubevent","HasPrerequisite"])
    return c.get_end_nodes(nodes, word)

# Todo: check that token[0] does not appear twice in the sentence.
# Pluralize
# Remove duplicates
# Remove adverbs "an", "the" etc.
# Check if related to is an adjective
def substitutions(sentence):
    tagged = pos_tag(word_tokenize(sentence))
    questions = []
    answers = []
    scores = []
    qtype= []
    
    for token in tagged:
        if token[0] not in stop_words and token[0].isalpha():
            obj = c.lookup(token[0])
            
            synonyms = get_synonyms(obj,token[0])
            #hasproperty = has_property(obj,token[0])
            entails = get_entailments(obj,token[0])
            #antonyms = antonym(obj,token[0])
            #isa = is_a_word(obj,token[0])
            #hasa = has_a(obj,token[0])
            #partof = part_of(obj,token[0])
            madeof = made_of(obj,token[0])
            usedfor = used_for(obj,token[0])
            causes = get_causes(obj,token[0])
            
            
#             print(token[0])
#             print("----")
#             print("entails")
#             print(entails)
#             print("synonyms")
#             print(synonyms)
#             print(hasproperty, antonyms, isa, hasa, partof, madeof, usedfor, causes)
            
            if usedfor:
                for u_ in usedfor:
                    (u,direction,score) = u_
                    if not(str.lower(u) == str.lower(str(token[0]))) and u.isalpha():
                        
                        if direction=="start":
                            if token[1] == 'NNS':
                                questions.append("What are "+token[0]+" used to do?")
                                answers.append(u)
                                scores.append(score)
                                qtype.append("used_for")
                            else:
                                questions.append("What is "+token[0]+" used to do?")
                                answers.append(u)    
                                scores.append(score)
                                qtype.append("used_for")
                        #else:
                        #    questions.append("What is used for "+token[0]+"?")
                        #    answers.append(u)
                            
            if synonyms:
                for syn_ in synonyms:
                    (syn, tmp, score) = syn_
                    if not(str.lower(syn) == str.lower(str(token[0]))) and syn.isalpha():
                        syn = pluralize(token[0], token[1], syn)
                        questions.append(generate_ques(sentence, token[0], syn))
                        answers.append("None")
                        scores.append(score)
                        qtype.append("synonyms")
            
#             if hasproperty:
#                 for hap_ in hasproperty:
#                     (hap, direction) = hap_
#                     if not(str.lower(hap) == str.lower(str(token[0]))) and hap!="stupid" and hap.isalpha():
#                         questions.append(generate_ques(sentence, token[0], (hap+" "+token[0])))
#                         answers.append("None")                     
                        
            if entails:
                for ent_ in entails:
                    (ent, direction, score) = ent_
                    ent = pluralize(token[0], token[1], ent)
                    if not(str.lower(ent) == str.lower(str(token[0]))) and ent.isalpha():
                        if direction=="start":
                            questions.append(generate_ques(sentence, token[0], ent))
                            answers.append("None")
                            scores.append(score)
                            qtype.append("entails")
                        
#             if isa and token[1].startswith('N'):
#                 for is_a_ in isa:
#                     (is_a, direction) = is_a_
#                     if not(str.lower(is_a) == str.lower(str(token[0]))) and is_a.isalpha():
#                         is_a = pluralize(token[0], token[1], is_a)
#                         questions.append(generate_ques(sentence, token[0], is_a))
#                         answers.append("None")
                        
            if causes:
                for c_ in causes:
                    (caus, direction, score) = c_
                    caus = pluralize(token[0], token[1], caus)
                    if not(str.lower(caus) == str.lower(str(token[0]))) and caus.isalpha():
                        if direction=="start":
                            questions.append("What is "+token[0]+" likely to cause ?")
                            answers.append(caus)
                            scores.append(score)
                            qtype.append("causes")
#                         else:
#                             questions.append("What is likely to cause a "+token[0])
#                             answers.append(caus)
                        

            if madeof:
                for m_ in madeof:
                    (m, direction, score) = m_
                    m = pluralize(token[0], token[1], m)
                    if not(str.lower(m) == str.lower(str(token[0]))) and m.isalpha():
                        if direction=="start":
                            questions.append("What is "+token[0]+" likely to be made of?")
                            answers.append(m)
                            scores.append(score)
                            qtype.append("made_of")
                        
    if questions!=[]:
        for i in range(len(questions)):
            writer.writerow([sentence,questions[i],answers[i], scores[i], qtype[i]])
#     print(sentence)
#     print(questions)
#     print(answers)

if __name__ == '__main__':
    data = json.load(open('/home/deshana/Code/data/mscoco/v2_OpenEnded_mscoco_train2014_questions.json'))
    stop_words = set(stopwords.words('english'))


#     parser = argparse.ArgumentParser()
#     parser.add_argument("index", type=int, help="path to annotations file")
    
#     args = parser.parse_args()
#     row = data['questions'][args.index]

    
    
    c = conceptNet()
    p = inflect.engine()    
#     substitutions(row['question'])
    

    #f = open("conceptnet.csv","r")
    #reader = csv.reader(f)
    #qns = []
    #for row in reader:
    #    qns.append(row[0])
    #f.close()

    f = open("conceptnet.csv","a")
    writer = csv.writer(f)
    for row in tqdm(data['questions']):
        if row in qns: continue
        try:
            substitutions(row['question'])
        except:
            print("Encountered an error. Skipping Question")
    f.close()
