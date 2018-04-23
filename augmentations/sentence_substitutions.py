from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
import json
from pprint import pprint
import argparse
from nltk.corpus import stopwords
import gensim.downloader as api
import operator



def get_synsets(word, pos_tag):
    synsets = wordnet.synsets(word, pos = pos_tag)
    #synsets = word_sense_profile(word, sentence, pos_tag)
    return synsets

def get_distance(word_profiles, profiles):
    sim = 0
    counter = 0
    for w in word_profiles:
        for p in profiles:
            try:
                
                sim += model.similarity(w,p)
                counter += 1
            except:
                pass
    return sim/counter

def word_sense_profile(word, sentence, pos):
    synsets = get_synsets(word, pos)[:2]
    #f = lambda target: ' '.join(target.split('_'))
    
    # question filtered by stop words and tokenized
    tokens = pos_tag(word_tokenize(sentence))
    word_profile = []
    accepted = ['N','J','V']
    for token in tokens:
        tok = token[0]
        if tok not in stop_words and tok.isalpha() and token[1][0] in accepted: word_profile.append(tok)

    distances = {}
    for syns in synsets:
        profiles = []
        
        for item in syns.lemma_names():
            profiles.append(item)
            
        for item in syns.hypernyms():
            for it in item.lemma_names():
                profiles.append(it)
                
        for item in syns.hyponyms():
            for it in item.lemma_names():
                profiles.append(it)        
        
        distances[syns] = get_distance(word_profile, profiles)
    print(distances)
    return max(distances.items(), key=operator.itemgetter(1))[0]
    
        
                                
    
def pluralize(word, pos_tag_orig, target):
    import inflect
    p = inflect.engine()
    if pos_tag_orig == 'NNS':
        return p.plural(target)
    else:
        return target
       
def get_synonyms(word, sentence, pos_tag):
    #synsets = get_synsets(word, pos_tag)[0]  
    synsets = word_sense_profile(word, sentence, pos_tag)
    print(synsets.lemma_names())
    return synsets.lemma_names()

def get_synonyms_without_word_sense(word, sentence, pos_tag):
    synsets = get_synsets(word, pos_tag)[0]  
    #synsets = word_sense_profile(word, sentence, pos_tag)
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
        if token[1].startswith('N') and token[0].lower() not in stop_words: 
            syns = get_synonyms(token[0], sentence, wordnet.NOUN)
            for synonyms in syns:
                target = pluralize(token[0], token[1], synonyms)
                if not(str.lower(str(target)) == str.lower(str(token[0]))):
                    target = ' '.join(target.split('_'))
                    questions.append(generate_ques(sentence, token[0], target))
    print(questions)
    
    
    questions = []
    for token in tagged:
        if token[1].startswith('N') and token[0].lower() not in stop_words: 
            syns = get_synonyms_without_word_sense(token[0], sentence, wordnet.NOUN)
            for synonyms in syns:
                target = pluralize(token[0], token[1], synonyms)
                if not(str.lower(str(target)) == str.lower(str(token[0]))):
                    target = ' '.join(target.split('_'))
                    questions.append(generate_ques(sentence, token[0], target))
    print(questions)

def hypernym_substitution(sentence):
    tagged = pos_tag(word_tokenize(sentence))
    questions = []
    for token in tagged:
        if token[1].startswith('N') and token[0].lower() not in stop_words:
            syns = get_hypernyms(token[0], wordnet.NOUN)
            for hyper in syns:
                nms = hyper.lemma_names()
                for hypernyms in nms:
                    target = pluralize(token[0], token[1], hypernyms)
                    if not(str(target) == str(token[0])):
                        target = ' '.join(target.split('_'))
                        questions.append(generate_ques(sentence, token[0], target))
    #print(sentence)
    print(questions)    

def entailment_substitution(sentence):
    tagged = pos_tag(word_tokenize(sentence))
    questions = []
    for token in tagged:
        if token[1].startswith('V') and token[0].lower() not in stop_words:
            syns = get_entailments(token[0], wordnet.VERB)
            for entail in syns:
                nms = entail.lemma_names()
                for entailments in nms:
                    target = pluralize(token[0], token[1], entailments)
                    if not(str(target) == str(token[0])):
                        target = ' '.join(target.split('_'))
                        questions.append(generate_ques(sentence, token[0], target))
    #print(sentence)
    print(questions)  
    
def substitutions(sentence):
    print(sentence)
    sentence = sentence.lower()
    print("Entailment substitutions:")
    entailment_substitution(sentence)
    print("Hypernym substitutions:")
    hypernym_substitution(sentence)
    print("Synonym substitutions:")
    synonym_substitution(sentence)
    
    
    
if __name__ == '__main__':
    data = json.load(open('/home/deshana/Code/data/mscoco/v2_OpenEnded_mscoco_train2014_questions.json'))
    stop_words = set(stopwords.words('english'))


    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help="path to annotations file")
    
    args = parser.parse_args()
    row = data['questions'][args.index]
    sentence = row['question']
    model=api.load("word2vec-google-news-300")
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
