import visual_genome.local as vg
import random
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from collections import defaultdict
import inflect


p = inflect.engine()
images = list(filter(lambda x: x.coco_id, vg.get_all_image_data(data_dir='data/')))


def clean_sentence(sentence):
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    new_tokens = []
    for i in range(len(tokens)):
        if 'N' in tags[i][1] or 'TO' in tags[i][1]:
            new_tokens.append(tokens[i])
    return ' '.join(new_tokens)


def get_vg_questions(image_id):
    scene_graph = vg.get_scene_graph(image_id, images='data/', image_data_dir='data/by-id/')
    sentences = scene_graph.relationships 
    knowledge = []
    adjectives = {}
    attributes = scene_graph.attributes
    for attr in attributes:
        subject = str(attr.subject)
        adjectives[subject] = list(attr.attribute)
    for sentence in sentences:
        subject = str(sentence.subject)
        predicate = str(sentence.predicate).lower()
        object = str(sentence.object)
        knowledge.append((subject, predicate.lower(), object))
    return knowledge, adjectives


def create_graph(knowledge, adjectives):
    graph = defaultdict(dict)
    for (sub, pred, obj) in knowledge:
        if sub in graph:
            graph[sub]['adj'].append(obj)
            graph[sub]['edge'].append(pred)
        else:
            graph[sub]['adj'] = [obj]
            graph[sub]['edge'] = [pred]
            if sub in adjectives:
                tags = pos_tag(adjectives[sub])
                tags = filter(lambda x: x[1] == 'JJ', tags)
                tags = list(map(lambda x: x[0], tags))
                graph[sub]['attr'] = tags
    return graph


def dfs(graph, start):
    stack = [(start, [start], [])]
    while stack:
        (vertex, path, join) = stack.pop()
        indices = []
        for i in range(len(graph[vertex]['adj'])):
            if graph[vertex]['adj'][i] not in set(path):
                indices.append(i)
        for idx in indices:
            next = graph[vertex]['adj'][idx] 
            if next not in graph:
                 sentence = []
                 tmp = path[1:]
                 for i in range(len(join)):
                     sentence.append(join[i])
                     if 'attr' in graph[tmp[i]]:
                         attr = graph[tmp[i]]['attr']
                         sentence.append(', '.join(random.sample(attr, min(2, len(attr)))))
                     sentence.append(tmp[i])
                 sentence.append(graph[vertex]['edge'][idx])
                 sentence.append(next)
                 yield ' '.join(sentence).strip() + " ?"
            else:
                 stack.append((next, path + [next], join + [graph[vertex]['edge'][idx]]))


image = random.choice(images)
knowledge, adjectives = get_vg_questions(image.id)
graph = create_graph(knowledge, adjectives)
questions, answers = [], []
for key in graph.keys():
    filter_word = ' '.join(map(lambda x: x[0], filter(lambda x: x[1] == 'NNS', pos_tag(word_tokenize(key)))))
    if len(filter_word) == 0:
        filter_word = key
    else:
        try:
            filter_word = p.singular_noun(filter_word)
        except:
            filter_word = key
    try:
        synsets = wn.synsets(filter_word)
    except:
        synsets = []
    root = wn.synset("person.n.01")
    if len(synsets) == 0:
        prefix = "What"
    else:
        lch = wn.synsets('person')[0].lowest_common_hypernyms(synsets[0])
        if len(lch) == 0:
            prefix = "What"
        elif lch[0] == root:
            prefix = "Who"
        else:
            prefix = "What"
    for sent in dfs(graph, key):
        tags = pos_tag(word_tokenize(sent)[:1])
        if tags[0][1] in ('VBG', 'IN', 'TO', 'JJ', 'DT') or 'ing' in word_tokenize(sent)[:1]:
            questions.append(prefix + " is " + sent)
        else:
            questions.append(prefix + " " + sent)
        answers.append(key)
fin = map(lambda x: ' '.join(x), list(set(zip(questions, answers))))
print(image)
print('\n'.join(fin))
