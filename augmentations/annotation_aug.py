import argparse
import sys
import random
import inflect
import config
sys.path.append('../cocoapi/PythonAPI')
from pycocotools.coco import COCO
from os import listdir
from collections import defaultdict, Counter, OrderedDict
from utils import *


p = inflect.engine()

def get_image_ids(path):
    ids = list(map(lambda x: int(x.strip()), open(path).readlines()))
    return ids


def get_category_info(coco, cat_id):
    try:
        cats = coco.loadCats(cat_id)[0]
        del cats['id']
        return cats
    except:
        return {}


def get_objects_from_image(annotation_file, image_path):
    coco = COCO(annotation_file)
    img_ids = get_image_ids(image_path)
    annIds = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(annIds)
    objects = defaultdict(list)
    cats = coco.loadCats(coco.getCatIds())
    cat_ids = {cat['id']: cat for cat in cats}
    for key in cat_ids:
       del cat_ids[key]['id']
    img_data = {}
    for img_id in img_ids:
        img_data[img_id] = coco.loadImgs(img_id)[0]
    for ann in anns:
        ann.update(cat_ids[ann['category_id']])
        objects[ann['image_id']].append(ann)    
    return objects, img_data


def get_counting_questions(objects):
    variations = ['How many {} are there in the image?',
                  'How many {} are visible in this picture?',
                  'How many {} are there?', 'How many {} do you see?',
                  'How many {} are in the photo?', 'How many {}?',
                  'How many {} can you count?', 'How many {} are shown',
                  'How many {} are here?', 'How many {} are depicted?',
                  'How many {} are there in the scene?', 'How many {} in total?']
    data = defaultdict(list)
    for key in objects:
        counter = Counter(map(lambda x: x['name'], filter(lambda x: x['area'] > 2000, objects[key])))
        counter = {key: counter[key] for key in counter if counter[key] > 1 or random.random() < 0.1}
        for object in counter:
            q = random.choice(variations).format(p.plural_noun(object))
            a = helper_ans_string([p.number_to_words(counter[object])])
            data[key].append((q, a))
    return data

 
def get_obj_recognition_questions(objects):
    variations = ['What {} is in the image?', 'What type of {} is this?',
                  'What type of {} is visible here?', 'What type of {} is pictured?',
                  'What type of {} are shown?', 'What type of {}?',
                  'What kind of {} is visible in the photo?', 'What kind of {} is that?',
                  'What kind of {} is in the photo?', 'What {} is shown in the image?',
                  'What {} is here?', 'What kind of {} can be seen?']
    data = defaultdict(list)
    for key in objects:
        supcat_count = Counter(map(lambda x: x['supercategory'], 
                               filter(lambda x: 'person' not in x['name'], objects[key])))
        cat_count = Counter(map(lambda x: (x['supercategory'], x['name']), 
                            filter(lambda x: 'person' not in x['name'], objects[key])))
        for (sc, c) in cat_count:
            if supcat_count[sc] != cat_count[(sc, c)]:
                continue
            q = random.choice(variations).format(sc)
            a = helper_ans_string([c])
            data[key].append((q, a))
    return data


def get_yes_no_questions(objects, cats):
    variations = ['Is there any {} in the picture?', 'Is there a {}?', 
                  'Is any {} shown?', 'Can you see a {} in this photo?',
                  'Is there a {} in this scene?', 'Is there a {} here?',
                  'Are there any {} in the photo?', 'Are there {} here?',
                  'Are there any {} visible?', 'Are there any {} in the image?',
                  'Are there {}?', 'Are there any {} shown here?']
    data = defaultdict(list) 
    for key in objects:
        objs = set(map(lambda x: x['name'], objects[key]))
        neg_objs = list(cats - objs)
        objs = set(map(lambda x: x['name'], filter(lambda x: x['area'] > 2000, objects[key])))
        for obj in objs:
            qlen = len(variations)
            qid = random.choice(range(qlen))
            if qid >= qlen/2:
                obj = p.plural_noun(obj)
            q = variations[qid].format(obj)
            a = helper_ans_string(["yes"])
            data[key].append((q, a))
            qid = random.choice(range(qlen))
            neg_obj = random.choice(neg_objs)
            if qid >= qlen/2:
                neg_obj = p.plural_noun(neg_obj)
            q = variations[qid].format(neg_obj)
            a = helper_ans_string(["no"])
            data[key].append((q, a))
    return data 


def create_graph(obj, others):
    graph = defaultdict(list)
    name, (x, y, w, h) = obj
    for name, (x1, y1, w1, h1) in others:
       if x1 < x:
           if y1 >= y - h/4 and y1 <= y + h/4 or (abs(h1-h) > min(h1, h) and x1 < x-w/2):
               graph['left'].append(name)
           elif y1 < y - h/4:
               graph['top'].append(name)
           else:
               graph['bottom'].append(name)
       else:
           if y1 >= y - h/4 and y1 <= y + h/4 or (abs(h1-h) > min(h1, h) and x1 > x+w/2):
               graph['right'].append(name)
           elif y1 < y - h/4:
               graph['top'].append(name)
           else:
               graph['bottom'].append(name)
    return graph 


def get_positional_questions(objects, img_data):
    variations = ['What is to the {} of {}?', 'Can you tell me what is to the {} of {} in the photo?',
                  'What is to the {} of {} in the image?', 'What is to the {} of {} in the picture?']
    abs_variations = ['What is the {}most object in image?', 'What is the object in the far {} of the image?']
    data = defaultdict(list)
    for key in objects:

        # absolute
        left_obj = list(filter(lambda x: x['bbox'][0] + x['bbox'][2]/2 < .25*img_data[key]['width'], objects[key]))
        if left_obj:
            q = random.choice(abs_variations).format('left')
            a = helper_ans_string(list(map(lambda x: x['name'], left_obj)))
            data[key].append((q, a))
        right_obj = list(filter(lambda x: x['bbox'][0] + x['bbox'][2]/2 > .75*img_data[key]['width'], objects[key]))
        if right_obj:
            q = random.choice(abs_variations).format('right')
            a = helper_ans_string(list(map(lambda x: x['name'], right_obj)))
            data[key].append((q, a))
        top_obj = list(filter(lambda x: x['bbox'][1] + x['bbox'][3]/2 < .25*img_data[key]['height'], objects[key]))
        if top_obj:
            q = random.choice(abs_variations).format('top')
            a = helper_ans_string(list(map(lambda x: x['name'], top_obj)))
            data[key].append((q, a))
        bottom_obj = list(filter(lambda x: x['bbox'][1] + x['bbox'][3]/2 > .75*img_data[key]['height'], objects[key])) 
        if bottom_obj:
            q = random.choice(abs_variations).format('bottom')
            a = helper_ans_string(list(map(lambda x: x['name'], bottom_obj)))
            data[key].append((q, a))

        # relative
        names = list(map(lambda x: x['name'], objects[key]))
        count = { k:v for k, v in Counter(names).items() if v > 1}
        if len(count.keys()) > 0:
            continue
        centers = list(map(lambda x: (x['bbox'][0] + x['bbox'][2]/2, x['bbox'][1] + x['bbox'][3]/2, x['bbox'][2], x['bbox'][3]), 
                      objects[key]))
        zipped = list(zip(names, centers))
        for i in range(len(zipped)):
            graph = create_graph(zipped[i], zipped[:i] + zipped[i+1:])
            for dir in graph.keys():
                q = random.choice(variations).format(dir, zipped[i][0])
                a = list(set(graph[dir]))
                if len(a) == 1 and 'person' in a:
                    q = q.replace("What", "Who")
                    q = q.replace("what", "who")
                a = helper_ans_string(a) 
                data[key].append((q, a))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path", help="path to annotations file")
    parser.add_argument("image_path", help="path to image files")
    args = parser.parse_args()
    coco = COCO(args.annotation_path)
    cats = set([cat['name'] for cat in coco.loadCats(coco.getCatIds())])
#    ids = get_image_ids(args.image_path)
    objects, img_data= get_objects_from_image(args.annotation_path, args.image_path)
    DataLoader = data_loader(config.question_train_path, config.answer_train_path)
    data = get_counting_questions(objects)
    add_to_dataset(DataLoader, data)
    print("Done... Counting Questions")
    data = get_obj_recognition_questions(objects)
    add_to_dataset(DataLoader, data)
    print("Done... Obj Recog Questions")
    data = get_yes_no_questions(objects, cats)
    add_to_dataset(DataLoader, data)
    print("Done... Yes/No Questions")
    data = get_positional_questions(objects, img_data)
    add_to_dataset(DataLoader, data)
    print("Done... Positional Questions")
    DataLoader.dump_ans_train_json("answers_annotations.json")
    DataLoader.dump_qns_train_json("questions_annotations.json")
    