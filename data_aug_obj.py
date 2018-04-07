import argparse
import sys
import numpy as np
import random
sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO
from os import listdir
from collections import defaultdict, Counter


def get_image_ids(path):
    files = listdir(path)
    ids = list(map(lambda x: int(x.split("_")[-1].split(".")[0]), 
                   files))
    return ids


def get_category_info(coco, ann):
    try:
        cats = coco.loadCats(ann['category_id'])[0]
        cats['image_id'] = ann['image_id']
        cats['area'] = ann['area']
        return cats
    except:
        return {}


def get_objects_from_image(annotation_file, image_path):
    coco = COCO(annotation_file)
    img_ids = get_image_ids(image_path)
    annIds = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(annIds)
    objects = defaultdict(list)
    for ann in anns:
        objects[ann['image_id']].append(get_category_info(coco, ann))    
    return objects


# TODO Turn to plural
# TODO Add variations
# TODO Format same as the dataset
# TODO Filter questions
def get_counting_questions(objects):
    variations = ['How many {} are there in the image?']
    questions, answers = [], []
    for key in objects:
        counter = Counter(map(lambda x: x['name'], objects[key]))
        for object in counter:
            q = random.choice(variations).format(object)
            a = str(counter[object])
            questions.append({'image_id': key, 'question': q})
            answers.append({'image_id': key, 'answer': a})
    return questions, answers
	        
 
# TODO Add variations
# TODO Format same as the dataset
# TODO Filter questions (not needed mostly)
def get_obj_recognition_questions(objects):
    variations = ['What {} is in the image?']
    questions, answers = [], []
    for key in objects:
        supcat_count = Counter(map(lambda x: x['supercategory'], objects[key]))
        cat_count = Counter(map(lambda x: (x['supercategory'], x['name']), 
                                objects[key]))
        for (sc, c) in cat_count:
            if supcat_count[sc] != cat_count[(sc, c)]:
                continue
            q = random.choice(variations).format(sc)
            a = c
            questions.append({'image_id': key, 'question': q})
            answers.append({'image_id': key, 'answer': a})
    return questions, answers


# TODO Add variation
# TODO Format same as the dataset
# TODO Filter questions (not needed mostly)
def get_yes_no_questions(objects, cats):
    variations = ['Is there a {} in the picture?']
    questions, answers = [], [] 
    for key in objects:
        objs = set(map(lambda x: x['name'], objects[key]))
        for obj in objs:
            q = random.choice(variations).format(obj)
            a = "yes"
            questions.append({'image_id': key, 'question': q})
            answers.append({'image_id': key, 'answer': a})
            neg_obj = random.choice(list(cats - objs))
            q = random.choice(variations).format(neg_obj)
            a = "no"
            questions.append({'image_id': key, 'question': q})
            answers.append({'image_id': key, 'answer': a})
    return questions, answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path", help="path to annotations file")
    parser.add_argument("image_path", help="path to image files")
    args = parser.parse_args()
    objects = get_objects_from_image(args.annotation_path, args.image_path)
