import argparse
import random
import sys
sys.path.append('../cocoapi/PythonAPI')
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import color
from matplotlib.patches import Polygon
from matplotlib.path import Path
from sklearn.cluster import DBSCAN
import webcolors
import csv
from collections import defaultdict
from collections import Counter
import random
from utils import *
import time

colors_list = defaultdict(list)

def process_colors_list():
    global colors_list
    f = open("colors_list.csv","r")
    reader = csv.reader(f)
    for row in reader:
        if row[0]=="dirty":continue
        colors_list[row[0]] = row[1:]
    f.close()
    
# TODO : What color is the <OBJ> <POS>? (Positional)
# TODO : Is this a multicolored <OBJ>? (Yes or No)
# TODO : What are the <ATTR>? (Descriptive)
# TODO : How many <OBJ> are <COL>? (Counting)


def gen_question(obj, supercat):
    endings = ["","", " in here", "", " in this scene", " in this image", " in this photo", " in this picture"]
    end = random.choice(endings)
               
    variations = [ ("What is the color of the " + obj + end + "?","None"),
                  ("What color is the " + obj + end + "?","None"),
                  ("What is the dominant color of the " + obj + end + "?","Dominant"),
                  ("What color most stands out in the " + obj + end + "?","Dominant")]
    return random.choice(variations)

# Not used.
def gen_answer(names):     
    if len(names) == 1:
        return names[0]
    else:
        return ", ".join(names)
    
def cluster(points_region):
    clt = DBSCAN(eps=20, min_samples=len(points_region)//20)
    clt = clt.fit(points_region)
    core_samples_mask = np.zeros_like(clt.labels_, dtype=bool)

    core_samples_mask[clt.core_sample_indices_] = True

    labels = clt.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    X = np.array(points_region)
    Y = []
    for k in set(labels):
        class_member_mask = (labels == k)
        if k == -1:
            continue
        Y.append(X[class_member_mask & core_samples_mask])
    return Y


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def colour(Y):
    names = []
    for item in Y:
        c = np.median(item, axis=0)
        requested_colour = (int(c[0]), int(c[1]), int(c[2]))

        actual_name, closest_name = get_colour_name(requested_colour)
        names.append(closest_name)
    return names

def get_color_from_anno(annID, I):
    ann = coco.loadAnns(annID)[0]
    if type(ann['segmentation']) == list:
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        P = Path(poly)

        region = points[np.where(P.contains_points(points))]
        points_region = []
        for r in region:
            points_region.append(I[r[1], r[0]])
        
        #if len(points_region)==0:
        #    continue
        #print("Running DBSCAN..")
        Y = cluster(points_region)
        names = colour(Y)
    else:
        raise 'segmentation not of type list'
       
    if len(names) == 1 or qtype == "Dominant":
        answer = [colors_list[names[0]][1]]
    elif len(names) == 2:
        answer = [colors_list[names[0]][0] + " and " + colors_list[names[1]][0], 
                  colors_list[names[0]][1] + " and " + colors_list[names[1]][0],
                  colors_list[names[0]][0] + " and " + colors_list[names[1]][1],
                  colors_list[names[0]][1] + " and " + colors_list[names[1]][1]]
        answer = list(set(answer))
    else: answer = []
    
    return answer
    
# Filter if multiple objects are there.
# Filter if size of object too small.
# Filter if too many colors.
# Limit # of Questions.
    
def get_color_from_image(coco, imgId):
    global data
    #imgIds = coco.getImgIds(imgIds=[imgId])
    #print(imgIds)
    img = coco.loadImgs(imgId)[0]
    I = io.imread(img['coco_url'])
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    (m, n, channels) = I.shape
    # make a canvas with coordinates
    x, y = np.meshgrid(np.arange(m), np.arange(n))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    
    # filter annotations.
    
    
    filtered_ann = []
    category_ids = []
    blacklist_ids = []
    for ann in anns:
        if ann['area']<=2000 and ann['area']>=4000:
            continue
            
        if ann['category_id'] in category_ids:
            blacklist_ids.append(ann['category_id'])
            continue
        
        filtered_ann.append(ann)
        category_ids.append(ann['category_id'])
    
#     copy = filtered_ann
#     for f in filtered_ann:
#         if f['category_id'] in blacklist_ids:
#             copy.remove(f)
#     anns = copy
    anns = list(filter(lambda x: x['category_id'] not in blacklist_ids, filtered_ann))
    
    if len(anns)>2:
        anns = random.sample(anns, 2)
    for ann in anns:
        cat = coco.loadCats(ann['category_id'])
        obj = cat[0]['name']
        supercat = cat[0]['supercategory']
        area = ann['area']
        
        question = gen_question(obj, supercat)

        if type(ann['segmentation']) == list:
            seg = ann['segmentation'][0]
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
            P = Path(poly)

            region = points[np.where(P.contains_points(points))]
            points_region = []
            try:
                for r in region:
                    points_region.append(I[r[1], r[0]])
            except:
                print("Exception caught at Image boundaries")
                continue
            if len(points_region)==0:
                continue
            Y = cluster(points_region)
            names = colour(Y)
            qtype = question[1]
            question = question[0]
                                       
            if len(names) == 1 or qtype == "Dominant":
                answer = [colors_list[names[0]][1], colors_list[names[0]][0]]
            elif len(names) == 2:
                answer = [colors_list[names[0]][0] + " and " + colors_list[names[1]][0], 
                          colors_list[names[0]][1] + " and " + colors_list[names[1]][0],
                          colors_list[names[0]][0] + " and " + colors_list[names[1]][1],
                          colors_list[names[0]][1] + " and " + colors_list[names[1]][1]]
                answer = list(set(answer))
            else: continue
        else: 
            print("Found Ann Mask. Skipping.") 
            continue
        q = question
        a = helper_ans_string(answer)
        
        data[imgId].append((q,a))
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #dataDir = '/home/deshana/Code/data/mscoco'
    #dataDir = '/Users/deshanadesai/Code/COCO'
    dataDir = '/scratch/abs699'
    dataType = 'train2014'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    parser.add_argument("--annotation_path", help="path to annotations file", default = annFile)
    parser.add_argument("image_path", help="path to image files")
    args = parser.parse_args()
    process_colors_list()
    data = defaultdict(list)
    coco = COCO(args.annotation_path)
    ids = list(map(lambda x: int(x.strip()), open(args.image_path).readlines()))
    from tqdm import tqdm
    ids = random.sample(ids,len(ids)//4)
    for imgid in tqdm(ids):
    	get_color_from_image(coco, imgid)
    
    #print(end-start)
    
    DataLoader = data_loader(config.question_train_path, config.answer_train_path)
    add_to_dataset(DataLoader, data)
    DataLoader.dump_qns_train_json("questions_with_color.json")
    DataLoader.dump_ans_train_json("answers_with_color.json")
        
