import argparse
import random
import sys
sys.path.append('../cocoapi/PythonAPI')
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io as io
from skimage import color
from matplotlib.patches import Polygon
from matplotlib.path import Path
from sklearn.cluster import DBSCAN
import webcolors
import csv
from collections import defaultdict

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
    variations = [ ("What is the color of the " + obj + "?","None"),
                  ("What color is the " + obj + "?","None"),
                  ("What is the dominant color of the " + obj + "?","Dominant"),
                  ("What color most stands out in the " + obj + "?","Dominant")]
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
        if len(points_region)==0:
            continue
        print("Running DBSCAN..")
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
    
    
    
def get_color_from_image(annoPath, imgId):
    coco = COCO(annoPath)
    imgIds = coco.getImgIds(imgIds=[imgId])
    print(imgIds)
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    I = io.imread(img['coco_url'])
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    #plt.imshow(I)
    #plt.axis('off')

    (m, n, channels) = I.shape
    # make a canvas with coordinates
    x, y = np.meshgrid(np.arange(m), np.arange(n))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    I_copy = I.copy()
    
    questions = []
    answers = []
    
    for ann in anns:
        print ("Processing segmented area..")
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        cat = coco.loadCats(ann['category_id'])
        obj = cat[0]['name']
        supercat = cat[0]['supercategory']
        question = gen_question(obj, supercat)

        if type(ann['segmentation']) == list:
            seg = ann['segmentation'][0]
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
            P = Path(poly)

            region = points[np.where(P.contains_points(points))]
            points_region = []
            for r in region:
                points_region.append(I[r[1], r[0]])
            if len(points_region)==0:
                continue
            print("Running DBSCAN..")
            Y = cluster(points_region)
            names = colour(Y)
            qtype = question[1]
            question = question[0]
            
            #answer = gen_answer(names)
                           
            if len(names) == 1 or qtype == "Dominant":
                answer = [colors_list[names[0]][1]]
            elif len(names) == 2:
                answer = [colors_list[names[0]][0] + " and " + colors_list[names[1]][0], 
                          colors_list[names[0]][1] + " and " + colors_list[names[1]][0],
                          colors_list[names[0]][0] + " and " + colors_list[names[1]][1],
                          colors_list[names[0]][1] + " and " + colors_list[names[1]][1]]
                answer = list(set(answer))
            else: continue
        
        print("Qn. " , question)
        questions.append(question)
        print("Ans. " , answer)   
        answers.append(answer)
    return zip(question,answer)
    #coco.showAnns(anns)
    #plt.imshow(I)
    #plt.axis('off')
    #plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    dataDir = '/home/deshana/Code/data/mscoco'
    dataType = 'train2014'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    parser.add_argument("--annotation_path", help="path to annotations file", default = annFile)
    parser.add_argument("image_path", help="path to image files", type=int)
    args = parser.parse_args()
    process_colors_list()
    colors = get_colors_from_image(args.annotation_path, args.image_path)
    
