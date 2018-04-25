# Using PlacesCNN for scene classification
# https://github.com/CSAILVision/places365
# Download the following files from: http://places2.csail.mit.edu/models_places365/
# whole_resnet18_places365_python36.pth.tar


import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import sys
sys.path.append('cocoapi/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io as io
import random

def load_model():# th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = 'whole_%s_places365_python36.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    useGPU = 0
    if useGPU == 1:
        model = torch.load(model_file)
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

    ## assume all the script in python36, so the following is not necessary
    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    #from functools import partial
    #import pickle
    #pickle.load = partial(pickle.load, encoding="latin1")
    #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    #torch.save(model, 'whole_%s_places365_python36.pth.tar'%arch)

    model.eval()
    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return model,classes

# load the test image

def load_img():
    dataDir='/Users/tushar/Downloads/VQA-DataAugmentation'
    dataType='train2014'
    annFile='{}/instances_{}.json'.format(dataDir,dataType)

    coco=COCO(annFile)
    imgIds = coco.getImgIds()
    img = [coco.loadImgs(id)[0] for id in imgIds]
    return img


def get_QA(img):
    QA = {}
    model,classes = load_model()
    
    #img_url = img['coco_url']
    #os.system('wget ' + img_url)
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #img_name = 'test.jpg'
    #if not os.access(img_name, os.W_OK):
    #    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    #    os.system('wget ' + img_url)
    variations = [ ("What is this place?"),
                  ("What is the scene in the image?"),
                  ("Where is this image taken?" ),
                  ("What place is shown in the picture?")]

    #img = Image.open(img_name)
    for i in range(0,len(img)):
        QA[i] = {}
        I = io.imread(img[i]['coco_url'])
        im = Image.fromarray(np.uint8((I)))
        #im.save("test1.jpg")
        input_img = V(centre_crop(im).unsqueeze(0), volatile=True)

        # forward pass
        try:
            logit = model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
        except:
            print("img id didn't work: %d"%i)
            #plt.axis('off')
            #plt.imshow(I)
            #plt.show()
                
        # output the prediction
        #for i in range(0, 5):
        #    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        #plt.axis('off')
        #plt.imshow(I)
        #plt.show()
        QA[i]['question'] = random.choice(variations) 
        QA[i]['answer'] = classes[idx[0]]
    return QA

if __name__ == '__main__':
    
    img = load_img()
    QA = get_QA(img)
    for i in range(0,len(QA)):
        print("Question: " + QA[i]['question'])
        print("Answer: " + QA[i]['answer'])   
