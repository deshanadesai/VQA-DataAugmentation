import sys
sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


def get_captions_method():
    dataDir='/Users/tushar/Downloads'
    dataType='train2014'

    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    print(annFile)
    annFile_instances='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    print(annFile_instances)
    coco_caps=COCO(annFile)
    coco=COCO(annFile_instances)
    imgIds = coco.getImgIds()
    img = [coco.loadImgs(id)[0] for id in imgIds]
    # load and display caption annotations
    #I = io.imread(img[0]['coco_url'])
    annIds = coco_caps.getAnnIds(img[np.random.randint(0, len(img))]['id']);
    anns = coco_caps.loadAnns(annIds)
    coco_caps.showAnns(anns)
    return anns[0]['caption']
   # plt.imshow(I); plt.axis('off'); plt.show()

'''def main():
    anns=get_captions_method()
    print(anns)

if __name__=='__main__':
    main()'''