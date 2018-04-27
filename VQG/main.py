import sys
import os
import preprocess_get_model
import skipthought_encoder
import skimage.io as io
import random
sys.path.append("cocoapi/PythonAPI")
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import time
import pickle



def main():
	
	model = preprocess_get_model.model()
	embeddings = {}
	
	dataDir='/Users/tushar/Downloads'
	dataType='train2014'

	# initialize COCO api for caption annotations
	annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType) 
	annFile_instances='{}/annotations/instances_{}.json'.format(dataDir,dataType)
	#print(annFile_instances)
	coco=COCO(annFile_instances)
	coco_caps=COCO(annFile)
	imgIds = coco.getImgIds()
	img = [coco.loadImgs(id)[0] for id in imgIds]
	#idx = np.random.randint(0, len(img))
	for i in range(0,len(img)):
		embeddings[i] = {}

		#coco_caps.showAnns(anns)
		I = io.imread(img[i]['coco_url'])
		im = Image.fromarray(np.uint8((I)))
		p_img = preprocess_get_model.pre_processing(im)
		img_embeddings = model(p_img)
		size = img_embeddings.data.shape
		#print(size) #  

		annIds = coco_caps.getAnnIds(img[i]['id'])
		anns = coco_caps.loadAnns(annIds)
		max_l = 0
		for ann in anns:
			if len(ann['caption']) > max_l:
				max_l = len(ann['caption'])
				max_ann = ann
		cap = max_ann['caption']
		cap_embeddings = skipthought_encoder.embed(cap)

		embeddings[i]['img_embeddings'] = img_embeddings
		embeddings[i]['cap_embeddings'] = cap_embeddings
		embeddings[i]['image_id'] = img[i]['id']

	return embeddings


if __name__ == "__main__":
	start = time.time()
	vectors = main()
	print("Main :")
	for key in vectors.keys():
		print(vectors[key]['img_embeddings'].data.shape)
		print(vectors[key]['cap_embeddings'].size())
		print(vectors[key]['image_id'])
	pickle.dump(vectors, open( "embeddings.p", "wb" ) )	
	end = time.time()
	print(end - start)	
