import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import pandas as pd


class VQG_Dataset(data.Dataset):
    
    """ Custom Dataset compatible with torch.utils.data.DataLoader."""
    
    def __init__(self, root, json, data, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            data: preprocessed csv dataset file path
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.data = pd.read_csv(data, names = ['image_id','q_id','question','answer','caption'])
        self.ids = self.data.index.values
        self.vocab = vocab
        self.transform = transform
        
        
    def __getitem__(self, index):
        """Returns one data triplet (image, caption, QA)."""
        coco = self.coco
        vocab = self.vocab
        dataset = self.data
        im_id = dataset['image_id'][index]
        path = coco.loadImgs(int(im_id))[0]['file_name']
        caption = dataset['caption'][index]
        question = dataset['question'][index]
        answer = dataset['answer'][index]

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target_caption = torch.Tensor(caption)

        # Combine question and answer (string) and convert to word ids.
        tokens = nltk.tokenize.word_tokenize(str(question).lower())
        QA = []
        QA.append(vocab('<start>'))
        QA.extend([vocab(token) for token in tokens])
        QA.append(vocab('<end>'))

        tokens = nltk.tokenize.word_tokenize(str(answer).lower())
        QA.append(vocab('<start>'))
        QA.extend([vocab(token) for token in tokens])
        QA.append(vocab('<end>'))

        target_QA = torch.Tensor(QA)
        return image, target_caption, target_QA
    
    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption, qa). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
            - qa: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets_c: torch tensor of shape (batch_size, padded_length).
        lengths_c: list; valid length for each padded caption.
        targets_q: torch tensor of shape (batch_size, padded_length).
        lengths_q: list; valid length for each padded qa.
        vocab_words: list of unique words in all captions
    """
    # Sort a data list by qa length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, captions, qa = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D to 2D tensor). 
    lengths_c = [len(cap) for cap in captions]
    unique_ids = []
    for cap in captions:
        unique_id = np.unique(cap.numpy())
        unique_ids = unique_ids + list(unique_id)
    unique_ids = np.unique(unique_ids)
    targets_c = torch.zeros(len(captions), max(lengths_c)).long()    
    for i, cap in enumerate(captions):
        end = lengths_c[i]
        targets_c[i, :end] = cap[:end]

    # Merge qa (from tuple of 1D tensor to 2D tensor).
    lengths_q = [len(cap) for cap in qa]
    targets_q = torch.zeros(len(qa), max(lengths_q)).long()
    for i, cap in enumerate(qa):
        end = lengths_q[i]
        targets_q[i, :end] = cap[:end] 
    
    return images, targets_c, lengths_c, targets_q, lengths_q, unique_ids



def get_loader(root, json, data, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom VQG_Dataset."""
    # VQG dataset
    vqg = VQG_Dataset(root=root,
                       json=json,
                       data=data,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for VQG_dataset
    # This will return (images, captions, caption_length, QA, QA_length) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # caption_length: list indicating valid length for each caption. length is (batch_size).
    # QA: tensor of shape (batch_size, padded_length).
    # QA_length: list indicating valid length for each QA. length is (batch_size).

    data_loader = torch.utils.data.DataLoader(dataset=vqg, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader