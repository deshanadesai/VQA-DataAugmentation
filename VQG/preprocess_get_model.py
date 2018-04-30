import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image


class ResNet101Bottom(nn.Module):

    def __init__(self, original_model):
        super(ResNet101Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


def pre_processing(img):
    scalar = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    #img = Image.open(img)
    t_img = Variable(normalize(to_tensor(scalar(img))).unsqueeze(0))
    return t_img


def model():
    res101_model = models.resnet101(pretrained=True)
    res101_conv2 = ResNet101Bottom(res101_model)
    return res101_conv2

'''def main():
    m= model()
    print(m)
    print("\nout of here\n")
    print(models.resnet101())

if __name__=='__main__':
    main()'''
