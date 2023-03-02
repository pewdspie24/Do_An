from torchvision import models
from torchsummary import summary
rn18 = models.resnet101(pretrained=True).to('cuda:0')
children_counter = 0
for n,c in rn18.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n,)
    children_counter+=1

summary(rn18, (3,224,224))