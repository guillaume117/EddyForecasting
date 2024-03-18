
import torchvision
from torchviz import make_dot
import torch
#from CNN.SimpleCNN import SimpleModel

""" This file is used to generate the model graph of the CNN model. """
model = UNet(10)


input_tensor = torch.randn((1,10, 96*4, 192*4))


output = model(input_tensor)
graph = make_dot(output, params=dict(model.named_parameters()))


graph.render('model_graph')


graph.view()