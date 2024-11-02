import torch
file=torch.load("resources/bert_features/shoulinrui.m4a/shoulinrui.m4a_0000063040_0000325440.wav.pt",weights_only=True)
print(file)
print(file.shape)