import torch
file=torch.load("resources/phones_bert/3-bert/shoulinrui.m4a_0002473600_0002630080.wav.pt",weights_only=True)
print(file)
print(file.shape)