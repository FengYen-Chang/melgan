import torch
import torch.onnx
import numpy as np
from model.generator import Generator

import os
if not os.path.exists('onnx'):
    os.mkdir('onnx')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mel = torch.randn(1, 80, 200)

if torch.cuda.is_available():
    melgan = torch.hub.load('seungwonpark/melgan', 'melgan')
    print('Moving data & model to GPU')
    melgan = melgan.cuda()
    mel = mel.cuda()
else:
    print('Moving data & model to CPU')
    melgan = Generator(80)
    checkpoint = torch.hub.load_state_dict_from_url('https://github.com/seungwonpark/melgan/releases/download/v0.3-alpha/nvidia_tacotron2_LJ11_epoch6400.pt', map_location="cpu")
    melgan.load_state_dict(checkpoint["model_g"])
    melgan.eval(inference=True)

opset_version = 11
torch.onnx.export(melgan, mel, "onnx/melgan.onnx",
    opset_version=opset_version,
    do_constant_folding=True,
    input_names=["mel"],
    output_names=["mel_output"])

