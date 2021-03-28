import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join

# import extractor hook functions
from extractor_utils import SaveOutput

RESULTDIR = '/Users/gt/Documents/GitHub/control-neural/control-neural/model-actv-control/VGGish/'
DATADIR = '/Users/gt/Documents/GitHub/control-neural/data/stimuli/165_natural_sounds_16kHz/'

files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
wav_files = [f for f in files if f.endswith('wav')]

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

### LOOP OVER AUDIO FILES ###
for filename in wav_files:
	# write hooks for the model
	save_output = SaveOutput()
	
	hook_handles = []
	layer_names = []
	for idx, layer in enumerate(model.modules()):
		layer_names.append(layer)
		# print(layer)
		if isinstance(layer, torch.nn.modules.conv.Conv2d):
			print('Fetching conv handles!\n')
			handle = layer.register_forward_hook(save_output)  # save idx and layer
			hook_handles.append(handle)
		if type(layer) == torch.nn.modules.ReLU:
			print('Fetching ReLu handles!\n')
			handle = layer.register_forward_hook(save_output)  # save idx and layer
			hook_handles.append(handle)
		if type(layer) == torch.nn.modules.MaxPool2d:
			print('Fetching MaxPool2d handles!\n')
			handle = layer.register_forward_hook(save_output)  # save idx and layer
			hook_handles.append(handle)
		if type(layer) == torch.nn.modules.Linear:
			print('Fetching Linear handles!\n')
			handle = layer.register_forward_hook(save_output)  # save idx and layer
			hook_handles.append(handle)
	
	out_features = model.forward(DATADIR + filename)
	processed_features = np.mean((out_features.detach().numpy()), axis=0)
	
	# act_keys = list(save_output.activations.keys())
	# act_vals = save_output.activations
	
	# detach activations
	detached_activations = save_output.detach_activations()
	
	# add the postprocssed features
	detached_activations['PostProcessed_Features'] = processed_features
	
	# sanity check sizes:
	for k, v in detached_activations.items():
		print(f'Shape {k}: {v.shape}')
	
	# store and save activations
	# get identifier (sound file name)
	id1 = filename.split('/')[-1]
	identifier = id1.split('.')[0]
	
	save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
