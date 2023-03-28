import torch
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

# import extractor hook functions
from extractor_utils import SaveOutput

import random
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/VGGish/'
if not os.path.exists(RESULTDIR):
	os.makedirs(RESULTDIR)
DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'

files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
wav_files = [f for f in files if f.endswith('wav')]

model = torch.hub.load('harritaylor/torchvggish', 'vggish')

### LOOP OVER AUDIO FILES ###
for filename in wav_files:
	model.eval()

	# Write hooks for the model
	save_output = SaveOutput(avg_type='avg')
	
	hook_handles = []
	layer_names = []
	for idx, layer in enumerate(model.modules()):
		layer_names.append(layer)
		# print(layer)
		if isinstance(layer, torch.nn.modules.conv.Conv2d):
			print('Fetching conv handles!\n')
			handle = layer.register_forward_hook(save_output)
			hook_handles.append(handle)
		if type(layer) == torch.nn.modules.ReLU:
			print('Fetching ReLu handles!\n')
			handle = layer.register_forward_hook(save_output)
			hook_handles.append(handle)
		if type(layer) == torch.nn.modules.MaxPool2d:
			print('Fetching MaxPool2d handles!\n')
			handle = layer.register_forward_hook(save_output)
			hook_handles.append(handle)
		if type(layer) == torch.nn.modules.Linear:
			print('Fetching Linear handles!\n')
			handle = layer.register_forward_hook(save_output)
			hook_handles.append(handle)
	
	# Run the forward pass
	out_features = model.forward(DATADIR + filename)
	processed_features = np.mean((out_features.detach().numpy()), axis=0)
	
	# Detach activations
	detached_activations = save_output.detach_activations()
	
	# Add the post-processed features (AudioSet embeddings, performs PCA, whitening and quantization)
	detached_activations['Post-Processed_Features'] = processed_features
	
	# Sanity check sizes:
	for k, v in detached_activations.items():
		print(f'Shape {k}: {v.shape}')
	
	# Store and save activations
	# Get identifier (sound file name)
	id1 = filename.split('/')[-1]
	identifier = id1.split('.')[0]
	
	save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
