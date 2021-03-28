import pickle
import numpy as np
import warnings
from pathlib import Path
import os

class SaveOutput:
	def __init__(self):
		self.outputs = []
		self.activations = {}  # create a dict with module name
		self.detached_activations = None
	
	def __call__(self, module, module_in, module_out):
		"""
		Module in has the input tensor, module out in after the layer of interest
		"""
		self.outputs.append(module_out)
		
		layer_name = self.define_layer_names(module)
		self.activations[layer_name] = module_out
	
	def define_layer_names(self, module):
		layer_name = str(module)
		current_layer_names = list(self.activations.keys())
		
		split_layer_names = [l.split('--') for l in current_layer_names]
		
		num_occurences = 0
		for s in split_layer_names:
			s = s[0]  # base name
			
			if layer_name == s:
				num_occurences += 1
		
		layer_name = str(module) + f'--{num_occurences}'
		
		if layer_name in self.activations:
			warnings.warn('Layer name already exists')
		
		return layer_name
	
	def clear(self):
		self.outputs = []
		self.activations = {}
	
	def get_existing_layer_names(self):
		for k in self.activations.keys():
			print(k)
		
		return list(self.activations.keys())
	
	def return_outputs(self):
		self.outputs.detach().numpy()
	
	def detach_one_activation(self, layer_name):
		return self.activations[layer_name].detach().numpy()
	
	def detach_activations(self, lstm_output='recent'):
		"""
		Detach activations (from tensors to numpy)

		Arguments:
			lstm_output: for LSTM, can output either the hidden states throughout sequence ('sequence')
						or the most recent hidden states ('recent')

		Returns:
			detached_activations = for each layer, the flattened activations
			packaged_data = for LSTM layers, the packaged data
		"""
		detached_activations = {}
		
		relu_4d = [0, 1, 2, 3, 4,
				   5]  # the first 6 ReLu layers are 4d, and needs to be averaged differently than the last 3 ReLu
		
		# testing:
		# activations = self.activations[
		#     'Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))--0'].detach().numpy()
		
		for k, v in self.activations.items():
			# print(f'Shape {k}: {v.detach().numpy().shape}')
			print(f'Detaching activation for layer: {k}')
			if k.startswith('Conv2d') or k.startswith('MaxPool2d'):
				activations = v.detach().numpy()
				
				# reshape to [frames; height (time); channels; mel bins]
				actv_reshape = np.moveaxis(activations, 2, 1)
				
				# expand channels x mel bins
				actv_expand = np.reshape(actv_reshape, [actv_reshape.shape[0], actv_reshape.shape[1],
														actv_reshape.shape[2] * actv_reshape.shape[3]])
				
				# average over height first! and then frames.
				actv_avg1 = np.mean(actv_expand, axis=1)
				actv_avg2 = np.mean(actv_avg1, axis=0)
				
				detached_activations[k] = actv_avg2
			
			if k.startswith('ReLU'):
				k_split = int(k.split('--')[1])
				if k_split in relu_4d:
					activations = v.detach().numpy()
					
					# reshape to [frames; height (time); channels; mel bins]
					actv_reshape = np.moveaxis(activations, 2, 1)
					
					# expand channels x mel bins
					actv_expand = np.reshape(actv_reshape, [actv_reshape.shape[0], actv_reshape.shape[1],
															actv_reshape.shape[2] * actv_reshape.shape[3]])
					
					# average over height first! and then frames.
					actv_avg1 = np.mean(actv_expand, axis=1)
					actv_avg2 = np.mean(actv_avg1, axis=0)
					
					detached_activations[k] = actv_avg2
				else:
					print(f'Not 4d ReLu. Layer number {k_split}')
					activations = v.detach().numpy()
					
					# mean over frames
					actv_avg1 = np.mean(activations, axis=0)
					detached_activations[k] = actv_avg1
			
			if k.startswith('Linear'):
				activations = v.detach().numpy()
				
				# mean over frames
				actv_avg1 = np.mean(activations, axis=0)
				detached_activations[k] = actv_avg1
		
		self.detached_activations = detached_activations
		
		return detached_activations
	
	def store_activations(self, RESULTDIR, identifier):
		RESULTDIR = (Path(RESULTDIR))
		
		if not (Path(RESULTDIR)).exists():
			os.makedirs((Path(RESULTDIR)))
		
		filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')
		
		with open(filename, 'wb') as f:
			pickle.dump(self.detached_activations, f)