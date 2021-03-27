import torch
import pickle

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()


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
		detached_packaged_data = {}
		
		# testing:
		# activations = self.activations[
		#     'Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))--0'].detach().numpy()
		
		for k, v in self.activations.items():
			print(f'Detaching activation for layer: {k}')
			if k.startswith('Conv2d'):  # no packaged data
				activations = v.detach().numpy()
				# squeeze batch dimension
				avg_activations = activations.squeeze()
				# expand (flatten) the channel x kernel dimension:
				avg_activations = avg_activations.reshape(
					[avg_activations.shape[0] * avg_activations.shape[1], avg_activations.shape[2]])
				# mean over time
				avg_activations = avg_activations.mean(axis=1)
				
				detached_activations[k] = avg_activations
			
			if k.startswith('LSTM'):  # packaged data available
				packaged_data = v[0].data.detach().numpy()
				detached_packaged_data[k] = packaged_data
				activations = v[1]
				if lstm_output == 'sequence':
					activations = activations[0].detach().numpy()
				elif lstm_output == 'recent':
					activations = activations[1].detach().numpy()
				else:
					print('LSTM output type not available')
				
				# squeeze batch dimension
				avg_activations = activations.squeeze()
				# average over the num directions dimension:
				avg_activations = avg_activations.mean(axis=0)
				
				detached_activations[k] = avg_activations
			
			if k.startswith('Linear'):  # no packaged data
				activations = v.detach().numpy()
				# mean over time dimension
				avg_activations = activations.mean(axis=0)
				detached_activations[k] = avg_activations
		
		self.detached_activations = detached_activations
		
		return detached_activations
	
	def store_activations(self, RESULTDIR, identifier):
		RESULTDIR = (Path(RESULTDIR))
		
		if not (Path(RESULTDIR)).exists():
			os.makedirs((Path(RESULTDIR)))
		
		filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')
		
		with open(filename, 'wb') as f:
			pickle.dump(self.detached_activations, f)

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


# Download an example audio file
import urllib
url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

out_features = model.forward(filename)

act_keys = list(save_output.activations.keys())
act_vals = save_output.activations

