# VGGish
A `torch`-compatible port of [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)<sup>[1]</sup>, 
a feature embedding frontend for audio classification models. The weights are ported directly from the tensorflow model, so embeddings created using `torchvggish` will be identical.


## Usage

```python
import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# Download an example audio file
import urllib
url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

model.forward(filename)
```

## Model activation extraction
To extract activations from multiple sound files, use run_model.py. 

Edits to the original model: ReLU activations were changed to not be performed inplace.

To run random network: In the vggish.py script, it is possible to generate randomly permuted tensors for the VGGish architecture (line 137, class VGGish). The permuted architecture can be loaded in lines 147-156 (class VGGish, remove the commented parts).
Also remember to change the filename such that we don't overwrite: https://github.com/gretatuckute/torchvggish/blob/2f49b6e16c3b5dec3fe28513f7cd24a59046f302/extractor_utils.py#L137 

*Note*: When I downloaded the pretrained weights and scripts under /torchvggish/ these were automatically placed under /Users/{USER}/.cache/torch/hub/harritaylor_torchvggish_master/. I moved these back to /torchvvgish/ after my edits.


<hr>
[1]  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
    

