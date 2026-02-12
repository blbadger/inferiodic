import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from einops import rearrange
import transformers
import mlflow
from transformers import AutoTokenizer
from safetensors.torch import load_model
from transformers import LlamaConfig, LlamaForCausalLM
import contextlib
from datasets import load_from_disk
from mtp_transformer_fineweb import MTPTransformer
from colorama import Fore, Back, Style
from periodicity import periodicity
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('n vocab: ', n_vocab)
tokenized_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 512
n_heads = 8
layers = 16
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': layers,
    'num_attention_heads': n_heads,
    'vocab_size': 8000,
    'use_cache': False
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).to(device)
load_model(model, '/home/bbadger/Desktop/fineweb_training/fineweb_llama_512_n16_h8_c512/checkpoint-200000/model.safetensors')

#model = MTPTransformer(model, n_tokens=1).to(device)
#load_model(model, '/home/bbadger/Desktop/fineweb_training/mtp3_fineweb_llama_512_n16_c512/checkpoint-200000/model.safetensors')
#load_model(model, '/home/bbadger/Desktop/mtp16_fineweb_llama_512_n16_c128_b16x4x1/checkpoint-40000/model.safetensors')

train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
iter_data = iter(test_dataset)

with contextlib.nullcontext():
	total = 0
	for i in range(10):
		item = next(iter_data)
		tokens = torch.tensor(item['input_ids']).unsqueeze(0)
		label_tokens = tokens.clone().detach()
		for i in range(len(label_tokens[0])):
			if int(label_tokens[0, i]) == 1:
				label_tokens[0, i] = -100
		if "attention_mask" in item:
			attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)
		else:
			attention_mask = None
		output = model.forward(input_ids=tokens.to(device), labels=label_tokens.to(device), attention_mask=attention_mask)
		print ('Given loss: ', output[0])
		total += float(output[0])
	print ('Average loss: ', total/i)

periodicities = []
for i in range(5):
	tokens = next(iter_data)
	last_index = 60
	tokens = tokens['input_ids'][:last_index]
	string = tokenizer.decode(tokens)
	#attention_mask =torch.tensor(tokens["attention_mask"]).unsqueeze(0)
	tokens = torch.tensor(tokens).unsqueeze(0)
	print (Fore.CYAN + Style.BRIGHT + string)
	print (tokens.shape)

	output = model.generate(tokens.to(device), max_new_tokens=100)
	string = tokenizer.decode(output[0])
	print (Fore.MAGENTA + Style.BRIGHT + 'Output: ', string, "\n", output)

	# label_tokens = torch.where(tokens==1, -100, tokens).clone().detach()
	# print ('Loss: ', model(tokens.to(device), labels=label_tokens.to(device))[0])
	first_periodic_token, periodicity = periodicity(list(output))
	first_periodic_tokens.append(first_periodic_token)
	periodicities.append(periodicity)

plt.hist(periodicities)
plt.show()
plt.close()




