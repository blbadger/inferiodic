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

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('n vocab: ', n_vocab)
tokenized_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 512
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 16,
    'num_attention_heads': 4,
    'vocab_size': 8000,
    'use_cache': False
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()
model = MTPTransformer(model).to(device)
load_model(model, '/home/bbadger/Desktop/fineweb_training/mtp3_fineweb_llama_512_n16_c512/checkpoint-200000/model.safetensors')

train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

load_from_checkpoint=False
if load_from_checkpoint:
	mlflow.end_run()
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=2,
		per_device_train_batch_size=1,
		per_device_eval_batch_size=1,
		warmup_steps=500,
		eval_steps=4000,
		save_steps=4000,
		learning_rate=2e-4, 
		fp16=True, 
		evaluation_strategy='steps',
		output_dir='~/Desktop/fineweb_transfixer_512_n8_c512',
		optim='adamw_torch',
		overwrite_output_dir=True,
		max_steps=96010
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	)
	model.train()
	trainer.train('/home/bbadger/Desktop/fineweb_transfixer_512_n8_c512/checkpoint-96000')
	model.eval()

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
	print ('Average loss: ', total / 20)

for i in range(20):
	tokens = next(iter_data)
	last_index = 200
	tokens = tokenizer.decode(tokens['input_ids'][:last_index]
	
	string = tokenizer.decode(tokens['input_ids'][:last_index]
	#attention_mask =torch.tensor(tokens["attention_mask"]).unsqueeze(0)
	tokens = torch.tensor(tokens['input_ids'][:last_index]).unsqueeze(0)
	print (string)
	print (tokens.shape)

	output = model.generate(tokens.to(device), max_new_tokens=200)
	string = tokenizer.decode(output[0])
	print ('Output: ', string, "\n", output)

	label_tokens = torch.where(tokens==1, -100, tokens).clone().detach()
	print ('Loss: ', model(tokens.to(device), labels=label_tokens.to(device))[0])
	#print ('Outputs: ', model(tokens.to(device), labels=label_tokens.to(device))[1][0].shape)



