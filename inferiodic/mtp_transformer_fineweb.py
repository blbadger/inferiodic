import torch
from einops import rearrange
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig, GenerationMixin
import torch.nn as nn
import mlflow
from datasets import load_from_disk
import datasets

device = 0 if torch.cuda.is_available else 'cpu'

dim = 512
context_length = 32
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': 16,
	'num_attention_heads': 4,
	'vocab_size': 8000
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()

class MTPTransformer(nn.Module, GenerationMixin):

	def __init__(self, model, n_tokens=2):
		super().__init__()
		self.model = model
		self.n_tokens = n_tokens
		self.cel = torch.nn.CrossEntropyLoss()
		config  = {
				 'hidden_size': 512,
				 'intermediate_size': 4*512,
				 'num_hidden_layers': 16,
				 'num_attention_heads': 4, # mock heads
				 'vocab_size': 8000
			 }
		self.config = LlamaConfig(**config)
		self.main_input_name = 'input_ids'
		max_input_length = 512
		generation_config_args = {'max_length': max_input_length}
		self.generation_config = GenerationConfig(**generation_config_args)
		self.max_length = 512
		self._supports_cache_class = False
		self.device = self.model.device

	def can_generate(self):
		return True

	def _is_stateful(self):
		return False

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		if labels is None:
			labels = torch.where(input_ids==1, -100, input_ids)
		all_outputs = []
		for i in range(self.n_tokens):
			output = self.model.lm_head(self.model.model(x)[0])
			output = rearrange(output, 'b t e -> b e t')
			shift_logits = output[..., :-(1 + i)].contiguous()
			shift_labels = labels[..., (1 + i):].contiguous()
			if 'loss' in vars():
				loss += self.cel(shift_logits, shift_labels)
			else:
				loss = self.cel(shift_logits, shift_labels)
			x = torch.argmax(output, dim=-2)
			all_outputs.append(rearrange(output, 'b e t -> b t e'))# rearrange for gen

		return CausalLMOutput(loss=loss, logits=all_outputs[0])

if __name__ == '__main__':
	model = MTPTransformer(model, n_tokens=4)
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)


	train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
	test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

	#map_dataset(train_path, test_path)
	datasets.config.IN_MEMORY_MAX_SIZE = 35e9
	train_dataset = load_from_disk(train_path)
	test_dataset = load_from_disk(test_path)


	mlflow.end_run()
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=3,
		per_device_train_batch_size=8,
		per_device_eval_batch_size=8,
		gradient_accumulation_steps=2,
		warmup_steps=500,
		eval_steps=4000,
		save_steps=8000,
		learning_rate=2e-4, 
		fp16=True, 
		#evaluation_strategy='steps',
		output_dir='~/Desktop/mtp4_fineweb_llama_512_n16_c512_b8x4x2',
		optim='adamw_torch',
		overwrite_output_dir=True,
		max_steps=200000,
		torch_compile=True
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	)

	model.train()
	trainer.train()
