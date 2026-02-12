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

device = 'cuda' if torch.cuda.is_available else 'cpu'

@torch.no_grad()
def hamming(model_output, labels):
	total_metric = 0
	# no shift for autoencoders
	labels= torch.tensor(labels)
	model_output = torch.tensor(model_output[0])
	nonpad_tokens = torch.where(labels != -100, 1, 0)
	equal_tokens = torch.where(model_output == labels, 1, 0) & nonpad_tokens
	average_metric = torch.sum(equal_tokens) / torch.sum(nonpad_tokens)
	return torch.tensor([average_metric])

def compute_hamming_metric(eval_preds):
	preds, labels = eval_preds
	hamming_metric = hamming(preds, labels)
	return {'Hamming Distance': hamming_metric}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer has a memory leak: a workaround to avoid saving all tensors
    """
    pred_ids = torch.argmax(logits, dim=-2)
    return pred_ids, labels


def tokenize_and_preprocess(example):
	text = example['text']
	global context_length
	tokens = tokenizer(text, max_length=context_length, padding='max_length', truncation=True) # return list, not tensor
	example['input_ids'] = tokens['input_ids']
	example['attention_mask'] = tokens['attention_mask']
	return example


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

			# loss accumulation
			if 'loss' in vars():
				loss += self.cel(shift_logits, shift_labels)
			else:
				loss = self.cel(shift_logits, shift_labels)
			x = torch.argmax(output, dim=-2)
			all_outputs.append(rearrange(output, 'b e t -> b t e')) # rearrange for gen

		return CausalLMOutput(loss=loss, logits=all_outputs[0])

class OptMTPTransformer(MTPTransformer, GenerationMixin):

	def __init__(self, model, n_tokens=2, gen_reshape=False):
		# for causal generation, set n_tokens=1
		super().__init__(model, n_tokens=n_tokens)
		self.gen_reshape = gen_reshape
		
	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		if labels is None:
			# label initialization
			labels = torch.where(input_ids==1, -100, input_ids)
		all_outputs = []
		for i in range(self.n_tokens):
			output = self.model.lm_head(self.model.model(x)[0])
			output = rearrange(output, 'b t e -> b e t')
			shift_logits = output[..., :-(1 + i)].contiguous()
			shift_labels = labels[..., (1 + i):].contiguous()
			loss = self.cel(shift_logits, shift_labels)

			# gradient accumulation
			if i < self.n_tokens - 1:
				trainer.optimizer.zero_grad()
				loss.backward()
				trainer.accelerator.scaler.scale(loss)
				trainer.optimizer.step()
				trainer.optimizer.zero_grad()
			x = torch.argmax(output, dim=-2)
			if self.gen_reshape:
				output = rearrange(output, 'b e t -> b t e') # rearrange for gen

		return CausalLMOutput(loss, output)

class AccumMTPTransformer(MTPTransformer, GenerationMixin):

	def __init__(self, model, n_tokens=2, gen_reshape=False):
		# for causal generation, set n_tokens=1
		super().__init__(model, n_tokens=n_tokens)
		self.gen_reshape = gen_reshape		

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
		x = input_ids
		if labels is None:
			# label initialization
			labels = torch.where(input_ids==1, -100, input_ids)
		all_outputs = []
		for i in range(self.n_tokens):
			output = self.model.lm_head(self.model.model(x, attention_mask=attention_mask)[0])
			output = rearrange(output, 'b t e -> b e t')
			shift_logits = output[..., :-(1 + i)].contiguous()
			shift_labels = labels[..., (1 + i):].contiguous()
			loss = self.cel(shift_logits, shift_labels)

			# gradient accumulation
			if i < self.n_tokens - 1:
				loss.backward()
			x = torch.argmax(output, dim=-2)
			if self.gen_reshape:
				output = rearrange(output, 'b e t -> b t e') # rearrange for torch .generate
		
		return CausalLMOutput(loss, output)


if __name__ == '__main__':
	dim = 512
	context_length = 128
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
	model = AccumMTPTransformer(model, n_tokens=16)
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)

	train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c128-packed-debatched"
	test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c128-packed-debatched"

	#map_dataset(train_path, test_path)
	datasets.config.IN_MEMORY_MAX_SIZE = 35e9
	train_dataset = load_from_disk(train_path)
	test_dataset = load_from_disk(test_path)


	mlflow.end_run()
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=3,
		per_device_train_batch_size=32,
		per_device_eval_batch_size=32,
		warmup_steps=500,
		eval_steps=200000,
		save_steps=4000,
		learning_rate=2e-4, 
		fp16=True, 
		eval_strategy='steps',
		output_dir='~/Desktop/mtp16_fineweb_llama_512_n16_c128_b16x4x1',
		optim='adamw_torch',
		overwrite_output_dir=True,
		max_steps=100000,
		torch_compile=True
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
		compute_metrics = compute_hamming_metric,
		preprocess_logits_for_metrics=preprocess_logits_for_metrics
	)

	model.train()
	trainer.train()
