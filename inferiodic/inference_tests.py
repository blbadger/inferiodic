from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print ('here')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B').to(device)
tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B')
print ('model loaded')

input = "Four score and seven years ago, our forefathers, for the purpose of forming a more perfect union,"
tokenized_input = tokenizer(input)

def token_generate(tokenized_input, tokens_to_generate=200):
	tokens, attention_mask = torch.tensor(tokenized_input.input_ids).unsqueeze(0).to(device), torch.tensor(tokenized_input.attention_mask)
	for i in range(tokens_to_generate):
		output = model(tokens, labels=tokens)
		print (output.loss)
		next_token = torch.argmax(output.logits[:, -1], dim=-1)
		tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)
	return tokens

def embedding_generate(tokenized_input, tokens_to_generate=50):
	tokens, attention_mask = torch.tensor(tokenized_input.input_ids).unsqueeze(0).to(device), torch.tensor(tokenized_input.attention_mask)
	embeddings = model.model.embed_tokens(tokens)
	for i in range(tokens_to_generate):
		output = model.model(inputs_embeds=embeddings, attention_mask=attention_mask)
		embedding = output.last_hidden_state[:, -1, :].unsqueeze(0)
		embeddings = torch.cat((embeddings, embedding), dim=1)
	return embeddings

def token_generate_embedding_noise(tokenized_input, tokens_to_generate=200, scale=0.02):
	tokens = torch.tensor(tokenized_input.input_ids).unsqueeze(0).to(device)
	for i in range(tokens_to_generate):
		embeddings = model.model.embed_tokens(tokens)
		embeddings += (torch.randn(embeddings.shape) * scale).to(device)
		output = model.model(inputs_embeds=embeddings.to(device))
		output = model.lm_head(output.last_hidden_state[:, -1, :]).unsqueeze(0)
		next_token = torch.argmax(output, dim=-1)
		tokens = torch.cat((tokens, next_token), dim=1)
	return tokens


tokens = token_generate(tokenized_input)
print (tokenizer.decode(tokens[0].tolist()))

# embeddings = embedding_generate(tokenized_input)
# tokens = torch.argmax(model.lm_head(embeddings), dim=-1).squeeze(0)
# print (tokens.shape)
# print (tokenizer.decode(tokens.tolist()))

# tokens = token_generate_embedding_noise(tokenized_input)
# print (tokenizer.decode(tokens[0].tolist()))