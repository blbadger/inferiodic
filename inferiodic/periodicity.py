import torch

def periodicity(tokens):
 	periodicity = 0
 	first_periodic_token = 0
 	for i, token in enumerate(tokens):
 		if token in tokens[i+1:]:
 			next_occurrences = [j for j, x in enumerate(tokens) if x == token and j>i]
 			for next_occurrence in next_occurrences:
	 			start_index, end_index = i, next_occurrence
	 			token_sequence = tokens[i:next_occurrence]
	 			while end_index <= len(tokens):
	 				gap = end_index - start_index
		 			if end_index >= len(tokens):
		 				if tokens[start_index:] == token_sequence[:len(tokens[start_index:])]:
				 			first_periodic_token = i
				 			periodicity = gap
				 			return first_periodic_token, periodicity

			 		if tokens[start_index:end_index] != token_sequence:
	 					break

			 		end_index += gap
			 		start_index += gap

 	return first_periodic_token, periodicity

if __name__ == '__main__':
	tokens = [0, 3, 8, 0, 2, 5, 0, 8, 7, 0, 8, 7, 0, 8, 7, 0, 8, 7, 1]
	print (periodicity(tokens))