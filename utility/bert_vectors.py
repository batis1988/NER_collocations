from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_PATH = "./model/pretrained/bert_finetuned"

TOKEN_ARGS = {"padding": True, 
              "truncation": True, 
              "max_length": 24, 
              "return_tensors": 'pt'}


class BERTVectorizer:

    def __init__(self, model_path: str = MODEL_PATH, token_args: dict = TOKEN_ARGS):
        self.model_path = model_path
        self.token_args = token_args
        
    def __repr__(self) -> str:
        return f"BERT model source: {self.model_path}"

    def _load_model(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        return model, tokenizer

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_bert_vectors(self, sentence: str) -> np.ndarray:
        model, tokenizer = self._load_model(self.model_path)
        encoded_input = tokenizer(sentence, **self.token_args)
        with torch.no_grad():
            model_output = model(**encoded_input)
        return self._mean_pooling(model_output, encoded_input['attention_mask']).flatten().numpy()
    
    def get_embeddings(self, sentence: str) -> np.ndarray:
        model, tokenizer = self._load_model(self.model_path)
        encoded_input = tokenizer(sentence, **self.token_args)
        with torch.no_grad():
            model_output = model(**encoded_input)
        return torch.mean(model_output.last_hidden_state, dim=1)

    
    
    
    