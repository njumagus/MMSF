from transformers import RobertaTokenizer, RobertaModel
from ..baseExtractor import baseTextExtractor
import torch
import numpy as np


class robertaExtractor(baseTextExtractor):
    """
    Text feature extractor using RoBERTa
    Ref: https://huggingface.co/docs/transformers/model_doc/roberta
    Pretrained models: https://huggingface.co/models
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing RoBERTa text feature extractor.")
            super().__init__(config, logger)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.config['pretrained'])
            self.model = RobertaModel.from_pretrained(self.config['pretrained']).to(self.config['device'])
            self.finetune = self.config['finetune'] if 'finetune' in self.config else False
        except Exception as e:
            logger.error("Failed to initialize robertaExtractor.")
            raise e
    
    def extract(self, text):
        try:
            input_ids = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(self.config['device'])
            with torch.no_grad():
                last_hidden_state = self.model(input_ids).last_hidden_state
            return last_hidden_state.squeeze().cpu().numpy()
        except Exception as e:
            self.logger.error(f"Failed to extract text features with RoBERTa for '{text}'.")
            raise e

    def tokenize(self, text):
        """
        For compatibility with feature files generated by MMSA DataPre.py
        Returns:
            input_ids: input_ids,
            input_mask: attention_mask,
            segment_ids: token_type_ids
        """
        try:
            text_bert = self.tokenizer(text, add_special_tokens=True, return_tensors='np')
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(text_bert['input_ids'].squeeze(0))
            text_bert = np.concatenate([text_bert['input_ids'].transpose(), 
                                        text_bert['attention_mask'].transpose(), 
                                        np.expand_dims(token_type_ids, axis=1)],
                                        axis=1)
            return text_bert
        except Exception as e:
            self.logger.error(f"Failed to tokenize text with RoBERTa for '{text}'.")
            raise e