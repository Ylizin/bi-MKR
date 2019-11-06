import torch
from torch import nn
import os
from collections.abc import Iterable
from pytorch_transformers import *

_CUDA = torch.cuda.is_available()


class BertWrapper(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.model = BertModel.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_id = self.tokenizer.encode(self.pad_token)
        self.model_dir = model_dir

    def __tokenize_sentences(self, sentence1, sentence2=None):
        if sentence2:
            sentence = "{}\t{}\t{}\t{}\t{}".format(
                self.cls_token, sentence1, self.sep_token, sentence2, self.sep_token
            )
        else:
            sentence = "{}\t{}\t{}".format(self.cls_token, sentence1, self.sep_token)

        return self.tokenizer.encode(sentence)

    def __get_sentences_ids(self, list_of_sentences):
        """
        if return sentence embedding at one invocation, the list should be a list of 
        (sen1,sen2)
        """
        if isinstance(list_of_sentences[0], tuple):
            assert (
                len(list_of_sentences[0]) == 2
            ), "If two_sent is True, the element should be a tuple of len:2."
            return [
                self.__tokenize_sentences(tup[0], tup[1]) for tup in list_of_sentences
            ]
        else:
            return [self.__tokenize_sentences(li) for li in list_of_sentences]

    def __pad_ids(self, ids):
        '''pad a batch into same length
        '''             
        self.max_len = 0  # max sequence length in batch
        for li in ids:
            self.max_len = self.max_len if self.max_len >= len(li) else len(li)
        self.max_len = 512 if self.max_len > 512 else self.max_len
        return torch.tensor(
            [i[: self.max_len] + self.pad_id * (self.max_len - len(i)) for i in ids]
        )

    def save_to_dir(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model)

        torch.save(self.model.state_dict(), os.path.join(self.model_dir, WEIGHTS_NAME))
        self.model.config.to_json_file(os.path.join(self.model_dir, CONFIG_NAME))
        self.tokenizer.save_vocabulary(self.model_dir)

    def get_vec(self, list_of_str, return_pooled=False):

        assert isinstance(
            list_of_str, Iterable
        ), "get vec param must be list, wrap it please."
        if not list_of_str:
            return None

        padded_ids = self.__pad_ids(self.__get_sentences_ids(list_of_str))
        if _CUDA:
            padded_ids = padded_ids.cuda()
        embeddings, pooled_embedding = self.model(padded_ids)
        if return_pooled:
            return embeddings, pooled_embedding
        else:
            return embeddings
