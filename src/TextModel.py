import bert_model
from torch import nn
import paths        

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = bert_model.BertWrapper(paths.bert_model)

    def forward(self,list_of_str :list):
        '''accept a list of str, suppose they are a batch of text to be encoded
        
        Args:
            list_of_str (list): list of str
        '''
        return self.encoder.get_vec(list_of_str)