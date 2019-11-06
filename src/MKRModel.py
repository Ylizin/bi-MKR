import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from gensim import corpora, models, similarities
from gensim.corpora.dictionary import Dictionary as gen_dict
from gensim.models.ldamulticore import LdaMulticore
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

import model
import paths
from layers import CrossCompressUnit, Dense
from TextModel import TextModel


class MKR_model(nn.Module):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, use_inner_product=True):
        super(MKR_model, self).__init__()
        print(n_user)
        print(n_item)
        print(n_entity)

        # <Lower Model>
        self.args = args
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.use_inner_product = args.use_inner_product
        self.user_enhanced = args.user_enhanced
        self.id2text = model.load_id2text()

        # Init embeddings
        self.text_encoder = TextModel()
        # we need to merge user/item embedding tables, it makes no difference since the id of user and item are distincted
        self.user_embeddings_lookup = nn.Embedding(self.n_user+self.n_item, self.args.dim)
        self.item_embeddings_lookup = self.user_embeddings_lookup
        # self.item_embeddings_lookup = nn.Embedding(self.n_item, self.args.dim)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entity, self.args.dim)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relation, self.args.dim)

        self.user_mlp = nn.Sequential()
        self.tail_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        for i_cnt in range(self.args.L):
            self.user_mlp.add_module('user_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                     CrossCompressUnit(self.args.dim))
        # <Higher Model>
        self.kge_pred_mlp = Dense(self.args.dim * 2, self.args.dim)
        self.kge_mlp = nn.Sequential()
        for i_cnt in range(self.args.H - 1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt),
                                    Dense(self.args.dim * 2, self.args.dim * 2))
        if self.use_inner_product==False:
            self.rs_pred_mlp = Dense(self.args.dim * 2, 1)
            self.rs_mlp = nn.Sequential()
            for i_cnt in range(self.args.H - 1):
                self.rs_mlp.add_module('rs_mlp{}'.format(i_cnt),
                                       Dense(self.args.dim * 2, self.args.dim * 2))

    def forward(self, user_indices=None, item_indices=None, head_indices=None,
            relation_indices=None, tail_indices=None):
        '''in out model, the head and the user/item has the same id 
        
        Keyword Arguments:
            user_indices {[type]} -- [description] (default: {None})
            item_indices {[type]} -- [description] (default: {None})
            head_indices {[type]} -- [description] (default: {None})
            relation_indices {[type]} -- [description] (default: {None})
            tail_indices {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        '''
        # <Lower Model>
        # if user and item together in the rs stage, the head_[0] is used to enhance user and the head_[1] is used to enhance item
        _user_item_together = True if isinstance(head_indices,list) else False
        if user_indices is not None:
            assert (user_indices<self.n_user+self.n_item).all(), torch.max(user_indices)
            self.user_indices = user_indices
            # if not _user_item_together:
            #     self.user_indices = user_indices-self.n_item # if not user item together, the user index is start from 
            # else:
            #     self.user_indices = user_indices
            user_text = self.id2text.loc[user_indices.cpu(),'description'].tolist()
            user_text_embeddings = self.text_encoder(user_text)[:,0,:].squeeze()
            self.user_embeddings = torch.cat([self.user_embeddings_lookup(self.user_indices)],dim = 1)
            
        if item_indices is not None:
            self.item_indices = item_indices
            assert (item_indices<self.n_item).all(), torch.max(item_indices)
            self.item_embeddings = self.item_embeddings_lookup(self.item_indices)

        if head_indices is not None:
            self.head_indices = head_indices
            if _user_item_together:
                self.head_embeddings = [self.entity_embeddings_lookup(self.head_indices[0]),self.entity_embeddings_lookup(self.head_indices[1])]
            else:
                self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)

        if relation_indices is not None:
            self.relation_indices = relation_indices
            self.relation_embeddings = self.relation_embeddings_lookup(self.relation_indices)

        if tail_indices is not None:
            self.tail_indices = tail_indices
            assert (tail_indices<self.n_entity).all(), torch.max(tail_indices)

            self.tail_embeddings = self.entity_embeddings_lookup(self.tail_indices)


        # Embeddings 
        if self.user_enhanced == 1 or (self.user_enhanced==2 and not _user_item_together): # when not user-item together and enhance=2, it means inference kge, the user indicies is occupied
            self.user_embeddings, self.head_embeddings = self.cc_unit([self.user_embeddings, self.head_embeddings])
        elif self.user_enhanced == 0:
            self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings])
        elif _user_item_together: # this means user and item enhanced together and train_RS, in this case, the head_indices is a tuple of user,item indices
            self.user_embeddings, self.head_embeddings[0] = self.cc_unit([self.user_embeddings, self.head_embeddings[0]])
            self.item_embeddings, self.head_embeddings[1] = self.cc_unit([self.item_embeddings, self.head_embeddings[1]])
            


        # if item_indices is not None: # item_indices is not None
        #     self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings])
        # elif user_indices is not None: # user indices is not None but item indices is None
        #     self.user_embeddings, self.head_embeddings = self.cc_unit([self.user_embeddings, self.head_embeddings])



        # <Higher Model>
        if user_indices is not None and item_indices is not None:
            # RS
            self.user_embeddings = self.user_mlp(self.user_embeddings)
            if self.use_inner_product:
                # [batch_size]
                self.scores = torch.sum(self.user_embeddings * self.item_embeddings, 1)
            else:
                # [batch_size, dim * 2]
                self.user_item_concat = torch.cat([self.user_embeddings, self.item_embeddings], 1)
                self.user_item_concat = self.rs_mlp(self.user_item_concat)
                # [batch_size]
                self.scores = torch.squeeze(self.rs_pred_mlp(self.user_item_concat))
            self.scores_normalized = torch.sigmoid(self.scores)
            outputs = [self.user_embeddings, self.item_embeddings, self.scores, self.scores_normalized]
        if relation_indices is not None:
            # KGE
            self.tail_embeddings = self.tail_mlp(self.tail_embeddings)
            # [batch_size, dim * 2]
            self.head_relation_concat = torch.cat([self.head_embeddings, self.relation_embeddings], 1)
            self.head_relation_concat = self.kge_mlp(self.head_relation_concat)
            # [batch_size, 1]
            self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
            self.tail_pred = torch.sigmoid(self.tail_pred)
            self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            self.rmse = torch.mean(
                torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
                           self.tail_pred, 2), 1) / self.args.dim))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]

        return outputs
