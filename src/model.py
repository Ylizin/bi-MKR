import sys
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from layers import Dense, CrossCompressUnit
import math
import paths
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary as gen_dict
from gensim import corpora, models, similarities
from scipy.spatial.distance import cosine
import pickle
# from trace_grad import plot_grad_flow
from MKRModel import *

def load_lda_model(model_path = paths.lda_model):
    return LdaMulticore.load(model_path,mmap='r')

def load_dict(model_path = paths.dict_path):
    return gen_dict.load(model_path,mmap='r')

def load_id2text(model_path = paths.id2text):
    return pickle.load(open(model_path,'rb'))

def load_index(lda_model,corpus,index_path = paths.index_path):
    '''return cached index if already loaded 
    else, the index is calcu.ed from the result of doc2bow method->corpus
    
    Args:
        lda_model ([type]): [description]
        corpus ([type]): [description]
        index_path ([type], optional): [description]. Defaults to paths.index_path.
    '''
    if hasattr(load_index,'index'):
        return load_index.index 
    # if (os.path.exists(index_path)):
    #     index = similarities.MatrixSimilarity.load(index_path)
    else:
        load_index.index = similarities.MatrixSimilarity(lda_model[corpus])  # transform corpus to LSI space and index it
        load_index.index.save(index_path)
    return load_index.index


# def cal_lda_sim(user_texts,item_texts,lda_model,dic):
#     user_bow = [dic.doc2bow(t) for t in user_texts]
#     # user_bow = dic.doc2bow(user_texts)
#     item_bow = [dic.doc2bow(t) for t in item_texts]
#     def get_vec(model,c):
#         vec = model.get_document_topics(c,minimum_probability= 0)
#         _,_vec = zip(*vec)
#         vec = np.array(_vec)
#         return vec
#     user_vecs = [get_vec(lda_model,b) for b in user_bow]
#     item_vecs = [get_vec(lda_model,b) for b in item_bow]
    
#     lda_scores = np.array([cosine(u,i) for u,i in zip(user_vecs,item_vecs)])
#     return lda_scores
        




class MKR:
    def __init__(self, args, n_user, n_items, n_entities,
                 n_relations):
        
        self.args = args
        self.user_enhanced = args.user_enhanced
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lda = load_lda_model()
        self.lda_dict = load_dict()
        self.id2text = load_id2text()
        self._parse_args(n_user, n_items, n_entities, n_relations)
        self._build_model()
        self._build_loss()
        self._build_ops()


    def _parse_args(self, n_user, n_items, n_entities, n_relations):
        self.n_user = n_user 
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

    def _build_model(self):
        print("Build models")
        self.MKR_model = MKR_model(self.args, self.n_user, self.n_item, self.n_entity, self.n_relation)
        if torch.cuda.device_count()>=1:
            self.MKR_model = torch.nn.DataParallel(self.MKR_model)
        for m in self.MKR_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
        # for param in self.MKR_model.parameters():
        #     param.requires_grad = True

    def _build_loss(self):
        self.sigmoid_BCE = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def _build_ops(self):
        self.optimizer_rs = torch.optim.Adam(self.MKR_model.parameters(),
                                             lr=self.args.lr_rs)
        self.optimizer_kge = torch.optim.Adam(self.MKR_model.parameters(),
                                              lr=self.args.lr_kge)

    def _inference_rs(self, inputs,inference = False):
        # Inputs
        self.user_indices = inputs[:, 0].long().to(self.device,
                non_blocking=True)
        self.item_indices = inputs[:, 1].long().to(self.device,
                non_blocking=True)
        labels = inputs[:, 2].float().to(self.device)
        # the inputs of inference rs is onlt the (user,item) tuple, not the really indices in the kg_dataset
        self.head_indices = inputs[:, 1].long().to(self.device,
                non_blocking=True) # this means that the head indices is from the same data as items

        # Inference
        if self.user_enhanced == 0 or self.user_enhanced ==1:
            pass
        elif self.user_enhanced == 2:# for ui enhanced together
            self.head_indices = [self.user_indices,self.item_indices]
        if inference:
            scores = []
            self.MKR_model.eval()
            torch.cuda.empty_cache()
            torch.autograd.set_grad_enabled(False)
            for user_index,item_index,head_index in zip(self.user_indices,self.item_indices,self.head_indices):
                *_,score = self.MKR_model(user_indices=self.user_indices,
                                     item_indices=self.item_indices,
                                     head_indices=self.head_indices,
                                     relation_indices=None,
                                     tail_indices=None)
                scores.append(score)
            torch.autograd.set_grad_enabled(True)
            return torch.tensor(scores),labels
            
        outputs = self.MKR_model(user_indices=self.user_indices,
                                     item_indices=self.item_indices,
                                     head_indices=self.head_indices,
                                     relation_indices=None,
                                     tail_indices=None)
        user_embeddings, item_embeddings, scores, scores_normalized = outputs
        return user_embeddings, item_embeddings, scores, scores_normalized, labels

    def _inference_kge(self, inputs):
        # Inputs
        if not self.user_enhanced:
            self.item_indices = inputs[:, 0].long().to(self.device,
                    non_blocking=True)
        else: # head and user/item are same indices
            self.user_indices = inputs[:,0].long().to(self.device,
                    non_blocking=True)

        self.head_indices = inputs[:, 0].long().to(self.device,
                non_blocking=True)
        self.relation_indices = inputs[:, 1].long().to(self.device,
                non_blocking=True)
        self.tail_indices = inputs[:, 2].long().to(self.device,
                non_blocking=True)


        # Inference
        if not self.user_enhanced:
            outputs = self.MKR_model(user_indices=None,
                                     item_indices=self.item_indices,
                                     head_indices=self.head_indices,
                                     relation_indices=self.relation_indices,
                                     tail_indices=self.tail_indices)
        else:
            # here we take the user indices as both user-item, since it makes no difference 
            outputs = self.MKR_model(user_indices=self.user_indices,
                                     item_indices=None,
                                     head_indices=self.head_indices,
                                     relation_indices=self.relation_indices,
                                     tail_indices=self.tail_indices)
                        
        head_embeddings, tail_embeddings, scores_kge, rmse = outputs
        return head_embeddings, tail_embeddings, scores_kge, rmse

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def loss_rs(self, user_embeddings, item_embeddings, scores, labels):
        # scores_for_signll = torch.cat([1-self.sigmoid(scores).unsqueeze(1),
        #                                self.sigmoid(scores).unsqueeze(1)], 1)
        # base_loss_rs = torch.mean(self.nll_loss(scores_for_signll, labels))
        base_loss_rs = torch.mean(self.sigmoid_BCE(scores, labels))
        l2_loss_rs = self.l2_loss(user_embeddings) + self.l2_loss(item_embeddings)
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('rs' in name) or ('cc_unit' in name) or ('user' in name)) \
                    and ('weight' in name):
                l2_loss_rs = l2_loss_rs + self.l2_loss(param)
        loss_rs = base_loss_rs + l2_loss_rs * self.args.l2_weight
        return loss_rs, base_loss_rs, l2_loss_rs

    def loss_kge(self, scores_kge, head_embeddings, tail_embeddings):
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        for name, param in self.MKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('kge' in name) or ('tail' in name) or ('cc_unit' in name)) \
                    and ('weight' in name):
                l2_loss_kge = l2_loss_kge + self.l2_loss(param)
        # Note: L2 regularization will be done by weight_decay of pytorch optimizer
        loss_kge = base_loss_kge + l2_loss_kge * self.args.l2_weight
        return loss_kge, base_loss_kge, l2_loss_kge

    def train_rs(self, inputs, show_grad=False, glob_step=None):
        self.MKR_model.train()
        user_embeddings, item_embeddings, scores, _, labels= self._inference_rs(inputs)
        loss_rs, base_loss_rs, l2_loss_rs = self.loss_rs(user_embeddings, item_embeddings, scores, labels)

        self.optimizer_rs.zero_grad()
        loss_rs.backward()
        # if show_grad:
        #     plot_grad_flow(self.MKR_model.named_parameters(),
        #                    "grad_plot/rs_grad_step{}".format(glob_step))

        self.optimizer_rs.step()
        loss_rs.detach()
        user_embeddings.detach()
        item_embeddings.detach()
        scores.detach()
        labels.detach()

        return loss_rs, base_loss_rs, l2_loss_rs

    def train_kge(self, inputs, show_grad=False, glob_step=None):
        self.MKR_model.train()
        head_embeddings, tail_embeddings, scores_kge, rmse = self._inference_kge(inputs)
        loss_kge, base_loss_kge, l2_loss_kge = self.loss_kge(scores_kge, head_embeddings, tail_embeddings)

        self.optimizer_kge.zero_grad()
        loss_kge.sum().backward()
        # if show_grad:
        #     plot_grad_flow(self.MKR_model.named_parameters(),
        #                    "grad_plot/kge_grad_step{}".format(glob_step))

        self.optimizer_kge.step()
        loss_kge.detach()
        head_embeddings.detach()
        tail_embeddings.detach()
        scores_kge.detach()
        rmse.detach()
        return rmse, loss_kge.sum(), base_loss_kge.sum(), l2_loss_kge

    def eval(self, inputs):
        print(inputs.shape)
        self.MKR_model.eval()
        inputs = torch.from_numpy(inputs)
        *_, scores, labels = self._inference_rs(inputs,True)
        labels = labels.to("cpu").detach().numpy()
        scores = scores.to("cpu").detach().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc

    def _attatch_train_sparse(self,train_sparse):
        if self.user_enhanced == 1 or self.user_enhanced == 0: # user enhanced
            # 获取所有的user的description
            train_texts_bow = self.id2text.loc[range(self.n_item,self.n_item+self.n_user),'description'].apply(self.lda_dict.doc2bow)
            self.index = load_index(self.lda,train_texts_bow)
        elif self.user_enhanced == -10: # item enhanced:
            # 获取所有item的des
            train_texts_bow = self.id2text.loc[range(self.n_item),'description'].apply(self.lda_dict.doc2bow)
            self.index = load_index(self.lda,train_texts_bow)

    def topk_eval(self, user_list, train_record, test_record, item_set, k_list,train_sparse = None):
        # test record 仅用于计算metrics的时候
        print("Eval TopK")
        self._attatch_train_sparse(train_sparse)

        if train_sparse is not None:
            self.train_sparse = train_sparse

        precision_list = {k: [] for k in k_list}
        recall_list = {k: [] for k in k_list}
        ndcg_list = {k: [] for k in k_list}
        map_list = { k: [] for k in k_list}
        for user in tqdm(user_list):
            # 整体集去除出现在train record中的item后作为待预测，得到的预测值与test record中的真实值做评估
            # train record记录所有出现的lib，test record仅记录label为1 的lib
            test_item_list = list(item_set - train_record[user])
            item_score_map = dict()
            # 我们首先获得除train record里面的有的item， 之外的其余item的预测值
            scores = self._get_scores(np.array([user]*len(test_item_list)),
                                      np.array(test_item_list),
                                      np.array(test_item_list))
            items = np.array(test_item_list)
            #对所有的item进行按分数的排序，取topk
            for item, score in zip(items, scores):
                item_score_map[item] = score
            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
            #item sorted is the item-score pair, item means the items' ids
            item_sorted = [i[0] for i in item_score_pair_sorted]
            
            for k in k_list: # topk_list
                hit_num = len(set(item_sorted[:k]) & test_record[user])
                precision_list[k].append(hit_num / k)
                recall_list[k].append(hit_num / len(test_record[user]))
                _ndcg,_map = self.get_NDCG_MAP(item_sorted[:k],test_record[user])
                ndcg_list[k].append(_ndcg)
                map_list[k].append(_map)
                
        precision = [np.mean(precision_list[k]) for k in k_list]
        recall = [np.mean(recall_list[k]) for k in k_list]
        f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]
        NDCG = [np.mean(ndcg_list[k]) for k in k_list]
        MAP = [np.mean(map_list[k]) for k in k_list]
        table = {'precision':precision,'recall':recall,'f1':f1,'NDCG':NDCG,'MAP':MAP}
        df = pd.DataFrame(table)
        print(df.T)
        return precision, recall, f1,NDCG,MAP

    def _get_scores(self, user, item_list, head_list):
        # Inputs
        # print(user)
        # print(item_list)
        user = torch.from_numpy(user)
        item_list = torch.from_numpy(item_list)
        head_list = torch.from_numpy(head_list)
        self.user_indices = user.long().to(self.device)
        self.item_indices = item_list.long().to(self.device)
        self.head_indices = head_list.long().to(self.device)
 
        scores = []
        self.MKR_model.eval()
        torch.cuda.empty_cache()
        torch.autograd.set_grad_enabled(False)
        for user_index,item_index,head_index in zip(self.user_indices,self.item_indices,self.head_indices):
            *_,score = self.MKR_model(user_indices=self.user_indices,
                                    item_indices=self.item_indices,
                                    head_indices=self.head_indices,
                                    relation_indices=None,
                                    tail_indices=None)
            scores.append(score)
        torch.autograd.set_grad_enabled(True)
        scores = torch.tensor(scores)
        # item_text = self.id2text.loc[item_list]['description'].apply(self.lda_dict.doc2bow)
        # doc2bow process input descriptions 
        if self.user_enhanced == 1 or self.user_enhanced == 0:
            user_bow = self.id2text.loc[user.cpu(),'description'].apply(self.lda_dict.doc2bow)
            user_vec = self.lda[user_bow]
            sim_vec = torch.tensor(self.index[user_vec])
            
            # result->(batch,n_item)    shape of (n_item,n_user)
            lda_scores = self.train_sparse.t()\
            .mm(
                        # shape of (n_user,batch)
                        sim_vec.t() 
            ).t()
            # gather函数，参见文档，通过从item_indicies这个tensor中取得indices，来对lda scores进行索引，来构造一个和item_indices对应的分数
            # 其中，lda_scores 是(batch,n_item), self.item_indices 是(batch,1)
            lda_scores = torch.gather(lda_scores,1,self.item_indices.view(-1,1)).view(-1)

        elif self.user_enhanced == -10:
            item_bow = self.id2text.loc[item_list.cpu(),'description'].apply(self.lda_dict.doc2bow)
            item_vec = self.lda[item_bow]
            sim_vec = torch.tensor(self.index[item_vec])
            # result->(batch,n_user)    trainsparse->shape of (n_item,n_user)
            lda_scores = self.train_sparse\
            .mm(
                        # shape of (n_item,batch)
                        sim_vec.t() 
            ).t()
            lda_scores = torch.gather(lda_scores,1,(self.user_indices).view(-1,1)).view(-1)

        scores = 1.0*lda_scores + 0.0*scores    
        return scores

    def get_NDCG_MAP(self,item_sorted,record):
        # assert len(item_sorted)<=len(record),"The length of record is smaller than topK."
        # get hit predicts
        hit_index = [] #record the index those hit the gound truth, for the map calculation
        predict_rels = []# record the sequence of rel for the calculation of NDCG
        for i,item in enumerate(item_sorted):  
            if item in record: # if in record then rel is 1, means that item has an interact with the user
                predict_rels.append(1)
                hit_index.append(i)
            else:
                predict_rels.append(0)

        if not hit_index: # if nothing hit
            return 0,0
        
        #if hit, get the NDCG and MAP
        record_array = np.ones(len(record))
        if len(item_sorted)>len(record):# if the predicted len is larger than record
            _pad_array = np.zeros(len(item_sorted)-len(record))
            _padded_record = np.concatenate([record_array,_pad_array])
            record_array = _padded_record
        # if len record is smaller than topK, we have not taken any operation?? 
        _index_of_predicts = np.array([i for i in range(1,len(item_sorted)+1)]) #generate the index of the predict sequence,start from 1
        #NDCG
        IDCG = np.sum(record_array[:len(_index_of_predicts)]/np.log((_index_of_predicts+1))) # remember that np.log is log_e
        DCG = np.sum((np.exp2(np.array(predict_rels))-1)/np.log(_index_of_predicts+1)) # dcg = (2**rel -1)/log(i+1) //here i satrt from 1 our i also from 1
        #MAP    
        #hit index from 0
        _map_hit_index = np.array(hit_index)+1 # idx start from 0 but in map calculation the denominator start from 1 so add 1
        MAP = np.sum(_index_of_predicts[:len(_map_hit_index)]/_map_hit_index)/len(record)
        return DCG/IDCG,MAP

        
