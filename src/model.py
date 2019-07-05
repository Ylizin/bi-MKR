import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from layers import Dense, CrossCompressUnit
import math
# from trace_grad import plot_grad_flow

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


        # Init embeddings
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
            self.user_embeddings = self.user_embeddings_lookup(self.user_indices)
            
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


class MKR(object):
    def __init__(self, args, n_users, n_items, n_entities,
                 n_relations):
        self.args = args
        self.user_enhanced = args.user_enhanced
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._parse_args(n_users, n_items, n_entities, n_relations)
        self._build_model()
        self._build_loss()
        self._build_ops()

    def _parse_args(self, n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

    def _build_model(self):
        print("Build models")
        self.MKR_model = MKR_model(self.args, self.n_user, self.n_item, self.n_entity, self.n_relation)
        self.MKR_model = self.MKR_model.to(self.device, non_blocking=True)
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

    def _inference_rs(self, inputs):
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
        user_embeddings, item_embeddings, _, scores, labels = self._inference_rs(inputs)
        labels = labels.to("cpu").detach().numpy()
        scores = scores.to("cpu").detach().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc

    def topk_eval(self, user_list, train_record, test_record, item_set, k_list):
        print("Eval TopK")
        precision_list = {k: [] for k in k_list}
        recall_list = {k: [] for k in k_list}
        ndcg_list = {k: [] for k in k_list}
        map_list = { k: [] for k in k_list}
        for user in tqdm(user_list):
            test_item_list = list(item_set - train_record[user])
            item_score_map = dict()
            scores = self._get_scores(np.array([user]*len(test_item_list)),
                                      np.array(test_item_list),
                                      np.array(test_item_list))
            items = np.array(test_item_list)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
            #item sorted is the item-score pair, item means the items' ids
            item_sorted = [i[0] for i in item_score_pair_sorted]
            
            for k in k_list:
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
        user = torch.from_numpy(user)
        item_list = torch.from_numpy(item_list)
        head_list = torch.from_numpy(head_list)
        self.user_indices = user.long().to(self.device)
        self.item_indices = item_list.long().to(self.device)
        self.head_indices = head_list.long().to(self.device)

        self.MKR_model.eval()
        outputs = self.MKR_model(self.user_indices, self.item_indices,
                                 self.head_indices)
        user_embeddings, item_embeddings, _, scores = outputs
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

        