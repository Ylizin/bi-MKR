import numpy as np
import os
import paths

class RSDataset:
    def __init__(self, args):
        self.args = args
        self.n_user, self.n_item, self.raw_data, self.data, self.indices = self._load_rating()

    def __getitem__(self, index):
        return self.raw_data[index] # raw data is the raw ratings matrix

    def _load_rating(self):
        print('Reading rating file')

        rating_file = paths.rating_final_file
        rating_file_prefix,_ = os.path.splitext(rating_file)
        if os.path.exists(rating_file_prefix + '.npy'):
            rating_np = np.load(rating_file_prefix + '.npy')
        else:
            rating_np = np.loadtxt(rating_file, dtype=np.int32) # load the ratings same as kg
            np.save(rating_file_prefix + '.npy', rating_np)

        n_user = len(set(rating_np[:, 0]))
        n_item = len(set(rating_np[:, 1]))
        raw_data, data, indices = self._dataset_split(rating_np)
        # user,item
        return n_user, n_item, raw_data, data, indices


    def _dataset_split(self, rating_np):
        print('Splitting dataset')

        # train:eval:test = 6:2:2
        eval_ratio = 0.0 # 0.001 maybe better 
        test_ratio = 0.3
        n_ratings = rating_np.shape[0] # the num of the ratings records

                                        #the obj choice conducted on is the indices of all ratings
        eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False) # do random choices get eval
        left = set(range(n_ratings)) - set(eval_indices) # get left ratings
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False) # do random choice get test
        train_indices = list(left - set(test_indices)) # get train

        train_data = rating_np[train_indices]
        eval_data = rating_np[eval_indices]
        test_data = rating_np[test_indices]

        return rating_np, [train_data, eval_data, test_data], [train_indices, eval_indices, test_indices]

class KGDataset:
    def __init__(self, args):
        self.args = args
        self.n_entity, self.n_relation, self.kg = self._load_kg()

    def __getitem__(self, index):
        return self.kg[index]

    def __len__(self):
        return len(self.kg)

    def _load_kg(self):
        print('Reading KG file')

        kg_file = paths.kg_final_user_file
        if not self.args.user_enhanced:
            print('using item enhanced')
            kg_file = paths.kg_final_item_file
        else: 
            print('using user enhanced')

        kg_file_prefix,_ = os.path.splitext(kg_file)
        if os.path.exists(kg_file_prefix + '.npy'):
            kg = np.load(kg_file_prefix + '.npy')
        else:
            kg = np.loadtxt(kg_file, dtype=np.int32)# now kg is a numpy array
            np.save(kg_file_prefix + '.npy', kg) # save the kg txt as npy, kg_final.npy

        # n_entity = len(set(kg[:, 0]) | set(kg[:, 2])) #take the columns 0 and 2 -- head and tail, they are entities
        n_entity = np.max(kg[:,2])+1
        n_relation = len(set(kg[:, 1]))

        return n_entity, n_relation, kg

