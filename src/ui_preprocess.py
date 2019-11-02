import copy
import argparse
import numpy as np
import copy 
import paths

def read_item_index_to_entity_id_file():
    item_file = paths.item_file
    user_file = paths.user_file
    print('reading user index to entity id file: ' + user_file + ' ...')
    print('reading item index to entity id file: ' + item_file + ' ...')
    item2index = {}
    user2index = {}
    entity2index = {}
    i = 0
    for line in open(item_file, 'r',encoding='utf-8').readlines():
        #here convert the corresponding relation, both item and the 
        #and the corresponding entity to the same id
        data = line.strip().split()
        item_index = data[0]
        entity_id = data[1]
        item2index[item_index] = i
        entity2index[entity_id] = i
        i += 1
    for line in open(user_file,'r', encoding='utf-8').readlines():
        data = line.strip().split()
        user_index = data[0]
        entity_id = data[1]
        user2index[user_index] = i
        entity2index[entity_id] = i
        i+=1
    return item2index,user2index,entity2index


def convert_rating(item2index,user2index,user=True):
    file_path = paths.rating_file
    print('reading rating file ...')
    item_set = set(item2index.values())
    user_set = set(user2index.values())

    user_pos_ratings = {}
    user_neg_ratings = {}

    for line in open(file_path,'r',encoding='utf-8').readlines():
        data = line.strip().split('::')
        user_index = data[0]
        item_index = data[1]

        # if an user_index/item_index in interaction file has never appeared in the margin files
        # it means that the user or the item has no corresponding head
        if user_index not in user2index or item_index not in item2index:
            print(user_index)
            print(user2index[user_index])
            print(item_index)
            print(item2index[item_index])
            continue
        user_index = user2index[user_index]
        item_index = item2index[item_index]

        if user_index not in user_pos_ratings:
            user_pos_ratings[user_index] = set()
        user_pos_ratings[user_index].add(item_index)

    writer = open(paths.rating_final_file,'w',encoding='utf-8')
    for user_id, pos_item_set in user_pos_ratings.items():
        for item_id in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_id, item_id))
        
        un_interacted = item_set - pos_item_set
        # 每一个user，从其uninteracted的item中选出和interacted的一样长的作为负样本,写入n_ratings
        for item_id in np.random.choice(list(un_interacted), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_id, item_id))
    writer.close()
    print('number of users: %d' % len(user_set))
    print('number of items: %d' % len(item_set))


def convert_kg(entity2index,user=True):
    rel2index={}
    n_entity = len(entity2index)
    all_kg_entity = copy.deepcopy(entity2index)
    n_rel_ = 0

    # generate kg_final_user/kg_final_item
    if not user:
        user_or_item_idx = item2index
        out_kg_path = paths.kg_final_item_file   
    else:
        user_or_item_idx = user2index
        out_kg_path = paths.kg_final_user_file
    writer = open(out_kg_path,'w',encoding='utf-8')
    for line in open(paths.kg_file,encoding='utf-8').readlines():
        data = line.strip().split()
        head_index = data[0]
        rel_ = data[1]
        tail_index = data[2]

        # if the head has no corresponding item/user
        if head_index not in entity2index:
            continue
        # if head is not user/item continue, controled by *user* param 
        if head_index not in user_or_item_idx:
            continue
        head_index = entity2index[head_index]

        # if the tail has not appeared before, tail not need to correspond to user/item
        if tail_index not in all_kg_entity:
            all_kg_entity[tail_index] = n_entity
            n_entity += 1
        tail_index = all_kg_entity[tail_index]

        if rel_ not in rel2index:
            rel2index[rel_] = n_rel_
            n_rel_ += 1
        relation_index = rel2index[rel_]
        writer.write('%d\t%d\t%d\n' % (head_index, relation_index, tail_index))
    writer.close()
    return rel2index

if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    # entity2index = {}
    # item2index = {}
    # user2index= {}

    item2index,user2index,entity2index = read_item_index_to_entity_id_file()

    convert_rating(item2index,user2index)
    convert_kg(entity2index) #convert user kg
    convert_kg(entity2index,False) # convert item kg

    user2index.update(item2index)
    import pickle
    pickle.dump(user2index,open(paths.ui2index,'wb'))

    
        
                


