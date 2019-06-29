import copy
import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'music': 'user_artists.dat',
                         'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0, 'news': 0})


def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        #here convert the corresponding relation, both item and the 
        #and the corresponding entity to the same id
        data = line.strip().split()
        item_index = data[0]
        satori_id = data[1]
        # 对齐user index和entity
        user_index_old2new[item_index] = i
        entity_id2index[satori_id] = i

        i += 1


def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(user_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        user_index_old = array[0]
        # if item_index_old not in user_index_old2new:  # if the item is not in the item-entity set
        if user_index_old not in user_index_old2new: 
            continue
        user_index = user_index_old2new[user_index_old] 
        # item_index = user_index_old2new[item_index_old] #the items' id is the same as its corresponding entity's

        # user_index_old = int(array[0]) # get the old interaction user/item id 

        rating = float(array[2])
        if user_index_old not in user_pos_ratings:
            user_pos_ratings[user_index_old] = set()
        user_pos_ratings[user_index_old].add(item_index_old)
        # if rating >= THRESHOLD[DATASET]: # if the rating is higher than the threshold 
        #     if user_index_old not in user_pos_ratings:
        #         user_pos_ratings[user_index_old] = set()
        #     user_pos_ratings[user_index_old].add(item_index) # add the item to the user pos ratings
        # else:
        #     if user_index_old not in user_neg_ratings: # else add to neg ratings
        #         user_neg_ratings[user_index_old] = set()
        #     user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new: 
            user_index_old2new[user_index_old] = user_cnt # if the user not in userid2newid, add the map relation
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]#here convert the old user id to new user id

        for item in pos_item_set: # for this user, the pos items will be writen
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set # the whole items - the pos items
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old] # if the user has neg ratings, uw-set should also - neg ratings 
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):# for uw-set we do random choice,replace means no duplicated values
            writer.write('%d\t%d\t0\n' % (user_index, item)) # the neg sample is selected from the uw-set
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg.txt file ...')
    entity_cnt = len(entity_id2index)
    raw_entity_id2index = copy.deepcopy(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    file = open('../data/' + DATASET + '/kg.txt', encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in raw_entity_id2index: #if the head not in the corresponding (item,entity) file, this triplet will be dropped
            continue
        head = entity_id2index[head_old] # here we got the entity id which is the same as its corresponding item's id

        if tail_old not in entity_id2index: 
        #if the tail is not in the entity2id then we add the map, but here introduced a problem into our system,
        # since sometimes the tail entity also appears in the 'head' but it does not appears in the 
        # 'item-entity' file so we changed this part somehow
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index: # for each realtion, it will be added in the rel2id
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail)) # the new triplet
    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset

    entity_id2index = dict()
    relation_id2index = dict()
    user_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
