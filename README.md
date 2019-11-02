# bi-MKR
this project is heavily based on [MKR.Torch](https://github.com/hsientzucheng/MKR.PyTorch)  

## 本文的模型输入  
本文模型的输入user_index2entity_id,item_index2entity_id,ratings  
经过uipreprocess之后user_index2entity_id和item_index2entity_id得到userid/itemid到*index*,同时也有entityid到*index*的三个字典。**item2index和user2index应该是没有交集的**。  
得到这三个字典之后通过convert kg，可以将输入的kg文件进行过滤得到kgfinal，分别生成user侧和item侧的kg final文件。   
dataloader中会将ncf部分的user和item嵌入到同一个空间中.  n_user为用户数目,n_item为item数目。
