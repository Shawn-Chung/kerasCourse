import os
import json
import random

data_lst = []

clss = os.listdir('./dataset')
clss.remove('.DS_Store')
for cls in clss:
    files = os.listdir(os.path.join('./dataset', cls))
    files = [a for a in files if '.jpg' in a]
    for f in files:
        data_lst.append([os.path.join('./dataset', cls, f), cls])
random.shuffle(data_lst)

info_dict = dict()
info_dict['data_lst'] = data_lst
with open("./train_data.json","w") as f:
    json.dump(info_dict, f)

