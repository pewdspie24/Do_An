import random
import copy

with open("data/splits/custom/tmp.txt", "r+") as f:
    lst = f.read().splitlines()

new_lst = []
tmp_lst = copy.deepcopy(lst)
random.shuffle(tmp_lst)

for idx, x in enumerate(tmp_lst):
    if idx > len(tmp_lst)//8:
        break
    new_lst.append(x)
    lst.remove(x)
    tmp_lst.pop(idx)

with open('data/splits/custom/trn/fold3.txt', 'w') as fp:
    for item in lst:
        fp.write("%s\n" % item)

with open('data/splits/custom/val/fold3.txt', 'w') as fp:
    for item in new_lst:
        fp.write("%s\n" % item)
