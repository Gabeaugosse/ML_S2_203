import random

def shuffle(lst):
    while True:
        perm = lst[:]
        random.shuffle(perm)
        if all(perm[i] != lst[i] for i in range(len(lst))):
            return perm