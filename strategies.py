import numpy as np
import random

global funcs 

def random_date(N, random_seed=666):
    """generate dating pair randomly

    Args:
        N (_type_): _description_
        match_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    random.seed(random_seed)
    romeos = [i for i in range(N)]
    juliets = [i for i in range(N, 2*N)]

    l1 = random.sample(romeos, N)
    l2 = random.sample(juliets, N)
    
    return list(zip(l1, l2))


def greedy(N, match_prob, mode, temp_type=[], ratio1=0.5, ratio2=0.5, random_seed=666):
    """based on current guess on the type and the matching prob to 
    pair romeo and juliet greedily.

    Returns:
        _type_: _description_
    """
    random.seed(random_seed)
    romeos = [i for i in range(N)]
    juliets = [i for i in range(N, 2*N)]
    dates = []
    if temp_type:
        [temp_rH, temp_rL, temp_jH, temp_jL] = temp_type
    else:
        temp_rH = [i for i in range(int(N*ratio1))]
        temp_rL = [i for i in range(int(N*ratio1), N)]
        temp_jH = [i+N for i in range(int(N*ratio2))]
        temp_jL = [i+N for i in range(int(N*ratio2), N)]
    
    if mode == 1:
        greedy_order = [(0, 0), (1, 1), (0, 1)]
    else:
        greedy_order = [(0, 1), (0, 0), (1, 1)]
    
    for pair in greedy_order:
        t1, t2 = pair
        # check whether the target list is empty
        if t1 == 0:
            lis1, id1 = (temp_rH, 0) if temp_rH else (temp_rL, 1)
        else:
            lis1, id1 = (temp_rL, 1) if temp_rL else (temp_rH, 0)
        
        if t2 == 0:
            lis2, id2 = (temp_jH, 0) if temp_jH else (temp_jL, 1)
        else:
            lis2, id2 = (temp_jL, 1) if temp_jL else (temp_jH, 0)

        if not lis1 or not lis2:
            continue
        
        l = min(len(lis1), len(lis2))
        lis1_ = random.sample(lis1, l)
        lis2_ = random.sample(lis2, l)
        
        # remove the paired people
        if l == len(lis1):
            if id1 == 0:
                temp_rH = []
            else:
                temp_rL = []
                
            if id2 == 0:
                temp_jH = list(set(lis2) - set(lis2_))
            else:
                temp_jL = list(set(lis2) - set(lis2_))
            
        else:
            if id2 == 0:
                temp_jH = []
            else:
                temp_jL = []
                
            if id1 == 0:
                temp_rH = list(set(lis1) - set(lis1_))
            else:
                temp_rL = list(set(lis1) - set(lis1_))

        dates += list(zip(lis1_, lis2_))
    return dates

funcs = [random_date, greedy]

