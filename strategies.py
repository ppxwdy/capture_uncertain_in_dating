import numpy as np
import random



# TODO: change the functions input
global funcs 

# def upper_bound(N, match_prob, romeo_identity, juliet_identity, ratio1, ratio2):
#     """_summary_

#     Args:
#         N (_type_): _description_
#         match_prob (_type_): _description_
#         romeo_identity (_type_): _description_
#         juliet_identity (_type_): _description_
#         ratio1 (_type_): _description_
#         ratio2 (_type_): _description_
#     """
#     temp_rH = [i for i in range(int(N*ratio1))]
#     temp_rL = [i for i in range(int(N*ratio1), N)]
#     temp_jH = [i+N for i in range(int(N*ratio2))]
#     temp_jL = [i+N for i in range(int(N*ratio2), N)]

#     dates = []
#     [HH, HL], [LH, LL] = match_prob
#     success_expectation = [(HH*HH, (0, 0)), (HL*LH, (0, 1)), (LL*LL, (1,1))]
#     success_expectation.sort(reverse=True, key=lambda x:x[0])
#     greedy_order = [pair[1] for pair in success_expectation]
#     # print(greedy_order)
    
#     random.seed(666)
#     for pair in greedy_order:
#         t1, t2 = pair
#         # check whether the target list is empty
#         if t1 == 0:
#             lis1, id1 = (temp_rH, 0) if temp_rH else (temp_rL, 1)
#         else:
#             lis1, id1 = (temp_rL, 1) if temp_rL else (temp_rH, 0)
        
#         if t2 == 0:
#             lis2, id2 = (temp_jH, 0) if temp_jH else (temp_jL, 1)
#         else:
#             lis2, id2 = (temp_jL, 1) if temp_jL else (temp_jH, 0)

#         if not lis1 or not lis2:
#             continue
        
#         # random.seed(666)
#         # use the shorter list first
#         l = min(len(lis1), len(lis2))
#         lis1_ = random.sample(lis1, l)
#         lis2_ = random.sample(lis2, l)
        
#         # remove the paired people
#         if l == len(lis1):
#             if id1 == 0:
#                 temp_rH = []
#             else:
#                 temp_rL = []
                
#             if id2 == 0:
#                 temp_jH = list(set(lis2) - set(lis2_))
#             else:
#                 temp_jL = list(set(lis2) - set(lis2_))
            
#         else:                              
#             if id2 == 0:
#                 temp_jH = []
#             else:
#                 temp_jL = []
                
#             if id1 == 0:
#                 temp_rH = list(set(lis1) - set(lis1_))
#             else:
#                 temp_rL = list(set(lis1) - set(lis1_))

#         dates += list(zip(lis1_, lis2_))
#     return dates
    


def random_date(N):
    """generate dating pair randomly

    Args:
        N (_type_): _description_
        match_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    romeos = [i for i in range(N)]
    juliets = [i for i in range(N, 2*N)]

    random.seed(666)
    l1 = random.sample(romeos, N)
    l2 = random.sample(juliets, N)
    
    return list(zip(l1, l2))


def greedy(N, match_prob, mode, temp_type=[], ratio1=0.5, ratio2=0.5):
    """based on current guess on the type and the matching prob to 
    pair romeo and juliet greedily.

    Returns:
        _type_: _description_
    """
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
        
    [HH, HL], [LH, LL] = match_prob
    success_expectation = [(HH*HH, (0, 0)), (HL*LH, (0, 1)), (LL*LL, (1,1))]
    success_expectation.sort(reverse=True, key=lambda x:x[0])
    greedy_order = [pair[1] for pair in success_expectation]
    
    if mode == 1:
        greedy_order = [(0, 0), (1, 1), (0, 1)]
    else:
        greedy_order = [(0, 1), (0, 1), (1, 1)]
    
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
        
        # random.seed(666)
        # use the shorter list first
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


def strategy1():
    pass





def market_choice(N, match_prob, temp_type=[], mode=0):
    """_summary_

    Args:
        N (_type_): _description_
        match_prob (_type_): _description_
        temp_type (_type_): [temp_rH, temp_rL, temp_jH, temp_jL]
        mode (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    romeos = [i for i in range(N)]
    juliets = [i for i in range(N, 2*N)]
    # random
    random.seed(666)
    if mode == 0:
        random.sample(romeos, len(romeos))
        random.sample(juliets, len(juliets))
        return list(zip(romeos, juliets))
    # greedy
    elif mode == 1:
        dates = []
        [temp_rH, temp_rL, temp_jH, temp_jL] = temp_type
        [HH, HL], [LH, LL] = match_prob
        success_expectation = [(HH*HH, (0, 0)), (HL*LH, (0, 1)), (LL*LL, (1,1))]
        success_expectation.sort(reverse=True, key=lambda x:x[0])
        greedy_order = [pair[1] for pair in success_expectation]
        for pair in greedy_order:
            t1, t2 = pair
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
            
            # random.seed(666)
            l = min(len(lis1), len(lis2))
            lis1_ = random.sample(lis1, l)
            lis2_ = random.sample(lis2, l)
            
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
