import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from strategies import *
    
def initialization(N, ratio1, ratio2, initial_guess):
    """_summary_

    Args:
        N (_type_): the agent number on each side
        ratio1 (_type_): type_H:type_L on romeo side
        ratio2 (_type_): type_H:type_L on juliet side
        initial_guess (_type_): type prob guess

    Returns:
        _type_: _description_
    """   
    romeo_identity = {}
    romeo_guess = {}
    
    juliet_identity = {}       
    juliet_guess = {}
    
    for i in range(N):
        
        if i < N * ratio1:
            romeo_identity[i] = 0
        else:
            romeo_identity[i] = 1
        
        if i < N * ratio2:
            juliet_identity[i+N] = 0
        else:
            juliet_identity[i+N] = 1
        
        romeo_guess[i] = initial_guess
        juliet_guess[i+N] = initial_guess
        
    return romeo_identity, romeo_guess, juliet_identity, juliet_guess
    
    
def bayes(romeo, juliet, date_res, match_prob, romeo_guess, juliet_guess):
    """Update type guess based on the date result

    Args:
        romeo (_type_): _description_
        juliet (_type_): _description_
        date_res (_type_): _description_
        romeo_guess (_type_): _description_
        julie_guess (_type_): _description_
    """
    [HH, HL], [LH, LL] = match_prob
    # prior
    rH, rL = romeo_guess[romeo]
    jH, jL = juliet_guess[juliet]
    ph_success = HH * HH + HL * LH
    pl_success = LH * HL + LL * LL
    if date_res:
        # Romeo
        A = rH * (jH * HH * HH + jL * HL * LH)
        p_rh = A / (A + (rL * (jH * HL * LH + jL * LL * LL))) 
        p_rl = 1 - p_rh
        # Romeo
        B = jH * (rH * HH * HH + rL * HL * LH )
        p_jh = B / (B + (jL * (rH * HL * LH + rL * LL * LL)))
        p_jl = 1 - p_jh
    else:
        # Romeo
        A = rH * (jH * (1 - HH * HH) + jL * (1 - HL * LH))
        p_rh = A / (A + (rL * (jH * (1 - HL * LH) + jL * (1 - LL * LL))))
        p_rl = 1 - p_rh
        # Juliet
        B = jH * (rH * (1 - HH * HH) + rL * (1 - HL * LH))
        p_jh = B / (B + (jL * (rH * (1 - HL * LH) + rL * (1 - LL * LL))))
        p_jl = 1 - p_jh
    # if date_res:
        
    #     # posterior
    #     # Romeo
    #     p_rh = rH * ph_success / (rH * ph_success + rL * pl_success)
    #     p_rl = 1 - p_rh
    #     # juliet
    #     p_jh = jH * ph_success / (jH * ph_success + jL * pl_success)
    #     p_jl = 1 - p_jh
    # else:
    #     ph_fail = 1 - ph_success
    #     pl_fail = 1 - pl_success
    #     # Romeo
    #     p_rh = rH * ph_fail / (rH * ph_fail + rL * pl_fail)
    #     p_rl = 1 - p_rh
    #     # Juliet
    #     p_jh = jH * ph_fail / (jH * ph_fail + jL * pl_fail)
    #     # if juliet == 11:
    #     #     print(jH * ph_fail, (jH * ph_fail + jL * pl_fail))
    #     p_jl = 1 - p_jh
    # # if juliet == 11:
    # #     print('res is', [p_jh, p_jl], date_res, rH, rL, jH, jL, ph_fail, pl_fail)    
    return [p_rh, p_rl], [p_jh, p_jl]


def find_best_strategy(N, T, func, match_probs, mode=0, ratio1=0.5, ratio2=0.5, initial_guess=[0.5, 0.5]):
    """_summary_

    Args:
        N (_type_): _description_
        T (_type_): _description_
        func (_type_): _description_
        match_probs (_type_): _description_
        mode (int, optional): _description_. Defaults to 0.
        ratio1 (float, optional): _description_. Defaults to 0.5.
        ratio2 (float, optional): _description_. Defaults to 0.5.
        initial_guess (list, optional): _description_. Defaults to [0.5, 0.5].

    Returns:
        _type_: _description_
    """
    RHJH = match_probs[0][0] * match_probs[0][0]
    RHJL = match_probs[0][1] * match_probs[1][0]
    RLJH = match_probs[1][0] * match_probs[0][1]
    RLJL = match_probs[1][1] * match_probs[1][1]
    e_random = ratio1*ratio2 * RHJH + ratio1 * (1 - ratio2) * RHJL + (1 - ratio2) * ratio2 * RLJH + \
                (1 - ratio1) * (1 - ratio2) * RLJL
    
    HL = RLJH if ratio1 < ratio2 else RHJL
    e_same_t =  min(ratio1, ratio2) * RHJH + (1 - max(ratio1, ratio2)) * RLJL + abs(ratio1 - ratio2) * HL
    
    same = RHJH if ratio1 < ratio2 else RLJL
    e_diff_t = min(ratio1, (1-ratio2)) * RHJL + min(ratio2, (1-ratio1)) * RLJH +  abs(ratio1 + ratio2 - 1) * same
    rank = list(enumerate([e_random, e_same_t, e_diff_t]))
    rank.sort(reverse=True, key=lambda x:x[1])
    return rank[0][0]



def simulator(N, T, match_probs, ratio1=0.5, ratio2=0.5, initial_guess=[0.5, 0.5], func=funcs):
    """_summary_

    Args:
        N (_type_): _description_
        T (_type_): _description_
        match_probs (_type_): [[H likes H, H likes L], [L likes H, L likes L]]
        mode (int, optional): _description_. Defaults to 1.
        ratio1 (float, optional): _description_. Defaults to 0.5.
        ratio2 (float, optional): _description_. Defaults to 0.5.
        initial_guess (list, optional): _description_. Defaults to [0.5, 0.5].
    """    
    romeo_identity, romeo_guess, juliet_identity, juliet_guess = initialization(N, ratio1, ratio2, initial_guess)
    np.random.seed(666)
    success = 0
    success_ub = 0
    success_rand = 0

    opt = find_best_strategy(N, T, func, match_probs)
    
    record_strategy = []
    record_optimal = []
    record_random = []
    
    for t in range(T):
        temp_s = 0
        temp_o = 0
        temp_rand = 0
        if t == 0:
            dates = func[0](N)
        else:
            if opt == 0:
                dates = func[0](N)
            else:
                dates = func[1](N, match_probs, opt, [temp_rH, temp_rL, temp_jH, temp_jL])
        temp_rH = []
        temp_rL = []
        temp_jH = []
        temp_jL = []
        # print(dates)
        # print(t, len(dates))
        # strategy
        for romeo, juliet in dates:
            # check_res
            
            true_type_r = romeo_identity[romeo]
            true_type_j = juliet_identity[juliet]
            
            r2j = True if np.random.random() < match_probs[true_type_r][true_type_j] else False
            j2r = True if np.random.random() < match_probs[true_type_j][true_type_r] else False
            
            date_res = r2j and j2r
            if date_res:
                success += 1 
                temp_s += 1
            # print(romeo, juliet, date_res)
            romeo_guess[romeo], juliet_guess[juliet] = bayes(romeo, juliet, date_res, match_probs, romeo_guess, juliet_guess)
    
            if romeo_guess[romeo][0] >= romeo_guess[romeo][1]:
                temp_rH.append(romeo)
            else:
                temp_rL.append(romeo)
            
            if juliet_guess[juliet][0] >= juliet_guess[juliet][1]:
                temp_jH.append(juliet)
            else:
                temp_jL.append(juliet)
        
        
        # Upper bound
        # if t == 0:
        #     dates_ub = func[0](N)
        # else:
        if opt == 0:
            dates_ub = func[0](N)
        else:
            dates_ub = func[1](N, match_probs, opt, ratio1=ratio1, ratio2=ratio2)
        # dates_ub = func[-1](N, match_probs, romeo_identity, juliet_identity, ratio1, ratio2)
        # print(dates_ub)
        for romeo, juliet in dates_ub:
            true_type_r = romeo_identity[romeo]
            true_type_j = juliet_identity[juliet]
            
            r2j = True if np.random.random() < match_probs[true_type_r][true_type_j] else False
            j2r = True if np.random.random() < match_probs[true_type_j][true_type_r] else False
            date_res = r2j and j2r
            if date_res:
                success_rand += 1           
                temp_o += 1

        dates_rand = func[0](N)
        # dates_ub = func[-1](N, match_probs, romeo_identity, juliet_identity, ratio1, ratio2)
        # print(dates_ub)
        for romeo, juliet in dates_ub:
            true_type_r = romeo_identity[romeo]
            true_type_j = juliet_identity[juliet]
            
            r2j = True if np.random.random() < match_probs[true_type_r][true_type_j] else False
            j2r = True if np.random.random() < match_probs[true_type_j][true_type_r] else False
            date_res = r2j and j2r
            if date_res:
                success_rand += 1           
                temp_rand += 1
            
        record_strategy.append(temp_s)
        record_optimal.append(temp_o)
        record_random.append(temp_rand)

    
    return record_strategy, record_optimal, record_random

# simulator(100, 1000, funcs, [[0.8, 0.2], [0.8, 0.2]], ratio1=0.4, ratio2=0.2, mode=1)    


# r_s, r_o = simulator(1000, 1000, [[0.6, 0.25], [0.25, 0.5]], 0.5, 0.5)