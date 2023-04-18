from env import initialization
import numpy as np
import copy
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


def best_strategy(match_probs, ratio1, ratio2):

    [[H2H, H2L], [L2H, L2L]] = match_probs
    HH = H2H**2
    HL = H2L * L2H
    LL = L2L**2

    e_random = ratio1*ratio2*HH + ((1-ratio1)*ratio2 + ratio1*(1 - ratio2))*HL +\
                (1 - ratio1)*(1 - ratio2)*LL
                
    ratio1, ratio2 = max(ratio1, ratio2),  min(ratio1, ratio2)
                
    rh1, rl1 = ratio1, 1 - ratio1
    rh2, rl2 = ratio2, 1 - ratio2             
    
    r = abs(rh1 - rh2)
    e_same = rh2 * HH + rl1 * LL + r * HL
    
    r1 = 0 if rh1 - min(rh1, rl2) <=0 else rh1 - min(rh1, rl2)
    r2 = 0 if rl1 - min(rh2, rl1) <=0 else rl1 - min(rh2, rl1)
    e_mix = (min(rh1, rl2) + min(rh2, rl1)) * HL + r1 * HH + r2 * LL
    # print(r1, r2)
    

    # e_same = ratio2*HH + (1-ratio1)*LL + abs(ratio1 - 1 + ratio1) * HL  # wrong

    # e_mix = (min(ratio1, 1-ratio2) + min(ratio2, 1 - ratio1)) * HL  # wrong
    # if ratio1 > 1 - ratio2:
    #     e_mix += abs(ratio1+ratio2 -1) * LL
    # else:
    #     e_mix += abs(ratio1+ratio2 -1) * HH

    mode = [(0, e_random), (1, e_same), (2, e_mix)]
    mode.sort(key=lambda x:x[1])
    # print(mode)
    return mode[-1][0]


def bayes(romeo, juliet, date_res, match_prob, romeo_guess, juliet_guess, ratio1, ratio2, full=True):
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
   
    if full :
        if date_res:
            # Romeo
            A = rH * (jH * HH * HH + jL * HL * LH)
            p_rh = A / (A + (rL * (jH * HL * LH + jL * LL * LL))) 
            p_rl = 1 - p_rh
            # juliet
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
    else:
        r_ph_success = ratio2 * HH * HH + (1-ratio2) * HL * LH
        r_pl_success = ratio2 * LH * HL + (1-ratio2) * LL * LL 
        j_ph_success = ratio1 * HH * HH + (1-ratio1) * HL * LH
        j_pl_success = ratio1 * LH * HL + (1-ratio1) * LL * LL 
        if date_res:
            
            # posterior
            # Romeo
            p_rh = rH * r_ph_success / (rH * r_ph_success + rL * r_pl_success)
            p_rl = 1 - p_rh
            # juliet
            p_jh = jH * j_ph_success / (jH * j_ph_success + jL * j_pl_success)
            p_jl = 1 - p_jh
        else:
            r_ph_fail = 1 - r_ph_success
            r_pl_fail = 1 - r_pl_success
            j_ph_fail = 1 - j_ph_success
            j_pl_fail = 1 - j_pl_success
            # Romeo
            p_rh = rH * r_ph_fail / (rH * r_ph_fail + rL * r_pl_fail)
            p_rl = 1 - p_rh
            # Juliet
            p_jh = jH * j_ph_fail / (jH * j_ph_fail + jL * j_pl_fail)
            # if juliet == 11:
            #     print(jH * ph_fail, (jH * ph_fail + jL * pl_fail))
            p_jl = 1 - p_jh
        # if juliet == 11:
        #     print('res is', [p_jh, p_jl], date_res, rH, rL, jH, jL, ph_fail, pl_fail)    
    return [p_rh, p_rl], [p_jh, p_jl]


def simulation(N, T, match_probs, ratio1=0.5, ratio2=0.5, initial_guess=[0.5, 0.5], func=funcs, full_info=True, epsilon=0.1, random_seed=666):
    romeo_identity, romeo_guess, juliet_identity, juliet_guess = initialization(N, ratio1, ratio2, initial_guess)

    romeo_guess_greedy = copy.deepcopy(romeo_guess)
    juliet_guess_greedy = copy.deepcopy(juliet_guess)

    romeo_guess_random = copy.deepcopy(romeo_guess)
    juliet_guess_random = copy.deepcopy(juliet_guess)

    romeo_guess_e_greedy = copy.deepcopy(romeo_guess)
    juliet_guess_e_greedy = copy.deepcopy(juliet_guess)

    record_greedy = []
    record_opt = []
    record_random = []
    record_e_greedy = []

    mode = best_strategy(match_probs, ratio1, ratio2)
    # print(mode)
    for t in range(T):
        
        # greedy
        match_greedy = 0
        if t == 0:
            dates = func[0](N, random_seed=t+random_seed)
        else:
            if mode == 0:
                dates = func[0](N, random_seed=t+random_seed)
            else:
                dates = func[1](N, match_probs, mode, [temp_rH_greedy, temp_rL_greedy, temp_jH_greedy, temp_jL_greedy], ratio1, ratio2, random_seed=t+random_seed)

        temp_rH_greedy = []
        temp_rL_greedy = []
        temp_jH_greedy = []
        temp_jL_greedy = []
        temp_r_greedy = []
        temp_j_greedy = []

        for romeo, juliet in dates:
            type_r = romeo_identity[romeo]
            type_j = juliet_identity[juliet]

            r2j = True if np.random.random() < match_probs[type_r][type_j] else False
            j2r = True if np.random.random() < match_probs[type_j][type_r] else False
            date_res = r2j and j2r

            if date_res:
                match_greedy += 1
            
            # bayes
            romeo_guess_greedy[romeo], juliet_guess_greedy[juliet] = bayes(romeo, juliet, date_res, match_probs, romeo_guess_greedy, juliet_guess_greedy, ratio1, ratio2,full_info)
            temp_r_greedy.append((romeo, romeo_guess_greedy[romeo][0]))
            temp_j_greedy.append((juliet, juliet_guess_greedy[juliet][0]))

        temp_r_greedy.sort(reverse=True, key=lambda x:x[1])
        temp_j_greedy.sort(reverse=True, key=lambda x:x[1])
        for i in range(N):
            if i < N*ratio1:
                temp_rH_greedy.append(temp_r_greedy[i][0])
            else:
                temp_rL_greedy.append(temp_r_greedy[i][0])
            
            if i < N*ratio2:
                temp_jH_greedy.append(temp_j_greedy[i][0])
            else:
                temp_jL_greedy.append(temp_j_greedy[i][0])

        
        # opt
        match_opt = 0
        dates = func[1](N, match_probs, mode, ratio1=ratio1, ratio2=ratio2, random_seed=t+random_seed)
        # print(dates)
        for romeo, juliet in dates:
            true_type_r = romeo_identity[romeo]
            true_type_j = juliet_identity[juliet]
            
            r2j = True if np.random.random() < match_probs[true_type_r][true_type_j] else False
            j2r = True if np.random.random() < match_probs[true_type_j][true_type_r] else False
            date_res = r2j and j2r
            # print(romeo, true_type_r, juliet, true_type_j, r2j, j2r, match_probs[true_type_r][true_type_j], match_probs[true_type_j][true_type_r])
            if date_res:
                match_opt += 1
        

        # random
        match_rand = 0
        
        if t == T-1:
            dates = func[1](N, match_probs, mode, [temp_rH_rand, temp_rL_rand, temp_jH_rand, temp_jL_rand], ratio1, ratio2, random_seed=t+random_seed)
            temp_rH_rand = []
            temp_rL_rand = []
            temp_jH_rand = []
            temp_jL_rand = []
            temp_r_rand = []
            temp_j_rand = []
            for romeo, juliet in dates:
                true_type_r = romeo_identity[romeo]
                true_type_j = juliet_identity[juliet]
                
                r2j = True if np.random.random() < match_probs[true_type_r][true_type_j] else False
                j2r = True if np.random.random() < match_probs[true_type_j][true_type_r] else False
                date_res = r2j and j2r
                if date_res:
                    match_rand += 1        

                romeo_guess_random[romeo], juliet_guess_random[juliet] = bayes(romeo, juliet, date_res, match_probs, romeo_guess_random, juliet_guess_random, ratio1, ratio2, full_info)
                temp_r_rand.append((romeo, romeo_guess_random[romeo][0]))
                temp_j_rand.append((juliet, juliet_guess_random[juliet][0]))

        else:
            dates = func[0](N, random_seed=t+random_seed)
            temp_rH_rand = []
            temp_rL_rand = []
            temp_jH_rand = []
            temp_jL_rand = []
            temp_r_rand = []
            temp_j_rand = []
            for romeo, juliet in dates:
                true_type_r = romeo_identity[romeo]
                true_type_j = juliet_identity[juliet]
                
                r2j = True if np.random.random() < match_probs[true_type_r][true_type_j] else False
                j2r = True if np.random.random() < match_probs[true_type_j][true_type_r] else False
                date_res = r2j and j2r
                if date_res:
                    match_rand += 1        

                romeo_guess_random[romeo], juliet_guess_random[juliet] = bayes(romeo, juliet, date_res, match_probs, romeo_guess_random, juliet_guess_random, ratio1, ratio2, full_info)
                temp_r_rand.append((romeo, romeo_guess_random[romeo][0]))
                temp_j_rand.append((juliet, juliet_guess_random[juliet][0]))

        temp_r_rand.sort(reverse=True, key=lambda x:x[1])
        temp_j_rand.sort(reverse=True, key=lambda x:x[1])

        for i in range(N):
            if i < N*ratio1:
                temp_rH_rand.append(temp_r_rand[i][0])
            else:
                temp_rL_rand.append(temp_r_rand[i][0])
            
            if i < N*ratio2:
                temp_jH_rand.append(temp_j_rand[i][0])
            else:
                temp_jL_rand.append(temp_j_rand[i][0])


        # e-greedy
        match_e = 0
        if t == 0:
            dates = func[0](N, random_seed=t+random_seed) 
        else:
            np.random.seed(t+random_seed)
            p = np.random.random()

            # explore
            if p < epsilon:
                dates = func[0](N, random_seed=t+random_seed)
            else:
                dates = func[1](N, match_probs, mode, [temp_rH_e, temp_rL_e, temp_jH_e, temp_jL_e], ratio1, ratio2, random_seed=t+random_seed)
        temp_rH_e = []
        temp_rL_e = []
        temp_jH_e = []
        temp_jL_e = []
        temp_r_e = []
        temp_j_e = []
        for romeo, juliet in dates:
            true_type_r = romeo_identity[romeo]
            true_type_j = juliet_identity[juliet]
            r2j = True if np.random.random() < match_probs[true_type_r][true_type_j] else False
            j2r = True if np.random.random() < match_probs[true_type_j][true_type_r] else False
            date_res = r2j and j2r
            if date_res:
                match_e += 1        

            romeo_guess_e_greedy[romeo], juliet_guess_e_greedy[juliet] = bayes(romeo, juliet, date_res, match_probs, romeo_guess_e_greedy, juliet_guess_e_greedy, ratio1, ratio2, full_info)
            temp_r_e.append((romeo, romeo_guess_e_greedy[romeo][0]))
            temp_j_e.append((juliet, juliet_guess_e_greedy[juliet][0]))

        temp_r_e.sort(reverse=True, key=lambda x:x[1])
        temp_j_e.sort(reverse=True, key=lambda x:x[1])
        for i in range(N):
            if i < N*ratio1:
                temp_rH_e.append(temp_r_e[i][0])
            else:
                temp_rL_e.append(temp_r_e[i][0])
            
            if i < N*ratio2:
                temp_jH_e.append(temp_j_e[i][0])
            else:
                temp_jL_e.append(temp_j_e[i][0])

        record_greedy.append(match_greedy)
        record_opt.append(match_opt)
        record_random.append(match_rand)
        record_e_greedy.append(match_e)

    correct_rate_greedy = 0
    correct_rate_random = 0
    correct_rate_e_greedy = 0
    target = int(N*ratio1)
    for i in range(target):
        greedy_ = temp_rH_greedy[i]
        random_ = temp_rH_rand[i]
        e_greedy_ = temp_rH_e[i]

        if greedy_ < target:
            correct_rate_greedy += 1
        if random_ < target:
            correct_rate_random += 1
        if e_greedy_ < target:
            correct_rate_e_greedy += 1
    
    # print('greedy', str(correct_rate_greedy/target)[:7])
    # print('random', str(correct_rate_random/target)[:7])
    # print('e_greedy', str(correct_rate_e_greedy/target)[:7])

    return record_greedy, record_opt, record_random, record_e_greedy,  correct_rate_greedy, correct_rate_random, correct_rate_e_greedy

# N = 100
# T = 4
# match_probs = [[0.1, 0.1], [0.8, 0.1]]
# ratio1 = 0.6
# ratio2 = 0.2
# initial_guess = [ratio1, ratio2]
# func=funcs
# full_info=True
# epsilon=0.1
# record_greedy, record_opt, record_random, record_e_greedy, correct_rate_greedy, correct_rate_random, correct_rate_e_greedy = simulation(N, T, match_probs, ratio1, ratio1, initial_guess)
# print(record_opt[-1]/N)

# match_probs = [[0.8, 0.3], [0.3, 0.7]]
# ratio1 = 0.3
# ratio2 = 0.3
# best_strategy(match_probs, ratio1, ratio2)