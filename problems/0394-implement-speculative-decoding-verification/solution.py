import numpy as np

def subtract_lists(list1: list, list2: list) -> list:
    res = []

    sum_diff = 0
    for i in range(len(list1)):
        sum_diff += max(0, list1[i] - list2[i])


    for i in range(len(list1)):
        res.append(max(0, list1[i] - list2[i]) / sum_diff)

    return res

def rejection_sampling(target_prob: list, draft_prob: list, accepted_tokens: list, resample_coin: float) -> None:
    residual_dist = subtract_lists(target_prob, draft_prob)

            

    cumsum = residual_dist[0]

    residual_dist_index = 0

    while(cumsum < resample_coin):
        #print(residual_dist, cumsum)
        residual_dist_index += 1
        cumsum += residual_dist[residual_dist_index]
        

    accepted_tokens.append(residual_dist_index)

def speculative_decode_verify(draft_tokens: list, draft_probs: list, target_probs: list, coin_flips: list, resample_coin: float) -> list:
    """
    Verify draft tokens using speculative decoding.
    
    Args:
        draft_tokens: List of K drafted token indices
        draft_probs: K x V array, draft model distributions at each position
        target_probs: K x V array, target model distributions at each position
        coin_flips: K random values in [0,1) for acceptance decisions
        resample_coin: Random value in [0,1) for resampling on rejection
    
    Returns:
        List of accepted/resampled token indices
    """
    accepted_tokens = []

    
    
    for i, draft_token in enumerate(draft_tokens):
        draft_prob = draft_probs[i]
        target_prob = target_probs[i]
        coin_flip = coin_flips[i]

        if min(1, target_prob[draft_token] / draft_prob[draft_token]) >= coin_flip:
            accepted_tokens.append(draft_token)
        else:
            rejection_sampling(target_prob, draft_prob, accepted_tokens, resample_coin)
            break

    
        
    return accepted_tokens
        
