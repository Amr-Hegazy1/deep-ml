import numpy as np

def rowwise_log_softmax(x):
    # Find the maximum value along each row (axis=1)
    # keepdims=True ensures proper shape broadcasting (Nx1)
    row_max = np.max(x, axis=1, keepdims=True)
    
    # Shift values to prevent overflow
    shifted_x = x - row_max
    
    # Compute the log of the sum of exponentials row-wise
    log_sum_exp = np.log(np.sum(np.exp(shifted_x), axis=1, keepdims=True))
    
    # Return log-softmax
    return shifted_x - log_sum_exp

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """

    log_probs = np.array(log_probs)
    correct_answers = np.array(correct_answers)
    probs = np.exp(rowwise_log_softmax(log_probs))
    
    
    correct = 0
    predictions = []
    sum_probs = 0
    n = len(correct_answers)

    
    predicted_answers = np.argmax(log_probs, axis=1).flatten()

    correct = int(np.sum(predicted_answers == correct_answers))
    
    sum_probs = np.sum(probs[np.arange(len(probs)), correct_answers])
    



    return {
        'accuracy': round(correct / n, 4),
        'predictions': list(predicted_answers),
        'avg_correct_prob': round(sum_probs / n, 4)
    }
