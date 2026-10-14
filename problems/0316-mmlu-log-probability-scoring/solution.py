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
    probs = np.exp(rowwise_log_softmax(log_probs))
    
    
    correct = 0
    predictions = []
    sum_probs = 0
    n = len(correct_answers)

    for prob, correct_answer, log_prob in zip(probs, correct_answers, log_probs):
        predicted_answer = np.argmax(log_prob, axis=0)

        if predicted_answer == correct_answer:
            correct += 1
        sum_probs += prob[correct_answer]
        predictions.append(int(predicted_answer))



    return {
        'accuracy': round(correct / n, 4),
        'predictions': predictions,
        'avg_correct_prob': round(sum_probs / n, 4)
    }
