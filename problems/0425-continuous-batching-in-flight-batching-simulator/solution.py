def check_requests(
    requests: list[dict],
    execute: list[dict],
    queue: list[dict],
    current_time: int,
    max_batch_size: int,
) -> None:
    
    for request in requests[:]:
        if request['arrival_time'] <= current_time:
            request['total_tokens'] = request['tokens_needed']
            queue.append(request)
            requests.remove(request)

    while len(queue) > 0 and max_batch_size > len(execute):
        request = queue.pop(0)
        request['start_time'] = current_time
        execute.append(request)


def step(
    requests: list[dict],
    execute: list[dict],
    queue: list[dict],
    finished: list[dict],
    current_time: int,
    max_batch_size: int,
) -> None:
    
    for request in execute[:]:
        request['tokens_needed'] -= 1

        if request['tokens_needed'] == 0:
            request['completion_time'] = current_time
            execute.remove(request)
            finished.append(request)


def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """Simulate continuous batching for LLM inference.

    Args:
        requests: List of dicts with 'arrival_time' (int) and 'tokens_needed'
          (int)
        max_batch_size: Maximum concurrent sequences in the batch

    Returns:
        Dict with 'total_time', 'avg_latency', 'avg_ttft', 'throughput'
    """
    if not requests:
        return {
            'total_time': 0,
            'avg_latency': 0.0,
            'avg_ttft': 0.0,
            'throughput': 0.0,
        }

    current_time = 0
    execute = []
    queue = []
    finished = []
    n = len(requests)

    while len(queue) != 0 or len(execute) != 0 or len(requests) != 0:
        check_requests(requests, execute, queue, current_time, max_batch_size)
        step(requests, execute, queue, finished, current_time, max_batch_size)

        
        if len(queue) == 0 and len(execute) == 0 and len(requests) > 0:
            current_time = min(r['arrival_time'] for r in requests)
        else:
            current_time += 1

    total_latency = 0
    total_ttft = 0
    total_tokens = 0

    for request in finished:
        total_latency += request['completion_time'] - request['arrival_time']
        total_ttft += request['start_time'] - request['arrival_time']
        total_tokens += request['total_tokens']

    return {
        'total_time': current_time,
        'avg_latency': round(total_latency / n, 4),
        'avg_ttft': round(total_ttft / n, 4),
        'throughput': round(total_tokens / current_time, 4),
    }