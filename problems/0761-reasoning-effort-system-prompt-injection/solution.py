def inject_reasoning_effort(messages, effort_level):
    """
    messages: list of dicts with keys 'role' and 'content'
    effort_level: one of 'low', 'medium', 'high', 'max'
    Returns: new list of message dicts with reasoning-effort system prompt injected.
    """

    prompt = ""

    if effort_level == "low":
        prompt = "Reasoning effort: low. Provide concise answers with minimal deliberation."
    elif effort_level == "medium":
        prompt = "Reasoning effort: medium. Think step by step before answering."
    elif effort_level == "high":
        prompt = "Reasoning effort: high. Carefully decompose the problem and verify each step."
    elif effort_level == "max":
        prompt = "Reasoning effort: max. Exhaustively decompose the problem, stress-test all edge cases, and document every intermediate step."
    else:
        raise ValueError(f"Please select a correct effort level from loww, medium, high and max. Got: {effort_level}")

    if len(messages) == 0 or messages[0]['role'] != "system":
        messages = [{"role": "system", "content": ""}] + messages

    messages[0]['content'] = (f"{prompt}" if len(messages[0]['content']) == 0 else f"{prompt}\n") + messages[0]['content']

    return messages
