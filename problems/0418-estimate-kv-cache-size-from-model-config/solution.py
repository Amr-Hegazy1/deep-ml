def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    num_layers = model_config['num_layers']
    num_attention_heads = model_config['num_attention_heads']
    num_kv_heads = model_config['num_kv_heads'] if 'num_kv_heads' in model_config else num_attention_heads
    hidden_size = model_config['hidden_size']
    dtype_bytes = model_config['dtype_bytes']

    head_dim = hidden_size / num_attention_heads


    kv_cache_elements = int(num_layers * num_kv_heads * head_dim * batch_size * seq_len * 2)

    kv_cache_size_bytes = int(kv_cache_elements * dtype_bytes)

    kv_cache_size_mb = kv_cache_size_bytes / (1024.0 * 1024.0)

    per_layer_size_mb = kv_cache_size_mb / num_layers

    per_token_size_kb = kv_cache_size_mb / (batch_size * seq_len) * 1024

    return {
        "kv_cache_elements": kv_cache_elements,
        "kv_cache_size_bytes": kv_cache_size_bytes,
        "kv_cache_size_mb": round(kv_cache_size_mb, 4),
        "per_layer_size_mb": round(per_layer_size_mb, 4),
        "per_token_size_kb": round(per_token_size_kb, 4)
    }