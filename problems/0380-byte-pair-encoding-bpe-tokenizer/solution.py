def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    merges = []

    for _ in range(num_merges):
        pairs = {}
        for word, freq in corpus.items():
            symbols = word.split()
            for s1, s2 in zip(symbols, symbols[1:]):
                pair = (s1, s2)
                pairs[pair] = pairs.get(pair, 0) + freq

        if not pairs:
            break

        most_freq_pair = max(pairs, key=pairs.get)
        merges.append(most_freq_pair)

        new_corpus = {}
        pair_str = f"{most_freq_pair[0]} {most_freq_pair[1]}"
        replacement_str = f"{most_freq_pair[0]}{most_freq_pair[1]}"

        for word, freq in corpus.items():
            new_word = word.replace(pair_str, replacement_str)
            new_corpus[new_word] = freq
            
        corpus = new_corpus

    return merges
