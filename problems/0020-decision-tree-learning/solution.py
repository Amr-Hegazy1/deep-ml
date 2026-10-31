import math
from collections import Counter
import numpy as np

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    counts = np.zeros(len(unique_labels))
    for i, label in enumerate(unique_labels):
        count = (labels == label).sum()
        counts[i] = count

    probabilites = counts / len(labels)

    return -1 * (probabilites * np.log2(probabilites)).sum()

def get_potential_thresholds(feature: np.ndarray) -> np.ndarray:
    return np.unique(feature)

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    labels = [example[target_attr] for example in examples]
    base_entropy = calculate_entropy(labels)
    labels = np.array(labels)
    feature = np.array([example[attr] for example in examples])

    thresholds = get_potential_thresholds(feature)

    n = len(examples)
    entropy = 0

    for threshold in thresholds:
        split_feature = feature[feature == threshold]
        split_labels = labels[feature == threshold]

        m = len(split_feature)

        entropy += m / n * calculate_entropy(split_labels)

    return base_entropy - entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    
    labels = [example[target_attr] for example in examples]

    counts = Counter(labels)

    max_count = max(counts.values())

    tied_labels = [
        label
        for label, count in counts.items()
        if count == max_count
    ]

    return sorted(tied_labels)[0]

def get_highest_information_gain(examples: list[dict], attributes: str, target_attr: str) -> tuple[str, float]:

    best_attr = None
    best_information_gain = -np.inf

    for attr in attributes:
        information_gain = calculate_information_gain(examples, attr, target_attr)

        if information_gain > best_information_gain:
            best_attr = attr
            best_information_gain = information_gain

    return best_attr, best_information_gain


def build_tree(
    examples: list[dict],
    attributes: list[str],
    target_attr: str,
    node: dict
):

    
    labels = [example[target_attr] for example in examples]

    if len(set(labels)) == 1:
        return labels[0]

    if len(attributes) == 0:
        return majority_class(examples, target_attr)

    
    best_feature, _ = get_highest_information_gain(
        examples,
        attributes,
        target_attr
    )

    node[best_feature] = {}

    
    feature_values = np.unique([
        example[best_feature]
        for example in examples
    ])

    
    remaining_attributes = [
        attr
        for attr in attributes
        if attr != best_feature
    ]

    
    for value in feature_values:

        subset = [
            example
            for example in examples
            if example[best_feature] == value
        ]

        node[best_feature][value] = build_tree(
            subset,
            remaining_attributes,
            target_attr,
            {}
        )

    return node




    

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""



    return build_tree(examples, attributes, target_attr, {})
    
    