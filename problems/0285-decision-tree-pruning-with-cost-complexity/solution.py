def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.
    
    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None
        
    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """
    alphas = []

    solve(tree, alphas)
    


    return sorted(alphas)

def solve(tree: dict, alphas: list):

    errors_left = 0
    errors_right = 0

    num_nodes_left = 0
    num_nodes_right = 0

    if tree['left'] != None:
        solve(tree['left'], alphas)
        errors_left = tree['left']['R_T']
        num_nodes_left = tree['left']['num_nodes']
        

    if tree['right'] != None:
        solve(tree['right'], alphas)
        errors_right = tree['right']['R_T']
        num_nodes_right = tree['right']['num_nodes']
        

    if tree['right'] == None and tree['left'] == None:
        
        R_T = tree['errors']
        num_nodes = 1
    else:
        R_T = errors_right + errors_left
        num_nodes = num_nodes_left + num_nodes_right


    tree['R_T'] = R_T
    tree['num_nodes'] = num_nodes
    

    if tree['right'] != None or tree['left'] != None:
        alpha = (tree['errors'] - R_T) / (num_nodes - 1)
        

        alphas.append(alpha)

