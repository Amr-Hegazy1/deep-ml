import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """

    f_coeffs = np.array(f_coeffs)
    g_coeffs = np.array(g_coeffs)

    f_dash = np.array([f_coeff * i for i, f_coeff in enumerate(f_coeffs)])
    g_dash = np.array([g_coeff * i for i, g_coeff in enumerate(g_coeffs)])

    

    part1 = np.convolve(f_dash, g_coeffs, mode="full")
    part2 = np.convolve(f_coeffs, g_dash, mode="full")

    result = part1 + part2

    return result[1:] if len(result) > 1 else result

