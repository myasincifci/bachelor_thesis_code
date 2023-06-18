import numpy as np
import scipy.stats as stats

def p_uni(i: int, tau: int, I: int):
    '''
    pdf for sampling from a uniform distribution
    '''

    x = np.arange(I)
    m = ((x != i) & (x <= i + tau) & (x >= i - tau)).astype(int)
    p = 1 / (min(tau, i) + min(tau, I - i - 1))

    return m * p

def p_nml(i: int, sigma: float, I):
    '''
    pdf for sampling from a normal distribution
    '''
    x = np.arange(I)
    pdf = stats.norm.pdf(x, i, sigma)
    pdf[i] = 0.0
    p = pdf / pdf.sum()

    return p

# Index for registered pdf functions. 
# Newly added pdfs must be registered here.
pdf_index = {
    "uniform": p_uni,
    "normal": p_nml,   
}