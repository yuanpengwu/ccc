import pytest
import numpy as np
from layers import MeanOnlyBatchNorm

def test_BN_gradient_check(X, dim, grad, epsilon=1e-7):
    bn1 = MeanOnlyBatchNorm(dim)
    bn2 = MeanOnlyBatchNorm(dim)
    bn1.beta += np.full((1, dim), epsilon)
    bn2.beta += np.full((1, dim), -epsilon)

    J_plus = bn1.forward(X)
    J_minus = bn2.forward(X)

    gradapprox = (J_plus - J_minus)/(2*epsilon)
    # check if gradapprox is close enough

    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    diff = numerator / denominator
    if diff < epsilon:
        print('The gradient is correct!')
    else:
        print('The gradient is wrong!')
    return diff
