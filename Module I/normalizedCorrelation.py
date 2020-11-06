import numpy as np

from scipy import signal
from sklearn.preprocessing import normalize


m1 = np.asmatrix(np.array([[0, 2, 1, 2],
                           [1, 4, 2, 0],
                           [3, 3, 0, 1]]))

m2 = np.asmatrix(np.array([[2, 1, 2],
                           [4, 2, 0],
                           [3, 0, 1]]))

#array([[  0.,   3.,   6.],
#   [  9.,  12.,  15.],
#   [ 18.,  21.,  24.]])

m1_normalized = normalize(m1, norm='l2')
m2_normalized = normalize(m2, norm='l2')

#[[ 0.          0.33333333  0.66666667]
#[ 0.25        0.33333333  0.41666667]
#[ 0.28571429  0.33333333  0.38095238]]


corr = signal.correlate2d(m1_normalized, m2_normalized, boundary='fill', mode='same')

print(corr)




'''
# Iterando sobre a transposta da matriz obtem-se linha a linha dela:
m1_columns = m1.T
m2_columns = m2.T

for col_index in len(m1_columns):
    m1_minus_u1 = np.subtract(m1, u1)
    # linalg.norm utiliza o módulo de Frobenius
    r1 = m1_minus_u1 / np.linalg.norm(m1_minus_u1)

    u2 = np.full(m1.shape, m1.mean())
    m2_minus_u2 = np.subtract(m2, u2)
    # linalg.norm utiliza o módulo de Frobenius
    r2 = m2_minus_u2 / np.linalg.norm(m2_minus_u2)
    # Inner Product entre r1 e r2
    r = np.inner(r1, r2)
u1 = np.full(m1.shape, m1.mean())
m1_minus_u1 = np.subtract(m1, u1)
# linalg.norm utiliza o módulo de Frobenius
r1 = m1_minus_u1 / np.linalg.norm(m1_minus_u1)

u2 = np.full(m1.shape, m1.mean())
m2_minus_u2 = np.subtract(m2, u2)
# linalg.norm utiliza o módulo de Frobenius
r2 = m2_minus_u2 / np.linalg.norm(m2_minus_u2)
# Inner Product entre r1 e r2
r = np.inner(r1, r2)'''
