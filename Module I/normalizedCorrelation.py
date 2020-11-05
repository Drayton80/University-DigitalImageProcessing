import numpy as np


m1 = np.asmatrix(np.array([[1, 2],
                           [2, 0]]))

m2 = np.asmatrix(np.array([[1, 2],
                           [2, 0]]))

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

'''
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
