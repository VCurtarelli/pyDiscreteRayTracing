import numpy as np
from numpy.linalg import pinv, inv
np_precision = 4
np.set_printoptions(precision=np_precision,suppress=True,linewidth=255,formatter={'float': '{: 0.4f}'.format})
from scipy.stats import ortho_group


def cost(mat_J, vec_z, vec_t):
    temp = mat_J @ vec_z - vec_t
    return (temp.T @ temp).item()

def cost2(mat_D, vec_z):
    temp = mat_D @ vec_z
    return (temp.T @ temp).item()

def grad(mat_J, vec_z, vec_t):
    return 2*mat_J.T @ (mat_J @ vec_z - vec_t)

def grad2(mat_D, vec_z):
    return 2*mat_D.T @ mat_D @ vec_z


## Generation of J matrix
R_dim = 30
L_dim = 20
U = ortho_group.rvs(L_dim)
V = ortho_group.rvs(R_dim)

rank_J = int(min(R_dim,L_dim)*0.8)

s = np.random.rand(rank_J,1)**2 * 20
s = np.flipud(np.sort(s,axis=0))
# s[rank_J:] = 0
S = np.zeros([L_dim,R_dim])
for i in range(rank_J):
    S[i,i] = s[i].item()
s = np.diag(S)
J = U @ S @ V.T

rank_J_t = len(s[s >= 0.1*s[0]])  # _t -> _truncated
S_t = np.zeros_like(S)
for i in range(rank_J_t):
    S_t[i,i] = s[i].item()
J_t = U @ S_t @ V.T

V2 = V[:, rank_J:].reshape((R_dim, -1))
V2_t = V[:, rank_J_t:].reshape((R_dim, -1))
## Generation of D matrix and t vector
D = np.random.randn(R_dim, R_dim)* 3 + np.eye(R_dim)*0.5
t = np.linspace(0, 1, L_dim).reshape(-1, 1)
alpha = 0.1

## Calculating optimal vectors
z_min = pinv(J) @ t
z_lit = inv(2*J.T @ J + alpha**2 * D.T @ D) @ J.T @ t
z_prp = (np.eye(R_dim) - V2 @ pinv(D @ V2) @ D) @ pinv(J) @ t
z_tnc = (np.eye(R_dim) - V2_t @ pinv(D @ V2_t) @ D) @ pinv(J_t) @ t

## Metrics
print("Singular values:\t", s.T)
print()
print("Dim J:  {}x{}".format(L_dim, R_dim))
print("Rank J: ",  rank_J)
print("Rank Jt:", rank_J_t)
print()
print("Costs:\t\tPrimary:\tSecondary:")
print('\tMin:\t{:.2f}\t\t{:.2f}'.format(cost(J, z_min, t),cost2(D, z_min)))
print('\tLit:\t{:.2f}\t\t{:.2f}'.format(cost(J, z_lit, t),cost2(D, z_lit)))
print('\tPrp:\t{:.2f}\t\t{:.2f}'.format(cost(J, z_prp, t), cost2(D, z_prp)))
print('\tTnc:\t{:.2f}\t\t{:.2f}'.format(cost(J, z_tnc, t), cost2(D, z_tnc)))

print()
print("Gradient:")
print('\tMin:\t', grad(J, z_min, t).T)
print('\tLit:\t', grad(J, z_lit, t).T)
print('\tPrp:\t', grad(J, z_prp, t).T)
print('\tTnc:\t', grad(J, z_tnc, t).T)

# # print()
# # print("Second cost function:")
# # print('\tMin:\t', np.around(cost2(D, z_min), np_precision))
# # print('\tLit:\t', np.around(cost2(D, z_lit), np_precision))
# # print('\tPrp:\t', np.around(cost2(D, z_prp), np_precision))
#
# L_dim = 8
# R_dim = 5
# R = 3
# U = ortho_group.rvs(L_dim)
# I = np.eye(L_dim)
# Ir = np.zeros_like(I)
# for n in range(R):
#     Ir[n,n] = 1
# U_t = U[:, R:]
# t = np.logspace(0,1,L_dim)
#
# A = np.random.randn(L_dim, R_dim)
# U, s, Vh = np.linalg.svd(A)
# s[-2:] = 0
# S = np.zeros_like(A)
# for i in range(R_dim):
#     S[i,i] = s[i]
# V = Vh.T
# A = U @ S @ V.T
#
# V2 = V[:, -2:]
#
# mu = np.random.randn(2).reshape(-1, 1)
#
# print(A @ V2 @ mu)
#
