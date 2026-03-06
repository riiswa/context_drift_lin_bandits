import numpy as np
import matplotlib.pyplot as plt
from arms_generator import sample_spherical
from arms_generator import sample_random
import numpy.linalg as la
from line_profiler import profile


def init():
    return 0

@profile
def volume_approx(X, i_init_arms=[], opt="default"):
    """
    initial volume estimation
    accepts n by d matrix
    i_init_arms: initial arm indices; assumed to be linearly independent.
    opt="default": pulls 2 arms at a time; total 2d
    opt="economy": pulls one arm at a time: total d points.
    """
    n, d = X.shape
    if (opt == "default"):
        if (n <= 2 * d):
            return np.arange(n)
    elif (opt == "economy"):
        if (n <= d - len(i_init_arms)):
            return np.arange(n)
    else:
        raise ValueError()

    I = np.eye(d)
    leng = np.sqrt((X ** 2).sum(1))
    accept = []
    basis = []
    Vperp = np.zeros((d, d))

    # --- use init arms
    n_init_arms = len(i_init_arms)
    for i in range(n_init_arms):
        x = X[i_init_arms[i], :]
        if (i != 0):
            vperp = basis[i - 1] - Vperp @ basis[i - 1]
            tmp = vperp / la.norm(vperp)
            Vperp += np.outer(tmp, tmp)
        accept.append(i_init_arms[i])
        basis.append(x / la.norm(x))

    for i in range(n_init_arms, d):
        if (i == 0):
            b_i = I[0]
        else:
            vperp = basis[i - 1] - Vperp @ basis[i - 1]
            tmp = vperp / la.norm(vperp)
            Vperp += np.outer(tmp, tmp)
            b_i = I[i] - Vperp @ I[i]

        v = X @ b_i
        if opt == "default":
            i_alpha = np.argmax(v)
            i_beta = np.argmin(v)
            accept += [i_alpha, i_beta]
            direction = X[i_alpha] - X[i_beta]
        elif opt == "economy":
            i_alpha = np.argmax(np.abs(v))
            accept += [i_alpha]
            direction = X[i_alpha]

        basis.append(direction / la.norm(direction))
        i += 1

    return np.unique(accept)

def find_max_norm_arm(X,V_t_inv):
    max_lev_score = 0
    index_max_lev_score = 0
    n, d = X.shape
    for i in range(n):
        x = X[i, :]
        lev_score = np.matmul(np.matmul(x.T,V_t_inv),x)
        if (lev_score > max_lev_score):
            max_lev_score = lev_score
            index_max_lev_score = i
    return index_max_lev_score, max_lev_score
@profile
def KY_BH_Sampling(X):
    i_init_arms = []
    accept = volume_approx(X, i_init_arms, opt="default")
    return accept

@profile
def optimal_design_algo(X):
    val_lambda = 0
    n,d = X.shape
    I = np.eye(d)
    V_0 = I * val_lambda
    accept = KY_BH_Sampling(X)
    #print(accept)
    V_KY = V_0
    # print(V_KY)
    for i in range(len(accept)):
        x = X[accept[i], :]
        V_KY = V_KY + np.outer(x, x)
    V_t = V_KY
    # print(V_t)
    counter = 0
    #print(accept)
    #print(np.linalg.det(V_t))
    try:
        V_t_inv = np.linalg.inv(V_t)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            pass
            #print("Here")
        else:
            raise

    max_arm_ind, max_lev_score = find_max_norm_arm(X, V_t_inv)
    x = X[max_arm_ind, :]
    while max_lev_score > 1:
        V_t = V_t + np.outer(x, x)
        V_t_inv = np.linalg.inv(V_t)
        max_arm_ind, max_lev_score = find_max_norm_arm(X, V_t_inv)
        x = X[max_arm_ind, :]
        # print(max_lev_score)
        counter = counter + 1
        #print(max_arm_ind)
        accept = np.concatenate((accept,max_arm_ind), axis=None)
    return accept, counter

@profile
def optimal_probability(X, sVal_opt_design_arms):
    n,d = X.shape
    prob_dist = np.zeros((n,1))
    for i in range(n):
        prob_dist[i][0] = np.count_nonzero(sVal_opt_design_arms == i)
    prob_dist = prob_dist/len(sVal_opt_design_arms)
    return prob_dist

@profile
def calc_q_opt_design_old(A):
    sVal_opt_design_arms, sampling_time = optimal_design_algo(A)
    prob_dist = optimal_probability(A, sVal_opt_design_arms)
    return prob_dist

def calc_q_opt_design(A):
    threshold = 0.000001
    n, d = A.shape

    cov_A = np.cov(A, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_A)

    non_zero_indices = np.where(np.abs(eigenvalues) > threshold)[0]

    # --- guard: no variance in features, fall back to uniform ---
    if len(non_zero_indices) == 0:
        return np.ones((n, 1)) / n

    non_zero_eigenvalues    = eigenvalues[non_zero_indices]
    selected_eigenvectors   = eigenvectors[:, non_zero_indices]
    low_dimensional_vectors = np.real(np.dot(A, selected_eigenvectors))

    sVal_opt_design_arms, _ = optimal_design_algo(low_dimensional_vectors)

    # --- guard: optimal_design_algo returned empty ---
    if len(sVal_opt_design_arms) == 0:
        return np.ones((n, 1)) / n

    prob_dist = optimal_probability(low_dimensional_vectors, sVal_opt_design_arms)

    # --- guard: NaN in prob_dist (div-by-zero survived somehow) ---
    if np.any(np.isnan(prob_dist)):
        return np.ones((n, 1)) / n

    return prob_dist


def graphical_testing():
    trials = 50
    val_lambda = 0
    Dim = [2, 4, 8, 16, 32, 64]
    stopping_time = np.zeros(len(Dim))

    for j in range(len(Dim)):
        d = Dim[j]
        I = np.eye(d)
        V_0 = I * val_lambda
        n = 10000
        # i_init_arms = np.arange(n)
        # i_init_arms = []
        acc_counter = 0
        for k in range(trials):
            X = sample_random(n, d)
            # X = sample_spherical(n, d)
            # print(X.shape)
            # accept = volume_approx(X, i_init_arms, opt="default")
            # print(accept)
            accept, counter = optimal_design_algo(X)

            acc_counter = acc_counter + counter
        stopping_time[j] = acc_counter / trials
        print(stopping_time[j])

    f_x = stopping_time

    plt.plot(np.log(1 + f_x), np.log(Dim), label="Stopping time")

    # plt.fill_between(np.log(set_K), np.log(confident_low_Reg), np.log(confident_up_Reg_Expected), color='b', alpha=.1)
    # Naming the x-axis, y-axis and the whole graph

    plt.xlabel("Dimension")
    plt.ylabel("Stopping time")
    plt.title("Stopping time Vs Dimension")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()


#init()
#graphical_testing()