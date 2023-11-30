from disks_data import *
import cvxpy as cp
import numpy as np

def main():
    c_rem = cp.Variable(shape=(n - k, 2))
    r_rem = cp.Variable(n - k, nonneg=True)
    get_c = lambda index: Cgiven[index] if index < k else c_rem[index - k]
    get_r = lambda index: Rgiven[index] if index < k else r_rem[index - k]
    soc_constraints = list()
    for first_circle_index, second_circle_index in Gindexes:
        soc_constraints.append(cp.SOC(get_r(first_circle_index) + get_r(second_circle_index), get_c(first_circle_index) - get_c(second_circle_index)))
    prob_l2 = cp.Problem(cp.Minimize(cp.sum(r_rem)), soc_constraints)
    prob_l2.solve()
    print(f'The optimal area is {np.pi * prob_l2.value ** 2}')
    plot_disks(np.concatenate([Cgiven, c_rem.value], axis=0), np.concatenate([Rgiven, r_rem.value], axis=0), Gindexes)
    prob_l1 = cp.Problem(cp.Minimize(cp.sum_squares(r_rem)), soc_constraints)
    prob_l1.solve()
    print(f'The optimal area is {2 * np.pi * prob_l1.value}')
    plot_disks(np.concatenate([Cgiven, c_rem.value], axis=0), np.concatenate([Rgiven, r_rem.value], axis=0), Gindexes)

if __name__=='__main__':
    main()