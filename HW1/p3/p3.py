from zero_crossings_data import *
import matplotlib.pyplot as plt
import cvxpy as cp

def main():
    time_shots = np.arange(1, n + 1)
    frequencies = np.arange(f_min, f_min + B)
    angles = 2 * np.pi / n * time_shots[:, np.newaxis] * frequencies[np.newaxis, :]
    features = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    coefficients = cp.Variable(2 * B)
    outputs = features @ coefficients
    constraints = [s * outputs >= 0] + [cp.sum(s * outputs) == n]
    objective = cp.Minimize(cp.norm(outputs, 2))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    y_hat = features @ coefficients.value
    plt.plot(time_shots, y, label='actuals')
    plt.plot(time_shots, y_hat, label='predictions')
    plt.legend()
    plt.show()
    plt.savefig('result.png')
    print(np.sqrt(((y - y_hat) ** 2).sum() / (y ** 2).sum()))
    



if __name__=='__main__':
    main()