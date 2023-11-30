import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

train_data = [0, 4, 2, 2, 3, 0, 4, 5, 6, 6, 4, 1, 4, 4, 0, 1, 3, 4, 2, 0, 3, 2, 0, 1]
validation_data = [0, 1, 3, 2, 3, 1, 4, 5, 3, 1, 4, 3, 5, 5, 2, 1, 1, 1, 2, 0, 1, 2, 1, 0]
rho_values = [0.1, 1, 10, 100]

def main():
    rates = cp.Variable(24)
    rho = cp.Parameter(nonneg=True)
    objective = cp.Minimize(cp.sum_squares(rates[1:] - rates[:-1]) + cp.square(rates[0] - rates[23]) + \
                rho * (cp.sum(rates) - cp.sum(train_data * cp.log(rates))))
    constraints = [rates >= 0]
    prob = cp.Problem(objective, constraints)
    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    validation_losses = []
    for rho_value, ax in zip(rho_values, axes.flatten()):
        rho.value = rho_value
        prob.solve()
        ax.plot(rates.value)
        ax.set_title(f'{rho.value=}')
        validation_losses.append(rates.value.sum() - (validation_data * np.log(rates.value)).sum())
    plt.show()
    plt.savefig('result.png')
    print(validation_losses)
    
    
    
if __name__=='__main__':
    main()