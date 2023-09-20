from typing import List, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def gradient_descent(X: np.array, y: np.array, 
                     lr: float = 0.01, 
                     epoch: int = 1000) -> Tuple[
                                            float, 
                                            float, 
                                            List[Tuple[float, float]],
                                            List[float]]:
    """
    Gradient Descent for a single feature
    """

    m, b = 0.33, 0.48  # initial guess for parameters
    log, mse = [], []  # lists to store learning process
    N = len(X)         # number of samples
    
    for _ in range(epoch):
                
        f = y - (m*X + b)

        # Updating m and b
        m -= lr * (-2 * X.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        
        log.append((m, b))
        mse.append(mean_squared_error(y, (m*X + b)))        
    
    return m, b, log, mse


if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    ## Hyperparameter tuning
    lr = 0.05   # 0.001 0.01 0.1
    epoch = 10   # 10 1000 5000
         
    m, b, log, mse = gradient_descent(x, y, lr, epoch)
    print(f"Params:\n\t{m = }\n\t{b = }")

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(x, m*x+b, "r--.", label="Gradient Descent")
    ax.legend(loc="best")

    plt.savefig("gd_regression.png")


    # Plot MSE value on epoch number
    plt.figure()
    plt.plot(range(len(mse)), mse, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('MSE value on epoch number')
    plt.savefig("MSE value on epoch number.png")

    plt.show()