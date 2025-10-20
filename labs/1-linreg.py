import torch
from matplotlib import pyplot as plt

# Define the model
def model(x, w, b):
  return w*x + b

# Define the loss function
def mse_loss(y_pred, y_ref):
  return ((y_pred - y_ref)**2).mean()

def main():
  # Synthetic reference data
  x = torch.linspace(0, 1, 100).reshape(-1, 1)  # -1 is equal to omitting the paramter (automaticcaly computed by pytorch)
  y = 3 * x + 2 + 0.1 * torch.randn(x.size())  # Evaluation of fake model plus some noise
  
  # Random init of theta
  w = torch.randn(1, requires_grad=True)
  b = torch.randn(1, requires_grad=True)

  # This can be avoided with the package torch.optim
  learning_rate = .1
  for _ in range(100):  # For each epochs
    predictions = model(x, w, b)
    loss = mse_loss(predictions, y)

    loss.backward()

    with torch.no_grad():
      w -= learning_rate * w.grad
      b -= learning_rate * b.grad

      w.grad.zero_()
      b.grad.zero_()

  with torch.no_grad():
    plt.scatter(x.numpy(), y.numpy(), label='Data', color='blue')
    plt.plot(x.numpy(), predictions.numpy(), label='Fitted line', color='red')
    plt.legend()
    plt.show()

if __name__ == '__main__':
  main()