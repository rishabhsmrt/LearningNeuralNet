
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

x_train = torch.linspace(-10, 10, 1000).unsqueeze(1)
y_train = x_train ** 3

class WithRelu(nn.Module):
  def __init__(self) -> None:
    super(WithRelu, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(1, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
  def forward(self, x):
      return self.model(x)

class WithoutRelu(nn.Module):
  def __init__(self) -> None:
    super(WithoutRelu, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(1, 50),
        nn.Linear(50, 50),
        nn.Linear(50, 50),
        nn.Linear(50, 1)
    )
  def forward(self, x):
      return self.model(x)

import matplotlib.pyplot as plt

def plot_graph(model, title):
  model.eval()
  x_plot = torch.linspace(-10, 10, 1000).unsqueeze(1)
  y_plot = x_train ** 3
  y_pred = model(x_plot) + 100

  x_plot = x_plot.detach().numpy()
  y_plot = y_plot.detach().numpy()
  y_pred = y_pred.detach().numpy()

  plt.plot(x_plot, y_plot, label = "True value", color="green")
  plt.plot(x_plot, y_pred, label = "Pred Value", color="red")
  plt.title(title)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.show()

def train_model(model, x, y, epochs=10000, lr=0.001):
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)

  for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
      print("loss is ", loss.item())

  return model

# Test Model with ReLU

model1 = WithRelu()
model1 = train_model(model1, x_train, y_train)

plot_graph(model1, "WITH RELU")

# Test Model without ReLU

model2 = WithoutRelu()
model2 = train_model(model2, x_train, y_train)

plot_graph(model2, "WITHOUT RELU")