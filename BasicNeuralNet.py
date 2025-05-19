import torch
import torch.nn as nn
import torch.nn.functional as F

xs = torch.tensor([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
], dtype=torch.float32)
ys = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32).view(-1, 1)

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(3, 2)
    self.fc2 = nn.Linear(2, 2)
    self.fc3 = nn.Linear(2, 1)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


model = MLP()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for i in range(1000):
  y_pred = model(xs)
  loss = criterion(y_pred, ys)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if i % 100 == 0:
    print(f"Loss is : {loss.item():.4f}")

print("Final prediction ")
print(model(xs).detach())


