import torch
import torch.nn.functional as F
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    ix3 = stoi[ch3]
    xs.append((ix1, ix2))
    ys.append(ix3)
xs = torch.tensor(xs)

print(f"Shape of xs is {xs.shape}")
ys = torch.tensor(ys)
num = xs.shape[0]
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((54, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(500):
  
  # forward pass
  xenc = torch.cat([
      F.one_hot(xs[:,0], num_classes=27),
      F.one_hot(xs[:,1], num_classes=27)
  ], dim=1).float()  # shape (num, 54)

  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  if k % 100 == 0:
    print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad


# Inference

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix1, ix2 = 0, 0 
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = torch.cat([F.one_hot(torch.tensor([ix1]), num_classes = 27), F.one_hot(torch.tensor([ix2]), num_classes = 27)], dim = 1).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix3 = torch.multinomial(p[0], num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix3])
    if ix3 == 0:
      break
    ix1, ix2 = ix2, ix3
  print(''.join(out))