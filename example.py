# CS20B031 PUPPALA VISHNU VARDHAN
# FEB-15 ISL ASSIGNMENT TASK-2
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

label_mapping = {
    0: 0,  # T-shirt/Top -> Upper
    1: 1,  # Trouser -> Lower
    2: 1,  # Pullover -> Lower
    3: 0,  # Dress -> Upper
    4: 0,  # Coat -> Upper
    5: 2,  # Sandal -> Feet
    6: 0,  # Shirt -> Upper
    7: 2,  # Sneaker -> Feet
    8: 3,  # Bag -> Bag
    9: 2,  # Ankle boot -> Feet
}

train_data = datasets.FashionMNIST(
    root='data',
    download=True,
    train=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

for X,y in train_data:
  print(y)
  break

new_train_data = [(X,label_mapping[i]) for X,i in train_data]
new_test_data = [(X,label_mapping[i]) for X,i in test_data]

for X,y in new_train_data:
  print(y)
  break

batchsize = 64
train_dataloader = DataLoader(new_train_data,batch_size=batchsize)
test_dataloader = DataLoader(new_test_data,batch_size=batchsize)

for X, y in train_dataloader:
  print(y)
  break

for X, y in test_dataloader:
  print(y)
  break

for X,y in test_dataloader:
  print(f"X shape {X.shape}")
  print(f"y shape {y.shape}")
  break

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork,self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 20),
            nn.ReLU(),
            nn.Linear(20,4)
        )
  def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

model = NeuralNetwork().to(device)

print(model)

model.parameters()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

classes = ["Upper", "Lower", "Feet", "Bag"]

model.eval()
x, y = new_test_data[0][0], new_test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')