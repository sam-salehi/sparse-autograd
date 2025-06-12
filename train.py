from operations.loss import MSELoss, KLDiv
from operations.sigmoid import Sigmoid
from optim import GD, SGD
from tensor import Tensor
from module import AutoEncoder
from keras.datasets import mnist
from matplotlib import pyplot as plt



import numpy as np


# Example setup
input_dim = 28 * 28
show_sample = False
n_inputs = 5000

model = AutoEncoder(input_dim, 64, Sigmoid)
optimizer = SGD(model.parameters(), lr=0.01)
sparsity = 0.05  # desired average activation
beta = 0.1       # weight for sparsity penalty




(train, _ ), (test, _) = mnist.load_data()
train = train[0:n_inputs]
test = test[0:n_inputs]


if show_sample:
    print(train.shape)
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(train[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()



losses = []

for j in range(10):
    # Suppose batch_x is your input batch, shape (batch_size, input_dim)
    x = Tensor(train[j].flatten())

    # Forward pass
    encoded = model.encoder(x)         # hidden activations
    sig = model.activation.apply(encoded)
    decoded = model.decoder(sig)         # reconstruction

    # Compute losses
    recon_loss = MSELoss.apply(x, decoded)  # reconstruction loss
    kl_loss = KLDiv.apply(sig, sparsity)      # sparsity penalty

    print(type(kl_loss.data))
    print(kl_loss.data)
    # Total loss (scalar)
    total_loss = recon_loss + beta * kl_loss.data

    # Backward pass
    total_loss.backward()

    # Update parameters
    optimizer.step()
    model.zero_grad()

    ''' Comments about training pass '''
    print("-------------")
    print("sample #:", j+1)
    print("total loss: ", total_loss)
    print("-------------")
    losses.append(total_loss)



plt.plot(total_loss)
plt.show()


