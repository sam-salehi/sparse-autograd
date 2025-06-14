from operations.loss import MSELoss, KLDiv, BCE
from operations.tanh import Tanh
from operations.sigmoid import Sigmoid
from optim import GD, SGD
from tensor import Tensor
from module import BigAutoEncoder
from keras.datasets import mnist
from matplotlib import pyplot as plt

# virtual phyical environment for exploration of  gradients where you initalize 
# with a loss sace,

import numpy as np

 # TODO keep track of each parameters gradients over training 

def get_learning_rate(epoch, total_epochs, initial_lr=0.01, final_lr=0.001):
    """Linear learning rate decay from initial_lr to final_lr"""
    return initial_lr - (initial_lr - final_lr) * (epoch / total_epochs)

# hyper params
PENALTY = True
INPUT_DIM =  28 * 28
SHOW_GRAD = False
show_sample = False
SAMPLE_COUNT = 1000
EPOCHS = 20
H_DIM = 256
H2_DIM = 64
Z_DIM = 32
HIDDEN_DIM = 32
INITIAL_LR = 0.01
FINAL_LR = 0.01

model = BigAutoEncoder(INPUT_DIM, H_DIM,H2_DIM,Z_DIM)
optimizer = SGD(model.parameters(), lr=INITIAL_LR)
# TODO train with batches and gradient descent instead

sparsity = 0.10 # desired average activation
beta = 0.1       # weight for sparsity penalty

# should probably just add more none linearity.
# do multiple hidden layers instead  of one.


(train, _ ), (test, _) = mnist.load_data()
train = train.astype('float32') / 255.0
test = test.astype("float32") / 255.0


if show_sample:
    print(train.shape)
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train[i], cmap=plt.get_cmap('gray'))
    plt.show()

losses = []

def gen(x): 
    encoded = model.encoder(x)
    h_out = Sigmoid.apply(encoded)
    decoded = model.decoder(h_out)
    output = Sigmoid.apply(decoded)
    return h_out, output


for epoch in range(EPOCHS):
    # Update learning rate
    current_lr = get_learning_rate(epoch, EPOCHS, INITIAL_LR, FINAL_LR)
    optimizer.lr = current_lr
    
    epoch_loss = 0
    for i in range(SAMPLE_COUNT):
        x = Tensor(train[i].flatten())
        
        z, output = gen(x)
        # Compute losses
        recon_loss = BCE.apply(x, output)
        
        total_loss = recon_loss
        if PENALTY:
            kl_loss = KLDiv.apply(z, sparsity)
            total_loss += Tensor(beta) * kl_loss
        
        total_loss.backward()
        optimizer.step()

        if SHOW_GRAD:
            print("\nPost-step gradients:")
            print(f"Input grad mean: {np.mean(x.grad)}")
            print(len(model.named_parameters().items()))
            for name, param in model.named_parameters().items():
                if param.grad is not None:
                    print(f"{name}: mean={np.mean(param.grad):.6f}, std={np.std(param.grad):.6f}")
        
        
        model.zero_grad()
        epoch_loss += total_loss.data
        mean_activation = np.mean(z.data)

    avg_loss = epoch_loss / SAMPLE_COUNT 
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}, Avg Hidden Activation: {mean_activation:.4f}")


plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.show()


plt.figure(figsize=(10, 4))
for i in range(5):  # Show 5 examples
    # Original image
    plt.subplot(2, 5, i + 1)
    plt.imshow(train[i], cmap='gray') # change train to test.
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(2, 5, i + 6)
    x = Tensor(train[i].flatten())
    z, output = gen(x)
    plt.imshow(output.data.reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.show()




