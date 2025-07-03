from operations.loss import MSELoss, KLDiv, BCE
from operations.tanh import Tanh
from operations.relu import ReLU
from operations.sigmoid import Sigmoid
from optim import GD, SGD, ADAM
from tensor import Tensor
from module import AutoEncoder
from keras.datasets import mnist
from matplotlib import pyplot as plt
import time
import numpy as np

def get_learning_rate(epoch, total_epochs, initial_lr=0.01, final_lr=0.001):
    """Linear learning rate decay from initial_lr to final_lr"""
    return initial_lr - (initial_lr - final_lr) * (epoch / total_epochs)

# constants
INPUT_DIM =  28 * 28

# hyper parameters
PENALTY = True 
SAMPLE_COUNT = 60000
EPOCHS = 50
HIDDEN_DIM = 64
INITIAL_LR = 0.01
FINAL_LR = 0.001
BATCH_SIZE = 32  

# training analytics
SHOW_GRAD = False
TRACK_TIME = True





model = AutoEncoder(INPUT_DIM, HIDDEN_DIM)
optimizer = ADAM(model.parameters(), lr=INITIAL_LR)


sparsity = 0.10 # desired average activation
beta = 0.1       # weight for sparsity penalty



(train, _ ), (test, _) = mnist.load_data()
train = train.astype('float32') / 255.0
test = test.astype("float32") / 255.0

if TRACK_TIME:
    start = time.time()

losses = []

def gen(x): 
    encoded = model.encoder(x)
    h_out = ReLU.apply(encoded)
    decoded = model.decoder(h_out)
    output = Sigmoid.apply(decoded)
    return h_out, output

def get_batch(data, batch_idx, batch_size):
    """Get a batch of data"""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(data))
    return data[start_idx:end_idx]

for epoch in range(EPOCHS):
    # Update learning rate
    current_lr = get_learning_rate(epoch, EPOCHS, INITIAL_LR, FINAL_LR)
    optimizer.lr = current_lr
    
    epoch_loss = 0
    num_batches = (SAMPLE_COUNT + BATCH_SIZE - 1) // BATCH_SIZE  
    
    for batch_idx in range(num_batches):
        batch_data = get_batch(train, batch_idx, BATCH_SIZE)
        batch_loss = 0
        
        for x_data in batch_data:
            x = Tensor(x_data.flatten())
            z, output = gen(x)
            

            recon_loss = BCE.apply(x, output)
            
            total_loss = recon_loss
            if PENALTY:
                kl_loss = KLDiv.apply(z, sparsity)
                total_loss += Tensor(beta) * kl_loss
            
            total_loss.backward()
            batch_loss += total_loss.data
        
        optimizer.step()
        model.zero_grad()
        
        if SHOW_GRAD:
            print("\nPost-step gradients:")
            for name, param in model.named_parameters().items():
                if param.grad is not None:
                    print(f"{name}: mean={np.mean(param.grad):.6f}, std={np.std(param.grad):.6f}")
        
        epoch_loss += batch_loss / len(batch_data)
    
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")


if TRACK_TIME:
    end = time.time()
    print("Time elapsed: ", end - start)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.show()

plt.figure(figsize=(10, 4))
for i in range(5):  
    # Original image
    plt.subplot(2, 5, i + 1)
    plt.imshow(test[i], cmap='gray') 
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(2, 5, i + 6)
    x = Tensor(test[i].flatten())
    z, output = gen(x)
    plt.imshow(output.data.reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.show()






