

## Training:

PENALTY = TRUE
INPUT_DIM =  28 * 28
SHOW_GRAD = False  training regimes
show_sample = False
SAMPLE_COUNT = 100
EPOCHS = 1000
HIDDEN_DIM = 64
LEARNING_RATE = 0.01

end loss = 0.1342

PENALTY = TRUE
INPUT_DIM =  28 * 28
SHOW_GRAD = False
show_sample = False
SAMPLE_COUNT = 100
EPOCHS = 1000
HIDDEN_DIM = 32
LEARNING_RATE = 0.01

loss =0.1368

PENALTY = TRUE
INPUT_DIM =  28 * 28
SHOW_GRAD = False
show_sample = False
SAMPLE_COUNT = 100
EPOCHS = 1000
H_DIM = 256
H2_DIM = 64
Z_DIM = 32
HIDDEN_DIM = 32
LEARNING_RATE = 0.01

loss = 0.0629 # train on more data i.e 100  less batches 


PENALTY = TRUE
INPUT_DIM =  28 * 28
SHOW_GRAD = False
show_sample = False
SAMPLE_COUNT = 100
EPOCHS = 500
H_DIM = 256
H2_DIM = 64
Z_DIM = 32
HIDDEN_DIM = 32
LEARNING_RATE = 0.001

loss = 0.2173




PENALTY = False
INPUT_DIM =  28 * 28
SHOW_GRAD = False
show_sample = False
SAMPLE_COUNT = 100
EPOCHS = 500
H_DIM = 256
H2_DIM = 64
Z_DIM = 32
HIDDEN_DIM = 32
LEARNING_RATE = 0.01

loss = 0.0630


Changed loss function to BCE: 
(loss between them isn't necessarily comparable since we have different eval method)

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
FINAL_LR = 0.001




Changed loss function to BCE: 
(loss between them isn't necessarily comparable since we have different eval method)

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
FINAL_LR = 0.001

loss = 0.2716



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
INITIAL_LR = 0.01 (constant lr)
FINAL_LR = 0.01
sparsity = 0.05 # desired average activation
beta = 0.1       # weight for sparsity penalty

loss = 0.2616


# changed spasity 
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
sparsity = 0.10 # desired average activation
beta = 0.1       # weight for sparsity penalty


