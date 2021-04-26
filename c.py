SAVE_EVERY = 5
SEQ_SIZE = 50
RANDOM_SEED = 11
TRAINING_SIZE = 0.75
VALIDATION_SIZE = 0.15
TESTING_SIZE = 0.10
LR = 1e-3
N_EPOCHS = 10
NUM_LAYERS, HIDDEN_SIZE = 1, 128
DROPOUT_P = 0
model_type = 'lstm'
CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(model_type, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)