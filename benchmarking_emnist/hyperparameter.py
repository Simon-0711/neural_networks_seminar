# Common hyperparameters
common_hyperparameters = {
    "NUM_CLASSES": 11,
    "BLANK_LABEL": 10,
    "IMAGE_HEIGHT": 28,
    "DIGITS_PER_SEQUENCE": 5,
    "NUMBER_OF_SEQUENCES": 10000,
    # "NUMBER_OF_SEQUENCES": 1000,
    "EPOCHS": 10,
    # "EPOCHS": 1,
    "WEIGHT_DECAY": 0.0001,
}

# Model-specific hyperparameters
model_specific_hyperparameters = {
    "GRU_HIDDEN_SIZE": [128, 256],
    "GRU_NUM_LAYERS": [2],
    "CNN_OUTPUT_HEIGHT": [4],
    "CNN_OUTPUT_WIDTH": [32],
    "LEARNING_RATE": [1e-2, 1e-3, 1e-4],
    "BATCH_SIZE": [64, 128],
    "BATCH_SIZE_VAL": [64],
}
