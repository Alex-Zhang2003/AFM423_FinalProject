import torch

DATA_DIR = r'./BenchmarkDatasets'
EPOCHS = 15
LR = 0.0001
BATCH_SIZE = 128
NUM_WORKER = 4


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model