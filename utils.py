import torch

DATA_DIR = r'./BenchmarkDatasets'
EPOCHS = 30
LR = 0.0001
BATCH_SIZE = 128
NUM_WORKER = 4


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = torch.load(path, weights_only=True)
    model.eval()
    return model