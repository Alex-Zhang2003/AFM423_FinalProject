import torch

# Configuration constants for training
DATA_DIR = r'./BenchmarkDatasets'
EPOCHS = 15
LR = 0.0001
BATCH_SIZE = 128
NUM_WORKER = 4

# Save model weights
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model weights into a given model class
def load_model(model_class, path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    return model