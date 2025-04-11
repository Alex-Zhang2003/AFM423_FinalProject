import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = torch.load(path, weights_only=True)
    model.eval()
    return model