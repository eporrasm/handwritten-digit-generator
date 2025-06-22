import torch
from torch import nn

# Define the same Generator class you used for training
class Generator(nn.Module):
    def __init__(self, z_dim=100, label_dim=10, img_dim=28 * 28):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + label_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedded = self.label_emb(labels)
        x = torch.cat([z, embedded], dim=1)
        img = self.net(x)
        return img.view(-1, 1, 28, 28)


def load_generator(path: str, device=torch.device("cpu")) -> Generator:
    """
    Load the trained generator weights from disk and return the model in eval mode.
    """
    G = Generator().to(device)
    state = torch.load(path, map_location=device)
    G.load_state_dict(state)
    G.eval()
    return G