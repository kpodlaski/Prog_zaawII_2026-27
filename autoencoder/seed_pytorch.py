import random
import os
import numpy as np
import torch


def set_pytorch_seed(seed: int = 42):
    """Ustawia ziarno losowości dla wszystkich bibliotek."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # dla konfiguracji multi-GPU

    # Kluczowe dla determinizmu operacji na GPU (może lekko spowolnić trening)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False