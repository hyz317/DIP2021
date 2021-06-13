import torch

BASE_PATH = '..'
DATA_PATH = f'{BASE_PATH}/data'
MODEL_PATH = f'{BASE_PATH}/models'
LOG_PATH = f'{BASE_PATH}/logs'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
