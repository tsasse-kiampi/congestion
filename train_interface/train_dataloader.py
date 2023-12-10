import os

import torch
import pandas as pd
from parameters import DEFAULT_DATA_DIR, BATCH_SIZE
from torch.utils.data import DataLoader, TensorDataset

NUM_WORKERS = os.cpu_count()


def create_congestion_dataset(data_dir=DEFAULT_DATA_DIR):
    """I assume that congestion, target congestion (n+1 congestion) is already calculated in a csv table
    """

    df = pd.DataFrame(data_dir)

    return TensorDataset(torch.tensor(df['congestion'].values),
                          torch.tensor(df['target_congestion'].values),
                          )
  

def create_dataloaders(
    train_data_dir: str, 
    test_data_dir: str, 
    batch_size: int=BATCH_SIZE, 
    num_workers: int=NUM_WORKERS
):
  

  train_data = create_congestion_dataset(train_data_dir)
  test_data = create_congestion_dataset(test_data_dir)


  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader