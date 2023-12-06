from model import CongestionModel
from train_dataloader import train_dataloader, test_dataloader
from parameters import DEVICE

import torch, train_interface



model = CongestionModel()

optimizer = torch.optim.Adam(params=model.parameters(), 
                             lr=3e-3, # Base LR better search in papers? 
                             betas=(0.9, 0.999), 
                             weight_decay=0.3) 


loss_fn = torch.nn.CrossEntropyLoss()


results = train_interface.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=10,
                       device=DEVICE)