import argparse
import torch
import torch.nn as nn
from BiDAF import BiDAF
from data_process import *

def train(config,train_dataloader):
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model=BiDAF(config)

	optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
	optimizer.zero_grad()
	criterion=nn.CrossEntropyLoss()

	model.train()
	loss=0
	for epoch in range(config.epochs):

		for step,batch_data in enumerate(train_dataloader):
			start_positions,end_positions=batch_data[-2],batch_data[-1]
			start_logits,end_logits=model(batch_data)
			loss=criterion(start_logits,start_positions)+criterion(end_logits,end_positions)
			print("loss : ",loss)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

