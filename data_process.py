import os,json
import numpy as np
import torch
from collections import Counter

def read_data(json_files,word_limit_freq=10):
	if type(json_files)!=list:
		json_files=[json_files]
	examples=[]
	all_words=[]
	unk_example_nums=0
	for json_file in json_files:
		with open(json_file) as f:
			lines=f.readlines()
		for line in lines:
			example=json.loads(line.strip())
			if example["answer"]["text"]=="null" or example["answer"]["start_position"]==-1:
				example["answer"]['end_position']=-1
				unk_example_nums+=1
			else:
				example["answer"]["end_position"]=len(str(example["answer"]["text"]))+example["answer"]["start_position"]
			context=example["context"]
			assert type(context)==str
			for word in context:
				all_words.append(word)
			all_words.append(char for char in example["context"])
			examples.append(example)
	counter_words=Counter(all_words)
	sorted_words=sorted(counter_words.items(),key=lambda x:x[1],reverse=True)
	word2id={"[PAD]":0,"[UNK]":1}
	for word,word_freq in sorted_words:
		if word_freq>word_limit_freq:
			word2id[word]=len(word2id)
	print("unknown question examples : %d , total examples : %d"%(unk_example_nums,len(examples)))
	return examples,word2id

def convert_to_ids(examples,word2id):
	features=[]
	def get_word_id(word):
		if word in word2id:
			return word2id[word]
		else:
			return word2id["[UNK]"]

	for example in examples:
		context=example["context"]
		question=example["question"]
		context_ids=[get_word_id(word) for word in context]
		question_ids=[get_word_id(word) for word in question]
		features.append({"context_ids":context_ids,"question_ids":question_ids,
						"start_position":example["answer"]["start_position"],
						"end_position":example["answer"]["end_position"]})
	return features

def convert_to_words(features,word2id):
	examples=[]
	id2word={id_:word for word,id_ in word2id.items()}
	if type(features)!=list:
		features=[features]
	for feature in features:
		context_ids=feature["context_ids"]
		question_ids=feature["question_ids"]
		start_position=feature["start_position"]
		end_position=feature["end_position"]
		context=""
		for id_ in context_ids:
			context+=id2word[id_]
		question=""
		for id_ in question_ids:
			question+=id2word[id_]
		answer=context[feature["start_position"]:feature["end_position"]]
		examples.append({"context":context,"question":question,
						"answer":{"text":answer,"start_position":feature["start_position"],"end_position":feature["end_position"]}})
	return examples


# class DataSet(torch.utils.data.DataSet):
# 	def __init__(self,examples,word2id,config,is_train=True):
# 		super(DataSet,self).__init__()
# 		self.context_len_limit=config.train_context_len_limit if is_train else config.test_context_len_limit
# 		self.question_len_limit=config.train_question_len_limit if is_train else config.test_question_len_limit

# 		self.features=convert_to_ids(examples,word2id)
# 		#features的每一个值是一个dict，keys=={"context_ids","question_ids","start_position","end_position"}

# 	def __len__(self):
# 		return len(self.features)

# 	def __getitem__(self,idx):

class Config:
	def __init__(self):
		self.train_question_len_limit=48
		self.train_context_len_limit=448
		self.test_question_len_limit=48
		self.test_context_len_limit=512
config=Config()

def pad_features(features,config,is_train=True):
	'''
	features的每一个值是一个dict，keys=={"context_ids","question_ids","start_position","end_position"}
	函数的目的是将每一个样本的context_ids和question_ids补全到context_len_limit和question_len_limit
	'''
	def pad_ids(obj_ids,max_seq_len):
		if len(obj_ids)>=max_seq_len:
			obj_ids=obj_ids[:max_seq_len]
		else:
			obj_ids+=[0]*(max_seq_len-len(obj_ids))
		return obj_ids

	context_len_limit=config.train_context_len_limit if is_train else config.test_context_len_limit
	question_len_limit=config.train_question_len_limit if is_train else config.test_question_len_limit
	deprected_example_nums=0
	new_features=[]
	for feature in features:
		context_ids=feature["context_ids"]
		question_ids=feature["question_ids"]
		start_position=feature["start_position"]
		end_position=feature["end_position"]

		if start_position>=context_len_limit or end_position>=context_len_limit:
			deprected_example_nums+=1
			continue

		context_ids=pad_ids(context_ids,context_len_limit)
		question_ids=pad_ids(question_ids,question_len_limit)
		new_features.append({"context_ids":context_ids,"question_ids":question_ids,"start_position":start_position,"end_position":end_position})
	print("total feature nums : %d , deprected_example_nums : %d , rest features : %d"%(len(features),deprected_example_nums,len(new_features)))
	return new_features

def get_dataloader(features,batch_size,is_train=True):
	all_context_ids=torch.tensor([feature["context_ids"] for feature in features],dtype=torch.long)
	all_question_ids=torch.tensor([feature["question_ids"] for feature in features],dtype=torch.long)
	all_start_positions=torch.tensor([feature["start_position"] for feature in features],dtype=torch.long)
	all_end_positions=torch.tensor([feature["end_position"] for feature in features],dtype=torch.long)

	dataset=torch.utils.data.TensorDataset(all_context_ids,all_question_ids,all_start_positions,all_end_positions)
	if is_train:
		sampler=torch.utils.data.RandomSampler(dataset)
	else:
		sampler=torch.utils.data.SequentialSampler(dataset)
	dataloader=torch.utils.data.DataLoader(dataset,sampler=sampler,batch_size=batch_size)
	return dataloader

'''
for i,batch_data in enumerate(dataloader):
	assert type(batch_data)==list
	assert len(batch_data)==4
	batch_data[0].shape==(batch_size,context_len_limit)
	batch_data[1].shape==(batch_size,question_len_limit)
	batch_data[2].shape==(batch_size,)==batch_data[3].shape
'''

