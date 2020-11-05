import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
	def __init__(self,in_features,out_features,dropout=0.0):
		super(Linear,self).__init__()

		self.linear=nn.Linear(in_features=in_features,out_features=out_features)
		if dropout>0.0:
			self.dropout=nn.Dropout(dropout)
		self.reset_params()

	def reset_params(self):
		nn.init.kaiming_normal_(self.linear.weight)
		nn.init.constant_(self.linear.bias,0)

	def forward(self,x):
		if hasattr(self,"dropout"):
			x=self.dropout(x)
		return self.linear(x)


'''
lstm:
	i_t=sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi})
	f_t=sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf})
	g_t=tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg})
	o_t=sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho})

	c_t=f_t*c_{t-1}+i_t*g_t
	h_t=o_{t}*tanh(g_t)
W_{ii}/W_{if}/W_{ig}/W_{io} shape is (hidden_size,input_size)
so weight_ih_l[k].shape==(4*hidden_size,input_size)
同样的weight_hh_l[k].shape==(4*hidden_size,hidden_size)
bias_ih_l[k].shape==(4*hidden_size,)
bias_hh_l[k].shape==(4*hidden_size,)

'''

class LSTM(nn.Module):
	def __init__(self,input_size,hidden_size,batch_first=True,num_layers=1,bidirectional=True,dropout=0.0):
		super(LSTM,self).__init__()
		if dropout>0.0:
			self.dropout=nn.Dropout(dropout)

		self.rnn=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
						bidirectional=bidirectional,batch_first=batch_first)
		self.reset_params()

	def reset_params(self):
		for i in range(self.rnn.num_layers):
			nn.init.orthogonal_(getattr(self.rnn,f'weight_hh_l{i}'))
			nn.init.kaiming_normal_(getattr(self.rnn,f'weight_ih_l{i}'))
			nn.init.constant_(getattr(self.rnn,f'bias_hh_l{i}'),val=0)
			nn.init.constant_(getattr(self.rnn,f'bias_ih_l{i}'),val=0)
			getattr(self.rnn,f'bias_hh_l{i}').chunk(4)[1].fill_(1)#将bias_hh_l[k]分成4块，第二块的张量用1填充，注意到第二块tensor为b_{hf}

			if self.rnn.bidirectional:
				nn.init.orthogonal_(getattr(self.rnn,f'weight_hh_l{i}_reverse'))#正交初始化可以缓解gradient vanish and gradient exploration
				nn.init.kaiming_normal_(getattr(self.rnn,f"weight_ih_l{i}_reverse"))
				nn.init.constant_(getattr(self.rnn,f"bias_ih_l{i}_reverse"),val=0)
				nn.init.constant_(getattr(self.rnn,f"bias_hh_l{i}_reverse"),val=0)
				getattr(self.rnn,f"bias_hh_l{i}_reverse").chunk(4)[1].fill_(1)

	def forward(self,inputs,inputs_length):
		'''
		eg: inputs_length=[25,14,26,85,11]
		那么：
			inputs_length_sorted=[85,26,25,14,11] and input_length_ids=[3,2,0,1,4]

		pack_padded_sequence(input,lengths,batch_first=False,enforce_sorted=True)
		if batch_first=True,then input.shape should be like (batch_size,max_seq_len,*)
		if enforce_sorted=True,then sequence inputs should be sorted by length in descending order, which input[0,:] should be 
		the longest sequence and input[-1,:] should be the shortest one

		pack_padded_sequence return a PackedSequence object他所返回的是经过pack的sequence
		'''
		if hasattr(self,"dropout"):
			print("Dropout before lstm layer")
			inputs=self.dropout(inputs)
		inputs_length_sorted,input_length_ids=torch.sort(inputs_length,descending=True)#将batch_size个句子按照句子长度由长到短排序
		#input_length_ids是排序后的batch_size个句子的索引
		inputs_sorted=inputs.index_select(0,input_length_ids)
		_,original_index=torch.sort(input_length_ids)

		packed_inputs=torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted,inputs_length_sorted,batch_first=True,enforce_sorted=True)
		rnn_output_packed,(h,c)=self.rnn(packed_inputs)

		rnn_output,packed_length=torch.nn.utils.rnn.pad_packed_sequence(rnn_output_packed,batch_first=True)
		assert packed_length.equal(inputs_length_sorted)
		#此时的rnn_output是按照顺序排列的，我们需要还原原来的顺序
		rnn_output=rnn_output.index_select(0,original_index)
		#rnn_output.shape==(batch_size,seq_length,num_directions*hidden_size)
		#h.shape==(num_directions*num_layers,batch_size,hidden_size)
		h=h.permute(1,0,2).contiguous()#(batch_size,num_directionals*num_layers,hidden_size)
		h=h.index_select(0,original_index)
		h=h.permute(1,0,2)

		return rnn_output,h


class BiDAF(nn.Module):
	def __init__(self,config,pretrained_word_embedding=None):
		super(BiDAF,self).__init__()
		self.args=config
		if pretrained_word_embedding is None:
			self.word_embedding=nn.Embedding(self.args.vocab_size,self.args.embed_dim,padding_idx=0)
			nn.init.uniform_(self.word_embedding.weight,-0.1,0.1)
		else:
			self.word_embedding=nn.Embedding.from_pretrained(pretrained_word_embedding,freeze=False)

		for i in range(self.args.highway_network_layers):
			setattr(self,"highway_linear{}".format(i),nn.Sequential(Linear(self.args.hidden_size*2,self.args.hidden_size*2),nn.ReLU()))
			setattr(self,"highway_gate{}".format(i),nn.Sequential(Linear(self.args.hidden_size*2,self.args.hidden_size*2),nn.Sigmoid()))

		self.contextLSTM=LSTM(input_size=self.args.hidden_size*2,hidden_size=self.args.hidden_size,bidirectional=True,batch_first=True,num_layers=1,dropout=self.args.dropout)

		self.output_layer=Linear(in_features=self.args.hidden_size*2,out_features=2)

	def highway_network(self,inputs):
		if inputs.size(-1)!=self.args.hidden_size*2:
			self.temp_layer=nn.Linear(in_features=inputs.size(-1),out_features=self.args.hidden_size*2)
			inputs=self.temp_layer(inputs)
		for i in range(self.args.highway_network_layers):
			print(inputs.shape)
			h=getattr(self,'highway_linear{}'.format(i))(inputs)
			g=getattr(self,"highway_gate{}".format(i))(inputs)
			inputs=g*h+(1-g)*inputs
		return inputs#(batch_size,seq_length,hidden_size*2)


	#def bi_attention_flow(self,context,question):


	def forward(self,batch_data):
		context_ids,question_ids,start_positions,end_positions=batch_data
		context_lengths=torch.sum(context_ids.bool().long(),1)
		question_lengths=torch.sum(question_ids.bool().long(),1)

		context_embeddings=self.word_embedding(context_ids)
		question_embeddings=self.word_embedding(question_ids)
		print("embeddigs.shape : ",context_embeddings.shape,question_embeddings.shape)
		context=self.highway_network(context_embeddings)
		question=self.highway_network(question_embeddings)
		#(batch_size,seq_length,hidden_size*2)

		context=self.contextLSTM(context,context_lengths)[0]
		question=self.contextLSTM(question,question_lengths)
		#num_layers=1,bidirectional=True (batch_size,seq_length,hidden_size*2)

		(start_logits,end_logits)=torch.unbind(self.output_layer(context),dim=-1)#2 个(batch_size,seq_length)
		return start_logits,end_logits






