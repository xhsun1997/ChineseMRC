import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    '''
    gamma*(x-miu)/sqrt(variance+eps)+beta
    后面的层的参数分布会因为前面的层的权重和偏执的不断变化而变化，使得后面的层要不停的适应这种变化
    gamma和beta的维度要和inputs的最后一个维度一致，也就是hidden_size
    '''
    def __init__(self,hidden_size=256):
        super(LayerNorm,self).__init__()
        self.gamma=torch.ones(hidden_size)
        self.beta=torch.zeros(hidden_size)
    def forward(self,inputs,epsilon=1e-6):
        mean=torch.mean(inputs,dim=2,keepdim=True)#(batch_size,seq_length,1)
        variance=torch.mean(torch.square(inputs-mean),dim=2,keepdim=True)
        normalized_inputs=(inputs-mean)/torch.sqrt(variance+epsilon)
        return self.gamma*normalized_inputs+self.beta

class Conv_Layer(nn.Module):
    '''
    inputs.shape==(batch_size,seq_length,dim)，outputs.shape==(batch_size,seq_length,dim)
    流程为：
    inputs-->(batch_size,dim,seq_length,1)-->depthwise_separable_conv-->
    -->(batch_size,dim,seq_length,1)-->outputs(batch_size,seq_length,1)
    depthwise_separable_conv的过程:
    根据inputs.size(1)，也就是in_channels的数目和kernel_size，
    构成出来in_channels个(kernel_size,1)形状的卷积核，padding=1，此过程为depthwise卷积
    输出的tensor为(batch_size,seq_length,1)，一共有in_channels个，构成了
    (batch_size,dim,seq_length,1)的tensor
    
    然后是pointwise卷积，此时卷积核的形状为(in_channels,1,1)，一共有out_channels个
    所以最终的输出tensor.shape==(batch_size,out_channels,seq_length,1)
    '''
    def __init__(self,in_channels,out_channles,kernel_size=3):
        super(Conv_Layer,self).__init__()
        #in_channls==out_channels==inputs.size(2)
        self.depthwise_layer=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,groups=in_channels,
                                       kernel_size=(kernel_size,1),padding=(1,0))
        self.pointwise_layer=nn.Conv2d(in_channels=in_channels,out_channels=out_channles,
                                      kernel_size=(1,1))
        self.layer_norm=LayerNorm(hidden_size=in_channels)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.depthwise_layer.weight)
        nn.init.kaiming_normal_(self.pointwise_layer.weight)
    def forward(self,inputs):
        inputs=self.layer_norm(inputs)
        if len(inputs.size())==3:
            inputs=torch.transpose(inputs,dim0=2,dim1=1)#(batch_size,dim,seq_length)
            inputs.unsqueeze_(3)
        outputs=self.pointwise_layer(self.depthwise_layer(inputs))#(batch_size,dim,seq_length,1)
        outputs.squeeze_()
        return torch.transpose(outputs,dim0=2,dim1=1)
        
class SelfAttention_Layer(nn.Module):
    '''
    对inputs计算self attention
    '''
    def __init__(self,hidden_size=256):
        super(SelfAttention_Layer,self).__init__()
        self.layer_norm=LayerNorm(hidden_size)
    
    def forward(self,inputs,inputs_mask=None):
        inputs=self.layer_norm(inputs)
        batch_size,seq_length,hidden_size=inputs.size()
        Q=inputs
        V=inputs
        K_transpose=torch.transpose(inputs,dim0=2,dim1=1)#(batch_size,dim,seq_length)
        S=torch.bmm(Q,K_transpose)#(batch_size,seq_length,seq_length)
        if inputs_mask is not None:
            #V_mask.shape==(batch_size,seq_length)
            inputs_mask.unsqueeze_(1)#(batch_size,1,seq_length)
            S+=(1.0-inputs_mask.long())*(-1e30)#pad的位置加上-1e30
        att_weights=F.softmax(S,dim=2)
        return torch.bmm(att_weights,V)
    
#     def forward(self,inputs,inputs_mask=None):
#         return self.dot_product_attention(inputinputs_mask)

class FeedForward_Layer(nn.Module):
    def __init__(self,intermediate_size=512,hidden_size=256):
        super(FeedForward_Layer,self).__init__()
        self.feedforward_layer1=nn.Linear(in_features=hidden_size,out_features=intermediate_size)
        self.feedforward_layer2=nn.Linear(in_features=intermediate_size,out_features=hidden_size)
        self.layer_norm=LayerNorm(hidden_size)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.feedforward_layer1.weight)
        nn.init.kaiming_normal_(self.feedforward_layer2.weight)
        nn.init.constant_(self.feedforward_layer1.bias,0)
        nn.init.constant_(self.feedforward_layer2.bias,0)
        
    def forward(self,inputs):
        outputs=self.feedforward_layer1(self.layer_norm(inputs))
        return self.feedforward_layer2(F.relu(outputs))
        
    
class Conv_with_Attention_Block(nn.Module):
    '''
    该模块下有3层卷积加一个self attention层
    具体的：　(LayerNorm+Conv)+(LayerNorm+Conv)+(LayerNorm+Conv)+(LayerNorm+self-att)+(LayerNorm+FF)
    '''
    def __init__(self,intermediate_size=512,hidden_size=256,kernel_size=3,conv_layer_nums=4):
        super(Conv_with_Attention_Block,self).__init__()
        self.conv_layer_nums=conv_layer_nums
        self.kernel_size=kernel_size
        self.hidden_size=hidden_size
        self.layer_norm=LayerNorm(hidden_size=hidden_size)
        self.linear_layer=nn.Linear(in_features=hidden_size*2,out_features=hidden_size)
        for i in range(conv_layer_nums):
            setattr(self,"conv_layer{}".format(i),Conv_Layer(in_channels=hidden_size,out_channles=hidden_size,kernel_size=kernel_size))
        self.feedforward_layer=FeedForward_Layer(intermediate_size=intermediate_size,hidden_size=hidden_size)
        self.selfAtt_layer=SelfAttention_Layer(hidden_size=hidden_size)
    def forward(self,inputs):
        if inputs.size(2)!=self.hidden_size:
            outputs=self.layer_norm(self.linear_layer(inputs))
        else:
            outputs=inputs
        for i in range(self.conv_layer_nums):
            outputs=getattr(self,"conv_layer{}".format(i))(outputs)+outputs
            
        selfAtt_outputs=self.selfAtt_layer(outputs)+outputs
        outputs=self.feedforward_layer(selfAtt_outputs)+selfAtt_outputs
        return outputs
        



# class Linear(nn.Module):
#     def __init__(self,in_features,out_features):
#         self.
class My_Model(nn.Module):
    def __init__(self,config,pretrained_word_embedding=None):
        super(My_Model,self).__init__()
        self.args=config
        self.turn_nums=config.turn_nums
        intermediate_size=config.intermediate_size
        hidden_size=config.hidden_size
        kernel_size=config.kernel_size
        conv_layer_nums=config.conv_layer_nums
        self.dropout_layer=nn.Dropout(self.args.dropout)
        if pretrained_word_embedding is None:
            self.word_embedding=nn.Embedding(self.args.vocab_size,embedding_dim=self.args.embed_dim,padding_idx=0)
            nn.init.uniform_(self.word_embedding.weight,-0.1,0.1)
        else:
            self.word_embedding=nn.Embedding.from_pretrained(pretrained_word_embedding,freeze=False)
        for i in range(self.turn_nums):
            setattr(self,"encoder_block{}".format(i),Conv_with_Attention_Block(intermediate_size=intermediate_size,
                                                                               hidden_size=hidden_size,
                                                                               kernel_size=kernel_size,
                                                                               conv_layer_nums=conv_layer_nums))

        self.output_layer=nn.Linear(in_features=hidden_size*2,out_features=2)
#         for i in range(self.args.highway_network_layers):
#             setattr(self,"highway_linear{}".format(i),nn.Sequential())
    def interaction_attention_op(self,context,question,context_mask=None,question_mask=None):
        '''
        inputs是context和question的向量表示
        outputs是context_aware_question_representation和question_aware_context_representation
        '''
        question_transpose=torch.transpose(question,dim0=2,dim1=1)
        similarity_matrix=torch.bmm(context,question_transpose)#(batch_size,context_length,question_length)
        batch_size,context_len,question_len=similarity_matrix.size()
        if question_mask is not None:
            #question_mask.shape==(batch_size,question_length)
            print("question_mask size : ",question_mask.size())
            question_mask=question_mask.unsqueeze(1).expand(-1,context_len,-1)
            similarity_matrix+=(1.0-question_mask.long())*(-1e30)
        question_aware_context_representation=torch.bmm(F.softmax(similarity_matrix,dim=2),question)
        
        if context_mask is not None:
            #context_mask.shape==(batch_size,context_length)
            context_mask=context_mask.unsqueeze(2).expand(-1,-1,question_len)
            similarity_matrix+=(1.0-context_mask.long())*(-1e30)
        context_aware_question_representation=torch.bmm(F.softmax(torch.transpose(similarity_matrix,dim0=2,dim1=1),dim=2),context)
        
        return question_aware_context_representation,context_aware_question_representation
    def forward(self,context_ids,question_ids):
        context_ids=context_ids.to(device)
        question_ids=question_ids.to(device)
        context_mask=context_ids.bool()
        question_mask=question_ids.bool()
        print("mask shape : ",context_mask.size(),question_mask.size())
        
        context=self.dropout_layer(self.word_embedding(context_ids))
        question=self.dropout_layer(self.word_embedding(question_ids))
        
        for i in range(self.turn_nums):
            encoder_block_i=getattr(self,"encoder_block{}".format(i))
            context=encoder_block_i(context)
            question=encoder_block_i(question)
            print(context.size(),question.size())
            Q2C,C2Q=self.interaction_attention_op(context,question,
                                                  context_mask=context_mask,
                                                 question_mask=question_mask)
            #Q2C.shape==(batch_size,context_length,dim)
            #C2Q.shape==(batch_size,question_length,dim)
            context=torch.cat([context,Q2C],dim=2)
            question=torch.cat([question,C2Q],dim=2)
        
        logits=self.output_layer(context)
        start_logits=logits[:,:,0]
        end_logits=logits[:,:,1]
        return start_logits,end_logits