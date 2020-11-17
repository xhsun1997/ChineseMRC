import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def positional_encoding_(inputs,mask=None):
    '''
    函数用来获得位置编码，返回的shape和inputs一样
    也就是batch_size个(seq_length,dim)的矩阵，每一个矩阵都是一样的
    也就是说seq_length的每一个单词用一个dim长度的向量表示
    '''
    batch_size,seq_length,dim=inputs.size()

    position_ind=torch.arange(0,seq_length)
    position_enc = np.array([
        [pos / np.power(10000, (i-i%2)/dim) for i in range(dim)]
        for pos in range(seq_length)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = torch.from_numpy(position_enc).index_select(0,position_ind) # (seq_length,dim)

    # lookup
    position_enc=position_enc.unsqueeze(0).expand(batch_size,-1,-1)
    return position_enc.to(device)

class LayerNorm(nn.Module):
    '''
    层正则化，hidden_size就是inputs的最后一个维度
    '''
    def __init__(self,hidden_size):
        super(LayerNorm,self).__init__()
        self.gamma=torch.ones([hidden_size]).to(device)
        self.beta=torch.zeros([hidden_size]).to(device)
    def forward(self,inputs,epsilon=1e-5):
        mean=torch.mean(inputs,dim=2,keepdim=True)
        variance=torch.mean((inputs-mean)**2,dim=2,keepdim=True)
        normalize_inputs=(inputs-mean)/torch.sqrt(variance+epsilon)
        #print(self.gamma.device,self.beta.device,normalize_inputs.device)
        return self.gamma*normalize_inputs+self.beta

def layer_dropout(layer_inputs,layer_outputs,dropout_of_this_layer):
    '''
    随机深度，随着层数的增加，dropout_of_this_layer会越来越大
    dropout_of_this_layer=dropout_rate*l/L
    '''
    random_number=torch.rand(1)
    if random_number<dropout_of_this_layer:
        return layer_inputs#直接去掉这一层，返回这一层的输入
    else:
        #这一层不去掉
        return nn.Dropout(dropout_of_this_layer)(layer_outputs)+layer_inputs

class Conv(nn.Module):
    '''
    整个模型都是用卷积代替全连接
    inputs-->(batch_size,dim,seq_length)-->(conv)-->(batch_size,seq_length,dim)
    '''
    def __init__(self,input_size,output_size,kernel_size=1,padding=0):
        super(Conv,self).__init__()
        self.conv_layer=nn.Conv1d(in_channels=input_size,out_channels=output_size,kernel_size=kernel_size,padding=padding)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_layer.weight)
        nn.init.constant_(self.conv_layer.bias,0.)
    def forward(self,inputs):
        #print("inputs to Conv shape : ",inputs.size())
        inputs_transpose=torch.transpose(inputs,dim0=2,dim1=1)
        conv_outputs=self.conv_layer(inputs_transpose)#(inputs.shape==(N,C_in,L),outputs.shape==(N,C_out,L_out))
        return torch.transpose(conv_outputs,dim0=2,dim1=1)

class ConvBlock(nn.Module):
    '''
    一共有num_conv_layers个卷积块
    每一个卷积块都是layerNormal+Conv1d+layer_dropout
    其中layer_dropout做了残差连接的操作
    '''
    def __init__(self,input_size,num_conv_layers,kernel_size,num_filters,is_train=True,dropout=0.0):
        super(ConvBlock,self).__init__()
        assert kernel_size>1
        for i in range(num_conv_layers):
            if i==0:
                setattr(self,"depthwise_separable_conv{}".format(i),Conv(input_size=input_size,output_size=num_filters,kernel_size=kernel_size,padding=1))
            else: 
                setattr(self,"depthwise_separable_conv{}".format(i),Conv(input_size=num_filters,output_size=num_filters,kernel_size=kernel_size,padding=1))
        #由于卷积操作时kernel_size大于１，所以padding要等于１才能保证seq_length不变
        self.layer_normal_op=LayerNorm(hidden_size=num_filters)
        self.num_conv_layers=num_conv_layers
        self.dropout_op=nn.Dropout(dropout)
        self.dropout_rate=dropout
    def forward(self,inputs,sublayers):
        layer_outputs=inputs
        l,L=sublayers
        for i in range(self.num_conv_layers):
            layer_inputs=layer_outputs
            layer_outputs=self.layer_normal_op(layer_inputs)
            #if i%2==0:
            #    layer_outputs=self.dropout_op(layer_outputs)
            layer_outputs=getattr(self,"depthwise_separable_conv{}".format(i))(inputs=layer_outputs)
            layer_outputs=layer_dropout(layer_inputs=layer_inputs,layer_outputs=layer_outputs,dropout_of_this_layer=self.dropout_rate*float(l)/L)
            l+=1
        return layer_outputs,l
    
def mask_logits(mask,mask_value=-1e30):
    #mask.shape==(batch_size,num_heads,seq_length,seq_length)
    return (1.0-mask.float())*mask_value

class SelfAttentionBlock(nn.Module):
    '''
    自注意力块，包含自注意力运算和前馈网络
    '''
    def __init__(self,input_size,num_filters,num_heads=8,is_train=True,dropout=0.0):
        super(SelfAttentionBlock,self).__init__()
        self.dropout_op=nn.Dropout(dropout)
        self.dropout_rate=dropout
        self.num_heads=num_heads
        self.num_filters=num_filters
        self.is_train=is_train
        self.layer_normal_op=LayerNorm(hidden_size=input_size)#input_size==num_filters
        #self.feed_forward_layer1=Conv(input_size=input_size,output_size=num_filters,kernel_size=1)
        #self.feed_forward_layer2=Conv(input_size=num_filters,output_size=num_filters,kernel_size=1)
        self.project_query=Conv(input_size=input_size,output_size=num_filters,kernel_size=1,padding=0)
        self.project_key=Conv(input_size=input_size,output_size=num_filters,kernel_size=1,padding=0)
        self.project_value=Conv(input_size=input_size,output_size=num_filters,kernel_size=1,padding=0)
        self.feed_forward_layer=nn.Sequential(Conv(input_size=input_size,output_size=num_filters,kernel_size=1,padding=0),
                                            nn.ReLU(),
                                            Conv(input_size=num_filters,output_size=num_filters,kernel_size=1,padding=0))

    def dot_product_attention_op(self,query,key,value,value_mask=None,is_train=True):
        '''
        query.size()==key.size()==value.size()==[batch_size,seq_length,num_heads,per_head_dim]
        '''
        query=torch.transpose(query,dim0=2,dim1=1)
        d_k=key.size(-1)
        key=torch.transpose(key,dim0=2,dim1=1)
        value=torch.transpose(value,dim0=2,dim1=1)
        #(batch_size,num_heads,seq_length,per_head_dim)
        
        #print(query.size(),key.size(),value.size(),"IN dot product attention")
        Q_K_transpose=torch.matmul(input=query,other=torch.transpose(key,dim0=3,dim1=2))#(batch_size,num_heads,seq_length,seq_length)
        Q_K_transpose_=Q_K_transpose/math.sqrt(d_k)
        if value_mask is not None:
            #value_mask.size() should be (batch_size,seq_length) and value_mask.dtype should be bool
            batch_size,num_heads,seq_length_q,seq_length_k=Q_K_transpose_.size()
            value_mask_=value_mask.unsqueeze(1).unsqueeze(1).expand(-1,num_heads,seq_length_q,-1)
            #告诉Q的每一个头的每一个句子的每一个单词，K/V的每一个句子的实际长度
            Q_K_transpose_+=mask_logits(value_mask_)
        att_weights=F.softmax(Q_K_transpose_,dim=3)
        att_weights=self.dropout_op(att_weights)
        weighted_sum=torch.matmul(att_weights,value)
        return weighted_sum


    def multihead_attention_op(self,inputs,mask=None):
        query=self.project_query(inputs)
        key=self.project_key(inputs)
        value=self.project_value(inputs)
        batch_size,seq_length,hidden_size=query.size()
        hidden_size=query.size(2)
        #print(query.size(),key.size(),value.size())
        assert hidden_size%self.num_heads==0
        per_head_dim=hidden_size//self.num_heads
        query=torch.reshape(query,(batch_size,seq_length,self.num_heads,per_head_dim))
        key=torch.reshape(key,(batch_size,seq_length,self.num_heads,per_head_dim))
        value=torch.reshape(value,(batch_size,seq_length,self.num_heads,per_head_dim))
        
        #print(query.size(),key.size(),value.size())
        weighted_sum=self.dot_product_attention_op(query=query,key=key,value=value,value_mask=mask,is_train=self.is_train)
        #(batch_size,num_heads,seq_length,per_head_dim)
        outputs=torch.transpose(weighted_sum,dim0=2,dim1=1)
        #print("tensor in multihead attention shape : ",outputs.size())
        assert outputs.size(0)==batch_size and outputs.size(1)==seq_length
        outputs=torch.reshape(outputs,(batch_size,seq_length,hidden_size))
        return outputs


    def forward(self,inputs,sublayers,mask=None):
        l,L=sublayers
        outputs=self.layer_normal_op(inputs=inputs)
        outputs=self.dropout_op(outputs)
        outputs=self.multihead_attention_op(inputs=outputs,mask=mask)
        outputs=layer_dropout(layer_inputs=inputs,layer_outputs=outputs,dropout_of_this_layer=self.dropout_rate*l/L)
        l+=1
        ##################FFN
        ffn_inputs=outputs
        ffn_outputs=self.layer_normal_op(inputs=ffn_inputs)
        ffn_outputs=self.dropout_op(ffn_outputs)
        ffn_outputs=self.feed_forward_layer(ffn_outputs)

        att_block_outputs=layer_dropout(layer_inputs=ffn_inputs,layer_outputs=ffn_outputs,dropout_of_this_layer=self.dropout_rate*l/L)
        l+=1
        return att_block_outputs,l

class ResidualBlock(nn.Module):
    '''
    input_size就是embedding_dim,num_filters就是所有模块的hidden_size
    '''
    def __init__(self,input_size,num_blocks,num_conv_layers,kernel_size,num_heads=8,input_projection=False,num_filters=128,dropout=0.0,is_train=True):
        super(ResidualBlock,self).__init__()
        self.input_projection=input_projection
        if input_projection:
            self.input_projection_layer=Conv(input_size=input_size,output_size=num_filters)

        self.conv_block=ConvBlock(input_size,num_conv_layers,kernel_size,num_filters,is_train=is_train,dropout=dropout)
        self.self_att_block=SelfAttentionBlock(input_size=num_filters,num_filters=num_filters,
                                            num_heads=num_heads,is_train=is_train,dropout=dropout)

        self.num_blocks=num_blocks
        self.total_sublayers=(num_conv_layers+2)*num_blocks

    def forward(self,inputs,mask=None):
        l=1
        if self.input_projection:
            #print("inputs_projection ",inputs.size())
            outputs=self.input_projection_layer(inputs)
        else:
            outputs=inputs

        for i in range(self.num_blocks):
            outputs,l=self.conv_block(inputs=outputs,sublayers=(l,self.total_sublayers))
            outputs,l=self.self_att_block(inputs=outputs,sublayers=(l,self.total_sublayers),mask=mask)
        return outputs


class QANet(nn.Module):
    def __init__(self,config,pretrained_word_embedding=None,is_train=True):
        super(QANet,self).__init__()
        self.args=config
        if pretrained_word_embedding is None:
            self.word_embedding=nn.Embedding(self.args.vocab_size,self.args.embed_dim,padding_idx=0)
            nn.init.uniform_(self.word_embedding.weight,-0.1,0.1)
        else:
            self.word_embedding=nn.Embedding.from_pretrained(pretrained_word_embedding,freeze=False)

        for i in range(self.args.highway_network_layers):
            setattr(self,"highway_linear{}".format(i),nn.Sequential(Conv(input_size=self.args.embed_dim,
                                                                        output_size=self.args.embed_dim),nn.ReLU()))
            setattr(self,"highway_gate{}".format(i),nn.Sequential(Conv(input_size=self.args.embed_dim,
                                                                        output_size=self.args.embed_dim),nn.Sigmoid()))
    
        self.Embedding_Encoder_Block=ResidualBlock(input_size=self.args.embed_dim,num_blocks=1,
                                        num_conv_layers=3,kernel_size=3,num_heads=8,
                                        input_projection=False,num_filters=self.args.num_filters,dropout=self.args.dropout,is_train=is_train)
        #因为input_projection是True，所以对于卷积层来说，她的输入channels是num_filters而不是embed_dim
        
        self.att_project_layer=Conv(input_size=self.args.num_filters*4,output_size=self.args.num_filters)
        for i in range(self.args.model_encoder_block_nums):
            setattr(self,"Model_Encoder_Block{}".format(i),ResidualBlock(input_size=self.args.num_filters,
                                                                           num_blocks=1,num_conv_layers=3,
                                                                           kernel_size=3,num_heads=8,
                                                                           input_projection=False,
                                                                           num_filters=self.args.num_filters,
                                                                           dropout=self.args.dropout,
                                                                           is_train=is_train))
                
        self.att_weight_context=Conv(input_size=self.args.num_filters,output_size=1)
        self.att_weight_question=Conv(input_size=self.args.num_filters,output_size=1)
        self.att_weight_cq=Conv(input_size=self.args.num_filters,output_size=1)

        self.start_logits_layer=Conv(input_size=self.args.num_filters*2,output_size=1)
        self.end_logits_layer=Conv(input_size=self.args.num_filters*2,output_size=1)

    def highway_network_layer(self,inputs):
        for i in range(self.args.highway_network_layers):
            h=getattr(self,"highway_linear{}".format(i))(inputs)
            g=getattr(self,"highway_gate{}".format(i))(inputs)
            inputs=h*g+(1-g)*inputs
        return inputs



    def Context_Question_Interaction(self,context,question,context_mask=None,question_mask=None):
        '''
        inputs:
            context.shape==(batch_size,seq_length_context,dim)
            question.shape==(batch_size,seq_length_question,dim)
        procedure:
            S=(batch_size,seq_length_context,seq_length_question)
            C2Q=matmul(S,question)
            Q2C=matmul(matmul(S,S_T),context)
        outputs:
            [context,C2Q,C2Q*context,Q2C*context] shape==(batch_size,seq_length_context,dim*4)
        '''
        c_length=context.size(1)
        q_length=question.size(1)

        c_att=self.att_weight_context(context).expand(-1,-1,q_length)
        q_att=self.att_weight_question(question).permute(0,2,1).expand(-1,c_length,-1)

        cq=[]
        for i in range(q_length):
            q_i=question.select(1,i).unsqueeze(1)#(batch_size,1,num_filters)
            elementwise_matmul=context*q_i#(batch_size,c_length,num_filters)
            cq.append(self.att_weight_cq(elementwise_matmul).squeeze())
        #cq是长度为q_length的list，每一个元素是(batch_size,c_length）的tensor
        cq_att=torch.stack(cq,dim=-1)#(batch_size,c_length,q_length)
        S=c_att+q_att+cq_att

        if question_mask is not None:
            S_=F.softmax(S+(1.0-question_mask.unsqueeze(1).expand(-1,c_length,-1).float())*(-1e30),dim=2)
        else:
            S_=F.softmax(S,dim=2)
        C2Q=torch.bmm(S_,question)

        if context_mask is not None:
            S_T=torch.transpose(F.softmax(S+(1.0-context_mask.unsqueeze(2).expand(-1,-1,q_length).float())*(-1e30),dim=1),
                                dim0=2,dim1=1)
        else:
            S_T=torch.transpose(F.softmax(S,dim=1),dim0=2,dim1=1)

        Q2C=torch.bmm(torch.bmm(S_,S_T),context)
        return torch.cat([context,C2Q,context*C2Q,context*Q2C],dim=2)

    def output_layer(self,model_encoder_blocks_result):
        assert len(model_encoder_blocks_result)==self.args.model_encoder_block_nums
        start_inputs=torch.cat([model_encoder_blocks_result[0],model_encoder_blocks_result[1]],dim=2)
        end_inputs=torch.cat([model_encoder_blocks_result[1],model_encoder_blocks_result[2]],dim=2)

        start_logits=self.start_logits_layer(start_inputs).squeeze()
        end_logits=self.end_logits_layer(end_inputs).squeeze()
        return start_logits,end_logits


    def Model_Encoder_Layer(self,att_outputs,mask=None):
        model_encoder_blocks_result=[]
        outputs=att_outputs
        for i in range(self.args.model_encoder_block_nums):
            model_encoder_block_i=getattr(self,"Model_Encoder_Block{}".format(i))
            model_encoder_blocks_result.append(model_encoder_block_i(inputs=outputs,mask=mask))
            outputs=model_encoder_blocks_result[-1]
        return model_encoder_blocks_result

    def forward(self,context_ids,question_ids):
        context_ids=context_ids.to(device)
        question_ids=question_ids.to(device)
        context_mask=context_ids.bool()
        question_mask=question_ids.bool()

        context=self.word_embedding(context_ids)
        context+=positional_encoding_(context)
        question=self.word_embedding(question_ids)
        question+=positional_encoding_(question)
        #(batch_size,seq_length,embed_dim)
        context=self.highway_network_layer(inputs=context)
        question=self.highway_network_layer(inputs=question)

        #(batch_size,seq_length,num_filters)

        context=self.Embedding_Encoder_Block(inputs=context,mask=context_mask)
        question=self.Embedding_Encoder_Block(inputs=question,mask=question_mask)

        att_outputs=self.Context_Question_Interaction(context=context,question=question,context_mask=context_mask,question_mask=question_mask)
        #(batch_size,c_length,dim*4)
        att_outputs=self.att_project_layer(att_outputs)#(batch_size,c_length,dim)
        #print("After Interaction layer shape : ",att_outputs.size())
        model_encoder_blocks_result=self.Model_Encoder_Layer(att_outputs,mask=context_mask)
        start_logits,end_logits=self.output_layer(model_encoder_blocks_result)
        return start_logits,end_logits
