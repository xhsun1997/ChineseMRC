import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def positional_encoding_(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):

    E = inputs.size(2) # static
    N, T = inputs.size(0), inputs.size(1) # dynamic
    #with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
    #position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)
    position_ind=torch.arange(0,T)
    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
        for pos in range(maxlen)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = torch.from_numpy(position_enc).index_select(0,position_ind) # (maxlen, E)

    # lookup
    return position_enc

class LayerNorm(nn.Module):
    def __init__(self,hidden_size):
        super(LayerNorm,self).__init__()
        self.gamma=torch.ones([hidden_size])
        self.beta=torch.zeros([hidden_size])
    def forward(self,inputs,epsilon=1e-5):
        mean=torch.mean(inputs,dim=2,keepdim=True)
        variance=torch.mean((inputs-mean)**2,dim=2,keepdim=True)
        normalize_inputs=(inputs-mean)/torch.sqrt(variance+epsilon)
        return self.gamma*normalize_inputs+self.beta

def layer_dropout(layer_inputs,layer_outputs,dropout_of_this_layer):
    '''
    dropout_of_this_layer=l/L
    '''
    random_number=torch.rand(1)
    if random_number<dropout_of_this_layer:
        return layer_inputs
    else:
        #这一层不去掉
        return nn.Dropout(dropout_of_this_layer)(layer_outputs)+layer_inputs

class Conv(nn.Module):
    def __init__(self,input_size,output_size,kernel_size=1):
        super(Conv,self).__init__()
        self.conv_layer=nn.Conv1d(in_channels=input_size,out_channels=output_size,kernel_size=kernel_size,stride=1)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_layer.weight)
        nn.init.constant_(self.conv_layer.bias,0.)
    def forward(self,inputs):
        inputs_transpose=torch.tranpose(inputs,dim0=2,dim1=1)
        conv_outputs=self.conv_layer(inputs_transpose)
        return torch.tranpose(conv_outputs,dim0=2,dim1=1)

class ConvBlock(nn.Module):
    def __init__(self,input_size,num_conv_layers,kernel_size,num_filters,is_train=True,dropout=0.0):
        super(ConvBlock,self).__init__()
        for i in range(num_conv_layers):
            if i==0:
                setattr(self,"depthwise_separable_conv{}".format(i),Conv(input_size=input_size,output_size=num_filters,kernel_size=kernel_size)
            else: 
                setattr(self,"depthwise_separable_conv{}".format(i),Conv(input_size=num_filters,output_size=num_filters,kernel_size=kernel_size)
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
            if i%2==0:
                layer_outputs=self.dropout_op(layer_outputs)
            layer_outputs=getattr(self,"depthwise_separable_conv{}".format(i))(inputs=layer_outputs)
            layer_outputs=layer_dropout(layer_inputs,layer_outputs,dropout_of_this_layer=self.dropout_rate*float(l)/L)
            l+=1
        return layer_outputs,l

class SelfAttentionBlock(nn.Module):
    def __init__(self,input_size,num_filters,seq_length,mask=None,num_heads=8,is_train=True,dropout=0.0):
        super(SelfAttentionBlock,self).__init__()
        self.dropout_op=nn.Dropout(dropout)
        self.dropout_rate=dropout
        self.num_heads=num_heads
        self.num_filters=num_filters
        self.seq_length=seq_length
        self.mask=mask
        self.is_train=is_train
        self.layer_normal_op=LayerNorm(hidden_size=input_size)#input_size==num_filters
        #self.feed_forward_layer1=Conv(input_size=input_size,output_size=num_filters,kernel_size=1)
        #self.feed_forward_layer2=Conv(input_size=num_filters,output_size=num_filters,kernel_size=1)
        self.project_query=Conv(input_size=input_size,output_size=num_filters,kernel_size=1)
        self.project_key=Conv(input_size=input_size,output_size=num_filters,kernel_size=1)
        self.project_value=Conv(input_size=input_size,output_size=num_filters,kernel_size=1)
        self.feed_forward_layer=nn.Sequential(Conv(input_size=input_size,output_size=num_filters,kernel_size=1),
                                            nn.ReLU(),
                                            Conv(input_size=num_filters,output_size=num_filters,kernel_size=1))

    def dot_product_attention_op(self,query,key,value,value_mask=None,is_train=True):
        '''
        query.size()==key.size()==value.size()==[batch_size,seq_length,num_heads,per_head_dim]
        '''
        query=torch.tranpose(query,dim0=2,dim1=1)
        d_k=key.size(-1)
        key=torch.tranpose(key,dim0=2,dim1=1)
        value=torch.tranpose(value,dim0=2,dim1=1)
        #(batch_size,num_heads,seq_length,per_head_dim)
        Q_K_transpose=torch.matmul(input=query,other=torch.tranpose(key,dim0=3,dim1=2))#(batch_size,num_heads,seq_length,seq_length)
        Q_K_transpose_=Q_K_transpose/torch.sqrt(d_k)
        if value_mask is not None:
            #value_mask.size() should be (batch_size,seq_length)
            batch_size,num_heads,seq_length_q,seq_length_k=Q_K_transpose_.size()
            value_mask_=value_mask.unsqueeze(1).unsqueeze(1).expand(-1,num_heads,seq_length_q,-1)
            #告诉Q，K/V的每一个句子的实际长度
            Q_K_transpose_+=mask_logits(value_mask_)
        att_weights=F.softmax(Q_K_transpose_,dim=3)
        att_weights=self.dropout_op(att_weights)
        weighted_sum=torch.matmul(att_weights,value)
        return weighted_sum


    def multihead_attention_op(self,inputs):
        query=self.project_query(inputs)
        key=self.project_key(inputs)
        value=self.project_value(inputs)
        batch_size,seq_length,hidden_size=query.size()
        hidden_size=query.size(2)
        assert hidden_size%self.num_heads==0
        per_head_dim=hidden_size//self.num_heads
        query=torch.reshape(query,(batch_size,seq_length,self.num_heads,per_head_dim))
        key=torch.reshape(key,(batch_size,seq_length,self.num_heads,per_head_dim))
        value=torch.reshape(value,(batch_size,seq_length,self.num_heads,per_head_dim))
        weighted_sum=self.dot_product_attention_op(query=query,key=key,value=value,value_mask=self.mask,is_train=self.is_train)
        #(batch_size,num_heads,seq_length,per_head_dim)
        outputs=torch.tranpose(weighted_sum,dim0=2,dim1=1)
        assert outputs.size(0)==batch_size and outputs.size(1)==seq_length
        outputs=torch.reshape(outputs,(batch_size,seq_length,hidden_size))
        return outputs


    def forward(self,inputs,sublayers):
        l,L=sublayers
        outputs=self.layer_normal_op(inputs=inputs)
        outputs=self.dropout_op(outputs)
        outputs=self.multihead_attention_op(inputs=outputs)
        outputs=layer_dropout(layer_inputs=inputs,layer_outputs=outputs,dropout_of_this_layer=self.dropout_rate*l/L)
        l+=1
        ##################FFN
        ffn_inputs=outputs
        ffn_outputs=self.layer_normal_op(inputs=ffn_inputs)
        ffn_outputs=self.dropout_op(ffn_outputs)
        ffn_outputs=self.feed_forward_layer(ffn_outputs)

        att_block_outputs=layer_dropout(layer_inputs=ffn_inputs,layer_outputs=ffn_outputs,dropout_of_this_layer=self.dropout_rate*l/L)
        return att_block_outputs,l






class ResidualBlock(nn.Module):
    def __init__(self,input_size,num_blocks,num_conv_layers,kernel_size,num_heads=8,input_projection=False,mask=None,num_filters=128,dropout=0.0,is_train=True):
        super(ResidualBlock,self).__init__()
        self.input_projection=input_projection
        if input_projection:
            self.input_projection_layer=Conv(input_size=input_size,output_size=num_filters)

        self.conv_block=ConvBlock(input_size,num_conv_layers,kernel_size,num_filters,is_train=is_train,dropout=dropout)
        self.self_att_block=SelfAttentionBlock(input_size=num_filters,num_filters=num_filters,mask=mask,
                                            num_heads=num_heads,is_train=is_train,dropout=dropout):

        self.num_blocks=num_blocks
        self.total_sublayers=(num_conv_layers+2)*num_blocks

    def forward(self,inputs):
        l=1
        outputs=inputs
        for i in range(self.num_blocks):
            outputs,l=self.conv_block(inputs=outputs,sublayers=(l,self.total_sublayers))
            outputs,l=self.self_att_block(inputs=outputs,outputs,sublayers=(l,self.total_sublayers))
        return outputs


        

