import torch
import torch.nn as nn
import torch.nn.functional as F
import math



D = 96
Nh = 2
Dword = 300
batch_size = 1
dropout = 0.5

Lc = 448
Lq = 48

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

def mask_logits(inputs, mask):
    mask = mask.float()
    #凡是pad的位置
    return inputs + (-1e30) * (1 - mask)

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1,2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.cuda()).transpose(1,2)

def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))



class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem_conv = Initialized_Conv1d(D, D*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(D, D, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        #(batch_size,dim,length)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        #(batch_size,length,dim)
        Q = self.split_last_dim(query, Nh)
        #(batch_size,num_heads,length,dim)
        K, V = [self.split_last_dim(tensor, Nh) for tensor in torch.split(memory, D, dim=2)]

        key_depth_per_head = D // Nh
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q,k.permute(0,1,3,2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)
    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret

class EncoderBlock(nn.Module):
    def __init__(self, conv_num=4, ch_num=D, k=1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(D)
        self.norm_2 = nn.LayerNorm(D)
        self.conv_num = conv_num
    def forward(self, x, mask):
        #total_layers = (self.conv_num+1)*blks
        #x.size()==(batch_size,D,seq_length)
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            #out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
            #l += 1
        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        #out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        #l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        #out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        return out

    # def layer_dropout(self, inputs, residual, dropout):
    #     if self.training == True:
    #         pred = torch.empty(1).uniform_(0,1) < dropout
    #         if pred:
    #             return residual
    #         else:
    #             return F.dropout(inputs, dropout, training=self.training) + residual
    #     else:
    #         return inputs + residual

class Model(nn.Module):
    def __init__(self,config,pretrained_word_embedding=None):
        super(Model,self).__init__()
        self.args=config
        if pretrained_word_embedding is None:
            self.word_embedding=nn.Embedding(self.args.vocab_size,self.args.embed_dim,padding_idx=0)
            nn.init.uniform_(self.word_embedding.weight,-0.1,0.1)
        else:
            self.word_embedding=nn.Embedding.from_pretrained(pretrained_word_embedding,freeze=False)

        for i in range(self.args.highway_network_layers):
            setattr(self,"highway_linear{}".format(i),nn.Sequential(Linear(self.args.embed_dim,self.args.embed_dim),nn.ReLU()))
            setattr(self,"highway_gate{}".format(i),nn.Sequential(Linear(self.args.embed_dim,self.args.embed_dim),nn.Sigmoid()))

        self.encoder_project=Linear(in_features=self.args.embed_dim,out_features=D)
        self.contextTransformer=EncoderBlock(conv_num=4, ch_num=D, k=3)

        self.att_weight_context=Linear(in_features=D,out_features=1)
        self.att_weight_question=Linear(in_features=D,out_features=1)
        self.att_weight_cq=Linear(in_features=D,out_features=1)
        self.cq_projection=Linear(in_features=D*4,out_features=D)

        self.ModelingTransformer=EncoderBlock(conv_num=2,ch_num=D,k=1)
        self.ModelingTransformer2=EncoderBlock(conv_num=2,ch_num=D,k=1)

        self.p1_weight=Linear(in_features=self.args.hidden_size*2*4+self.args.hidden_size*2,out_features=1)
        self.p2_weight=Linear(in_features=self.args.hidden_size*2*4+self.args.hidden_size*2,out_features=1)
        self.temp_layer=nn.Linear(in_features=100,out_features=self.args.hidden_size*2)

    def highway_network(self,inputs):
        for i in range(self.args.highway_network_layers):
            #print(inputs.shape)
            h=getattr(self,'highway_linear{}'.format(i))(inputs)
            g=getattr(self,"highway_gate{}".format(i))(inputs)
            inputs=g*h+(1-g)*inputs
        return inputs#(batch_size,seq_length,hidden_size*2)


    def bi_attention_flow(self,context,question,context_mask=None,question_mask=None):
        '''
        按照了论文中的思想，先计算相似性矩阵S，计算方式为：
        context通过一个全连接，question通过一个全连接，然后context与question的点乘通过一个全连接
        这三个全连接的out_features都是１，也就是将每一个单词由原来的高维特征压缩为一个数字表示，这个数字代表的就是该单词的重要性

        S.shape==(batch_size,context_len,question_len)
        '''
        c_length=context.size(1)
        q_length=question.size(1)
        c_att=self.att_weight_context(context).expand(-1,-1,q_length)
        #shape-->(batch_size,context_len,dim)-->(batch_size,context_len,1)-->(batch_size,context_len,question_len)
        #将context_len长度的每一个单词扩展了question_len的长度，相当于告诉context的每一个单词question的长度
        q_att=self.att_weight_question(question).permute(0,2,1).expand(-1,c_length,-1)
        #shape-->(batch_size,question_len,dim)-->(batch_size,question_len,1)-->(batch_size,1,question_len)-->(batch_size,context_len,question_len)

        #接下来就算question与context的点乘，做法是对于question的每一个单词，将该单词的向量与context矩阵做点乘，
        #具体的就是是利用该单词的向量与context矩阵的每一个单词的向量做点乘，注意点乘不是向量乘法，是逐个元素相乘。
        cq=[]
        for i in range(q_length):
            q_i=question.select(1,i).unsqueeze(1)#在question_len这个维度上选择第i个单词，形状为(batch_size,hidden_size*2)
            #,q_i.shape==(batch_size,1,hidden_size*2)
            elementwise_mul=context*q_i#(batch_size,context_len,hidden_size*2)
            cq.append(self.att_weight_cq(elementwise_mul).squeeze())#(batch_size,context_len)
        #cq有q_length个张量，每一个tensor的shape==(batch_size,context_len)
        cq_att=torch.stack(cq,dim=-1)#(batch_size,context_len,question_len)
        S=c_att+q_att+cq_att#(batch_size,context_len,question_len)

        C2Q=torch.bmm(F.softmax(S,dim=2),question)

        S_=torch.max(S,dim=2)[0]#S_.shape==(batch_size,context_length)
        b=F.softmax(S_,dim=1).unsqueeze(1)
        #b.shape==(batch_size,1,context_len)
        Q2C=torch.bmm(b,context)#(batch_size,1,hidden_size*2)
        Q2C=Q2C.expand(-1,c_length,-1)#(batch_size,context_len,hidden_size*2)
        assert Q2C.shape==C2Q.shape==context.shape
        G=torch.cat([context,C2Q,context*C2Q,context*Q2C],dim=2)#(batch_size,context_len,hidden_size*2*4)
        return G

    def output_layer(self,G,M,M2):
        start_logits=self.p1_weight(torch.cat([G,M],dim=2)).squeeze()
        #(batch_size,context_len)
        end_logits=self.p2_weight(torch.cat([G,M2],dim=2)).squeeze()
        #(batch_size,context_len)
        return start_logits,end_logits

    def forward(self,context_ids,question_ids):
        context_ids=context_ids.to(device)
        question_ids=question_ids.to(device)
        context_mask=(torch.zeros_like(context)!=context).float()#凡是pad的位置都是0
        question_mask=(torch.zeros_like(question)!=question).float()

        context_embeddings=self.word_embedding(context_ids)
        question_embeddings=self.word_embedding(question_ids)
        #print("embeddigs.shape : ",context_embeddings.shape,question_embeddings.shape)
        context=self.highway_network(context_embeddings)
        question=self.highway_network(question_embeddings)
        #(batch_size,seq_length,embed_dim)
        context=self.encoder_project(context).transpose(2,1)
        question=self.encoder_project(question).transpose(2,1)
        #(batch_size,D,seq_length)


        context=self.contextTransformer(context,context_mask)
        question=self.contextTransformer(question,question_mask)
        #num_layers=1,bidirectional=True (batch_size,seq_length,D)
        G=self.bi_attention_flow(context=context,question=question,context_mask=context_ids.bool(),question_mask=question_ids.bool())
        #(batch_size,context_len,D*4)
        G_=self.cq_projection(G)
        M=self.ModelingTransformer(G_,context_mask)
        M2=self.ModelingTransformer2(M,context_mask)

        start_logits,end_logits=self.output_layer(G=G,M=M,M2=M2)
        return start_logits,end_logits

