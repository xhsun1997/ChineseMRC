import os,copy,json,math,logging,tqdm,argparse
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertModel,PreTrainedModel,BertPreTrainedModel

import transformers
from transformers.tokenization_bert import BertTokenizer
import random
import numpy as np

'''
BertModel:
 1. inherited from BertPreTrainedModel
 2. init(config)-->embeddings=BertEmbedding(config) encoder=BertEncoder(config) pooler=BertPooler(config)
 3. forward(input_ids,attention_mask,token_type_ids,position_ids,return_dict=None):
     sequence_output=encoder_outputs[0]
     pooled_output=self.pooler(sequence_output)#特别注意这里，sequence_output[0]与pooled_output的区别
 4. return (sequence_output,pooled_output)
 

BertPreTrainedModel inherited from PreTrainedModel
PreTrainedModel:
 1 init(config,**kwargs)
 2 @classmethod from_pretrained(cls,pretrained_model_name_or_path,**kwargs):
     (1)调用位置必须放在model.train()之后，因为默认预训练模型的状态是model.eval()
     (2)pretrained_model_name_or_path指的是pytorch_model.bin存放的文件夹的位置，不是pytorch_model.bin的位置
'''


class BertNLI(BertPreTrainedModel):
    def __init__(self,config):
        super(BertNLI,self).__init__(config)
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.outputs_layer=nn.Linear(config.hidden_size,2)
        self.config=config
        self.init_weights()#init_weights inherited from BertPreTrainedModel
        #module.weight.data.normal_(mean=0.0,std=config.initializer_range)
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,label_ids=None):
        '''
        input_ids.size()==attention_mask.size()==token_type_ids.size()==position_ids.size()==(batch_size,seq_length)
        label_ids.size()==(batch_size,)
        '''
        (sequence_outputs,pooled_output)=self.bert(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        #要注意到sequence_output[0]与pooled_output的区别在于pooled_output是经过一层tanh的
        assert len(pooled_output.size())==2 and pooled_output.size(1)==self.config.hidden_size
        logits=self.outputs_layer(pooled_output)#(batch_size,2)
        predictions=torch.argmax(logits,dim=1)
        if label_ids is not None:
            loss=CrossEntropyLoss(reduction="mean")(input=logits,target=label_ids)
            accuracy=(predictions==label_ids).float().mean()
            return loss,accuracy
        else:
            return logits



class NLIFeature:
    def __init__(self,tokens,input_ids,input_mask,segment_ids,label_ids=None):
        self.tokens=tokens
        self.input_ids=input_ids
        self.input_mask=input_mask
        self.segment_ids=segment_ids
        self.label_ids=label_ids
        
class NLIExample:
    def __init__(self,context,question,label=None):
        self.context=context
        self.question=question
        self.label=label

def read_nli_example(input_file,is_training=True):
    with open(input_file) as f:
        lines=f.readlines()
    examples=[]
    for example_index,line in enumerate(lines):
        line_split=line.strip().split('\t')#数据集中含有"\n"和"\t"的时候显然会出现问题
        if len(line_split)!=3:
            #print(len(line_split))
            continue
        context=line_split[0]
        question=line_split[1]
        if is_training:
            label=int(line_split[2])
        else:
            label=0
        example=NLIExample(context=context,question=question,label=label)
        examples.append(example)
    logging.info(" Total raw examples : ",len(examples))
    return examples
def convert_examples_to_features(examples,tokenizer,max_seq_length,is_training=True):
    #tokenizer=transformers.tokenization_bert.BertTokenizer.from_pretrained("../")
    features=[]
    deprecated_example_nums=0
    for example_index,example in enumerate(examples):
        context=example.context
        question=example.question
        label=example.label
        tokens=['[CLS]']
        segment_ids=[0]
        for token in context:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)
        for token in question:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)
        #'[CLS]'对应101,'[SEP]'对应102,'[UNK]'对应100 '[PAD]'对应0
        input_ids=tokenizer.convert_tokens_to_ids(tokens)
        if len(input_ids)>=max_seq_length:
            deprecated_example_nums+=1
            continue
        input_mask=[1]*len(input_ids)
        needed_pad_length=max_seq_length-len(input_ids)
        if needed_pad_length>0:
            input_ids+=[0]*needed_pad_length
            input_mask+=[0]*needed_pad_length
            segment_ids+=[0]*needed_pad_length
        features.append(NLIFeature(tokens=tokens,input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids,label_ids=int(label)))
        if example_index<=5:
            print("tokens : ",tokens)
            print("input_ids : ",input_ids)
            print("input_mask : ",input_mask)
            print("segment_ids : ",segment_ids)

    return features

def train_model(args,model,optimizer,train_features,dev_features,device,t_total):
    all_input_ids=torch.tensor([f.input_ids for f in train_features],dtype=torch.long)
    all_input_mask=torch.tensor([f.input_mask for f in train_features],dtype=torch.long)
    all_segment_ids=torch.tensor([f.segment_ids for f in train_features],dtype=torch.long)
    all_label_ids=torch.tensor([f.label_ids for f in train_features],dtype=torch.long)
    train_data=torch.utils.data.TensorDataset(all_input_ids,all_input_mask,all_segment_ids,all_label_ids)
    train_sampler=torch.utils.data.RandomSampler(train_data)
    train_dataloader=torch.utils.data.DataLoader(train_data,sampler=train_sampler,batch_size=args.train_batch_size)
    
    if args.do_predict:
        all_test_input_ids=[f.input_ids for f in dev_features]
        all_test_input_mask=[f.input_mask for f in dev_features]
        all_test_segment_ids=[f.segment_ids for f in dev_features]
        all_test_label_ids=[f.label_ids for f in dev_features]
        
        dev_data=torch.utils.data.TensorDataset(all_test_input_ids,all_test_input_mask,all_test_segment_ids,all_test_label_ids)
        dev_sampler=torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader=torch.utils.data.DataLoader(dev_data,sampler=dev_sampler,batch_size=args.test_batch_size)
        
    best_dev_acc=0.0
    epoch=0
    global_step=0
    model.train()
    for _ in range(int(args.num_train_epochs)):
        training_loss=0.0
        for step,batch in enumerate(tqdm.tqdm(train_dataloader)):
            batch=tuple(t.to(device) for t in batch)
            input_ids,input_mask,segment_ids,label_ids=batch
            #print(input_ids.device,input_mask.device)
            loss,accuracy=model(input_ids,attention_mask=input_mask,token_type_ids=segment_ids,label_ids=label_ids)
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps
            loss.backward()
            training_loss+=loss.item()
            if (step+1)%args.print_loss_step==0:
                print("step: {} , average loss: {} , accuracy : {}".format(step,
                                                                           training_loss/args.print_loss_step,
                                                                          accuracy))
                training_loss=0.0
            if args.do_predict and (step+1)%args.predict_step==0:
                model.eval()
                dev_loss,dev_acc=test(args,model,dev_features,device)
                print("epoch : {} , dev loss : {} , dev accuarcy : {}".format(epoch+1,dev_loss,dev_acc))
                if dev_acc>best_dev_acc:
                    best_dev_acc=dev_acc
                    output_model_file=os.path.join(args.output_dir,"best_model.bin")
                    torch.save(model.state_dict(),output_model_file)
                    print("Save model to ",output_model_file)
                model.train()
                
            if (step+1)%args.gradient_accumulation_steps==0:
                #lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group["lr"]*=0.8
                optimizer.step()
                optimizer.zero_grad()
                global_step+=1
        epoch+=1

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="/home/xhsun/Desktop/chinese_pytorch/", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default="/home/xhsun/Desktop/ChineseMRC/TextMatch/textmatch.txt", type=str, help="triviaqa train file")
    parser.add_argument("--test_file", default="/home/xhsun/Desktop/ChineseMRC/TextMatch/", type=str, help="triviaqa train file")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, help="Whether to run validation when training")
    # model parameters
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--print_loss_step", default=300, type=int, help="Total batch size for predictions.")
    parser.add_argument("--predict_step", default=1000, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : ",device)
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer=BertTokenizer.from_pretrained(args.bert_model)
    model=BertNLI.from_pretrained(args.bert_model)

    model.to(device)
    optimizer=torch.optim.Adam(params=model.parameters(),lr=args.learning_rate)
    all_examples=read_nli_example(input_file=args.train_file)
    random.shuffle(all_examples)
    random.shuffle(all_examples)
    used_examples=all_examples[:20000]
    train_nums=int(len(used_examples)*0.8)
    train_examples=used_examples[:train_nums]
    train_features=convert_examples_to_features(examples=train_examples,tokenizer=tokenizer,max_seq_length=args.max_seq_length)
    #dev_examples=read_nli_example(input_file=args.test_file)
    #dev_features=convert_examples_to_features(examples=dev_examples,tokenizer=tokenizer,max_seq_length=args.max_seq_length)
    dev_examples=used_examples[train_nums:]
    dev_features=convert_examples_to_features(examples=dev_examples,tokenizer=tokenizer,max_seq_length=args.max_seq_length)

    num_train_steps=int(len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    t_total=num_train_steps
    if args.do_train:
        train_model(args,model,optimizer,train_features,dev_features,device,t_total)

    
main()
