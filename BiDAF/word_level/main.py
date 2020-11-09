import os
import argparse
import torch
import torch.nn as nn
from BiDAF import BiDAF
from data_word_process import *
from utils import get_metrics
import pickle
from tensorboardX import SummaryWriter
import random

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ",device)

#sed -n "line1,line2p" file | tee target_file
def train(config,train_dataloader,test_dataloader,eval_examples):
    writer=SummaryWriter(log_dir=config.log_dir)
    model=BiDAF(config)
    model.load_state_dict(torch.load("/home/sun_xh/xhsun/ChineseMRC/saved_best_model/model562.bin"))
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",factor=0.8,patience=3,verbose=True)
    #patience是指，如果第四个epochs都没有下降的话，那么new_lr=factor*orig_lr
    criterion=nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    loss=0
    best_f1_score=0
    training_loss_record=[]
    test_loss_record=[]
    test_f1_record=[]
    test_em_record=[]

    for epoch in range(config.epochs):

        for step,batch_data in enumerate(train_dataloader):
            batch_qa_ids,context_ids,question_ids,start_positions,end_positions=batch_data
            start_logits,end_logits=model(context_ids,question_ids)
            (batch_size,context_actual_length)=start_logits.size()
            # try:
            #     assert context_actual_length max(end_positions.tolist())
            optimizer.zero_grad()

            batch_loss=0.5*criterion(start_logits,start_positions.to(device))+0.5*criterion(end_logits,end_positions.to(device))
            loss+=batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            #if (step+1)%config.gradient_accumulation_steps==0:
            if (step+1)%config.print_loss_step==0:
                print("epoch : %d , step : %d , loss %.3f"%(epoch,step,loss/(config.print_loss_step)))
                training_loss_record.append(loss/config.print_loss_step)
                writer.add_scalar("loss/train",loss/config.print_loss_step,(step+1)//config.print_loss_step)
                loss=0

            if (step+1)%config.test_step==0:
                test_loss,f1_score,em_value=test(model,config,test_dataloader,eval_examples)

                writer.add_scalar("loss/test",test_loss,(step+1)//config.test_step)
                writer.add_scalar("em/test",em_value,(step+1)//config.test_step)
                writer.add_scalar("f1/test",f1_score,(step+1)//config.test_step)

                test_loss_record.append(test_loss)
                test_f1_record.append(f1_score)
                test_em_record.append(em_value)

                scheduler.step(test_loss)#监控test_loss
                print("epoch : %d , step : %d , learning rate : %f , test loss : %.3f , f1_score : %.3f , em_value : %.3f"%(epoch,step,optimizer.param_groups[0]["lr"],test_loss,f1_score,em_value))
                if f1_score>best_f1_score:
                    output_model_file=os.path.join(config.save_model_path,
                                                "model"+str(f1_score).split(".")[1][:3]+".bin")
                    torch.save(model.state_dict(),output_model_file)
                    print("Save model to ",output_model_file)
                    best_f1_score=f1_score
                model.train()
    with open("./log.pkl","wb") as f:
        obj=[training_loss_record,test_loss_record,test_f1_record,test_em_record]
        pickle.dump(obj,f)
    writer.close()


def test(model,config,test_dataloader,eval_examples):
    criterion=nn.CrossEntropyLoss()
    model.eval()
    loss=0
    total_ems=[]
    total_f1_scores=[]
    with torch.no_grad():
        for step,batch_data in enumerate(test_dataloader):
            batch_qa_ids,context_ids,question_ids,start_positions,end_positions=batch_data
            start_logits,end_logits=model(context_ids,question_ids)

            #ignore_index=start_logits.size(1)
            #start_positions.clamp_(0,ignore_index)
            #end_positions.clamp_(0,ignore_index)
            #start_loss=nn.CrossEntropyLoss(ignore_index=ignore_index)(start_logits,start_positions.to(device))
            #end_loss=nn.CrossEntropyLoss(ignore_index=ignore_index)(end_logits,end_positions.to(device))
            start_loss=criterion(start_logits,start_positions.to(device))
            end_loss=criterion(end_logits,end_positions.to(device))
            loss+=(0.5*start_loss+0.5*end_loss).item()

            batch_size,context_len=start_logits.size()
            log_softmax=nn.LogSoftmax(dim=1)#在context_len维度上做log_softmax
            mask=(torch.ones(context_len,context_len)*float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size,-1,-1)
            #mask是batch_size个形状为(context_len,context_len)的下三角矩阵，对角线以及上三角部分为0
            score=log_softmax(start_logits).unsqueeze(2)+log_softmax(end_logits).unsqueeze(1)+mask

            # score,start_predictions=score.max(dim=1)
            # score,end_predictions=score.max(dim=1)
            start_pred_ids=torch.argmax(torch.max(score,dim=2)[0],dim=1).tolist()
            end_pred_ids=torch.argmax(torch.max(score,dim=1)[0],dim=1).tolist()

            batch_ems,batch_f1_scores=get_metrics(start_pred_ids,end_pred_ids,batch_qa_ids,eval_examples)
            #batch_ems记录着batch_size个样本的em值，batch_f1_scores记录着batch_size个样本的f1值
            total_ems.extend(batch_ems)
            total_f1_scores.extend(batch_f1_scores)

    f1_score=sum(total_f1_scores)/len(total_f1_scores)
    em_value=sum(total_ems)/len(total_ems)
    #print("test f1_score : %.3f , test em value : %.3f"%(f1_score,em_value))
    return loss/step,f1_score,em_value


class Config:
    def __init__(self):
        self.train_question_len_limit=48
        self.train_context_len_limit=448
        self.test_question_len_limit=48
        self.test_context_len_limit=512
        #self.vocab_size=len(word2id)
        self.embed_dim=300
        self.highway_network_layers=2
        self.hidden_size=150
        self.dropout=0.2
        self.learning_rate=0.001
        self.epochs=20
        self.gradient_accumulation_steps=1
        self.print_loss_step=100
        self.data_folder="/home/sun_xh/xhsun/ChineseMRC/data"
        self.files_name=["my_cmrc2018.json","my_dureader.json","my_military.json","my_cail2019.json"]
        self.train_batch_size=48
        self.test_batch_size=48
        self.test_step=300
        self.save_model_path="/home/sun_xh/xhsun/ChineseMRC/saved_best_model/"
        self.log_dir="/home/sun_xh/xhsun/ChineseMRC/log_dir/"
        self.train_file_path="/home/sun_xh/xhsun/ChineseMRC/data/train_wordlevel.json"
        self.test_file_path="/home/sun_xh/xhsun/ChineseMRC/data/test_wordlevel.json"

    def add_vocab_size(self,vocab_size):
        self.vocab_size=vocab_size
def main():
    config=Config()
    
    train_examples,word2id,train_eval_examples=read_data(config.train_file_path,max_context_length=config.train_context_len_limit)#我们根据qa_id就可以得到对应的example
    config.add_vocab_size(len(word2id))
    print("length of word2id : ",len(word2id))
    test_examples,_,test_eval_examples=read_data(config.test_file_path,max_context_length=config.test_context_len_limit)

    train_features=convert_to_ids(train_examples,word2id)
    test_features=convert_to_ids(test_examples,word2id)
    #convert_to_ids并不改变examples和features之间的关系
    #pad_features可能会过滤掉一些example，但是我们利用unique_qa_id记住了每一个qa的id，
    #所以features里面的unique_qa_id和examples里面的unique_qa_id是可以对应的
    #只不过len(features)<=len(examples)

    train_features=pad_features(train_features,config)
    test_features=pad_features(test_features,config,is_train=False)

    print("Number of train : %d , number of test : %d "%(len(train_features),len(test_features)))

    train_dataloader=get_dataloader(train_features,batch_size=config.train_batch_size)
    test_dataloader=get_dataloader(test_features,batch_size=config.test_batch_size,is_train=False)

    train(config,train_dataloader=train_dataloader,test_dataloader=test_dataloader,eval_examples=test_eval_examples)


main()



