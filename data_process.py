import os,json
import numpy as np
import torch
from collections import Counter

def read_data(json_files,max_context_length,doc_stride=128,word_limit_freq=10):
    if type(json_files)!=list:
        json_files=[json_files]
    examples=[]
    eval_examples={}
    all_words=[]
    unk_example_nums=0
    unique_qa_id=0
    for json_file in json_files:
        with open(json_file,encoding="utf-8") as f:
            lines=f.readlines()
        for line in lines:
            example=json.loads(line.strip())
            if len(example["context"])<=1 or len(example["question"])<=1:
                print("bad example : ",example)
                continue

            if example["answer"]["text"]=="null" or example["answer"]["start_position"]==-1:
                #example["answer"]['end_position']=-1
                unk_example_nums+=1
                continue#最初不考虑无答案问题，循序渐进
            else:
                example["answer"]["end_position"]=len(str(example["answer"]["text"]))+example["answer"]["start_position"]
            
            context=example["context"]
            context_length=len(context)
            answer_text=example["answer"]["text"]
            start_position=example["answer"]["start_position"]
            end_position=example["answer"]["end_position"]
            question=example["question"]
            
            for word in context:
                all_words.append(word)
            all_words.append(char for char in example["context"])
            
            ##########################切割context######################################
            split_context_list=[]#记录每一个分割的context的字符串
            split_answerspan_list=[]#记录每一个分割的context对应的answer_span
            if len(context)>=max_context_length:
                start_offset=0
                while True:
                    length=context_length-start_offset
                    #print(length,start_offset,max_context_length,context_length)
                    if length>=max_context_length:
                        length=max_context_length
                    
                    split_context=context[start_offset:(start_offset+length)]
                    start_pos=split_context.find(str(answer_text))
                    end_pos=start_pos+len(str(answer_text))
                    if start_pos==-1 or end_pos>len(split_context):
                        end_pos=-1
                    #assert start_pos!=-1 and end_pos<len(split_context)
                    if start_pos!=-1 and end_pos!=-1:
                        split_context_list.append(split_context)
                        split_answerspan_list.append((start_pos,end_pos))
                    if (start_offset+length)==context_length:
                        break
                    start_offset+=min(doc_stride,length)
            ###########################################################################
            else:
                split_context_list.append(context)
                split_answerspan_list.append((start_position,end_position))
            
            for each_context,corespond_answer_span in zip(split_context_list,split_answerspan_list):
                (start_pos,end_pos)=corespond_answer_span
                assert start_pos==each_context.find(str(answer_text))
                new_example={"unique_qa_id":unique_qa_id,"context":each_context,"question":question,
                             "answer":{"text":answer_text,"start_position":start_pos,"end_position":end_pos}}
                examples.append(new_example)
                eval_examples[str(unique_qa_id)]={"context":each_context,
                                                "spans":[start_pos,end_pos],
                                                "answer_text":answer_text}
                unique_qa_id+=1
    counter_words=Counter(all_words)
    sorted_words=sorted(counter_words.items(),key=lambda x:x[1],reverse=True)
    word2id={"[PAD]":0,"[UNK]":1}
    for word,word_freq in sorted_words:
        if word_freq>word_limit_freq:
            word2id[word]=len(word2id)
    print("unknown question examples : %d , not unk question examples : %d"%(unk_example_nums,len(examples)))
    return examples,word2id,eval_examples

def convert_to_ids(examples,word2id):
    features=[]
    def get_word_id(word):
        if word in word2id:
            return word2id[word]
        else:
            return word2id["[UNK]"]

    for example in examples:
        qa_id=example["unique_qa_id"]
        context=example["context"]
        question=example["question"]
        context_ids=[get_word_id(word) for word in context]
        question_ids=[get_word_id(word) for word in question]
        features.append({"unique_qa_id":qa_id,"context_ids":context_ids,"question_ids":question_ids,
                        "start_position":example["answer"]["start_position"],
                        "end_position":example["answer"]["end_position"]})
    return features

def convert_to_words(features,word2id):
    examples=[]
    id2word={id_:word for word,id_ in word2id.items()}
    if type(features)!=list:
        features=[features]
    for feature in features:
        qa_id=feature["unique_qa_id"]
        context_ids=feature["context_ids"]
        question_ids=feature["question_ids"]
        context=""
        for id_ in context_ids:
            context+=id2word[id_]
        question=""
        for id_ in question_ids:
            question+=id2word[id_]
        answer=context[feature["start_position"]:feature["end_position"]]
        examples.append({"unique_qa_id":qa_id,"context":context,"question":question,
                        "answer":{"text":answer,"start_position":feature["start_position"],"end_position":feature["end_position"]}})
    return examples


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
        unique_qa_id=feature["unique_qa_id"]

        if start_position>=context_len_limit or end_position>=context_len_limit:
            deprected_example_nums+=1
            continue

        context_ids=pad_ids(context_ids,context_len_limit)
        question_ids=pad_ids(question_ids,question_len_limit)
        new_features.append({"unique_qa_id":unique_qa_id,"context_ids":context_ids,"question_ids":question_ids,"start_position":start_position,"end_position":end_position})
    print("total feature nums : %d , deprected_example_nums : %d , rest features : %d"%(len(features),deprected_example_nums,len(new_features)))
    return new_features

def get_dataloader(features,batch_size,is_train=True):
    all_unique_qa_ids=torch.tensor([feature["unique_qa_id"] for feature in features],dtype=torch.long)
    all_context_ids=torch.tensor([feature["context_ids"] for feature in features],dtype=torch.long)
    all_question_ids=torch.tensor([feature["question_ids"] for feature in features],dtype=torch.long)
    all_start_positions=torch.tensor([feature["start_position"] for feature in features],dtype=torch.long)
    all_end_positions=torch.tensor([feature["end_position"] for feature in features],dtype=torch.long)

    dataset=torch.utils.data.TensorDataset(all_unique_qa_ids,all_context_ids,all_question_ids,all_start_positions,all_end_positions)
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

