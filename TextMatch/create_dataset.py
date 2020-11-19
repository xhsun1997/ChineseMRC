import re
import json
import os
import numpy as np
import tqdm
from copy import deepcopy
import random

all_data_dir="/home/xhsun/Desktop/ChineseMRC/mrc_data"

all_files=os.listdir(all_data_dir)

def get_examples():
    examples=[]
    for file_ in tqdm.tqdm(all_files):
        file_name=os.path.join(all_data_dir,file_)
        with open(file_name,encoding="utf-8") as f:
            lines=f.readlines()
        for line in lines:
            mrc_example=json.loads(line.strip())#{"context","question","answer"}
            context=mrc_example["context"]
            question=mrc_example["question"]
            answer=str(mrc_example["answer"]["text"])
            answer_start_pos=mrc_example["answer"]["start_position"]
            if context.find(answer)==-1 or context.find(answer)!=answer_start_pos:
                continue
            assert type(context)==type(question)==str
            nli_example=[context,question,answer,1]
            examples.append(nli_example)
    print("Positive nli example nums %d "%len(examples))
    return examples

def get_random_number(num_positive_examples,example_index):
    random_number=random.randint(a=0,b=num_positive_examples-1)
    while random_number==example_index:
        random_number=random.randint(a=0,b=num_positive_examples-1)
    assert random_number!=example_index
    return random_number

def construct_neg_examples(all_positive_examples,my_define_number=3):
    '''
    1 去掉答案所在的句子，构成一个负样本
    2 随机选取其它的文档中的一个句子，替换掉答案所在地句子，构成3个负样本
    3 随机选取一篇文档，构造一个负样本
    
    随机的删除掉结尾，或者开头，总之距离答案所在的句子间隔1-2个句子的句子都可以删除，构造成正样本
    '''
    all_negative_examples=[]
    num_positive_examples=len(all_positive_examples)
    bad_examples=0
    for example_index,pos_example in enumerate(all_positive_examples):
        assert len(pos_example)==4
        #pos_example[0]==context [1]=question [2]=answer [3]=label 1
        context=pos_example[0]
        question=pos_example[1]
        answer=pos_example[2]
        sentence_list=context.split('。')#将整个文本段落分割为各个句子
        ans_pos_sentence=-1
        for i,sentence in enumerate(sentence_list):
            if sentence.find(answer)!=-1:
                ans_pos_sentence=i
                break
                
        
        if(ans_pos_sentence<0):
            bad_examples+=1
            continue
        #现在找到了答案所在的句子
        temp_sentence_list=deepcopy(sentence_list)#一定要deepcopy
        
        del temp_sentence_list[ans_pos_sentence]#删除掉答案所在的句子构造一个负样本
        

        all_negative_examples.append(["".join(temp_sentence_list),question,answer,0])#仍然保留answer是为了与all_positive_examples一致
        
        random_number=get_random_number(num_positive_examples,example_index=example_index)
        random_example=all_positive_examples[random_number]#得到一个随机的样本
        all_negative_examples.append([random_example[0],question,random_example[2],0])#随机产生的context与当前的question构成一个负样本
        
        #接下来就是随机产生几个句子，替换掉sentence_list中答案所在的句子，构造3个负样本
        for _ in range(my_define_number):
            random_number=get_random_number(num_positive_examples,example_index=example_index)
            random_example=all_positive_examples[random_number]#得到一个随机的样本
            random_example_sentence_list=random_example[0].split('。')
            random_sentence_number=random.randint(a=0,b=len(random_example_sentence_list)-1)
            random_sentence=random_example_sentence_list[random_sentence_number]#拿到一个随机的句子
            temp_sentence_list=sentence_list
            temp_sentence_list[ans_pos_sentence]=random_sentence
            all_negative_examples.append(["".join(temp_sentence_list),question,answer,0])
        #现在构造好了5个负样本
    for i,each_neg_example in enumerate(all_negative_examples):
        context=each_neg_example[0]
        question=each_neg_example[1]
        answer=each_neg_example[2]
        if context.find(answer)!=-1:
            del all_negative_examples[i]
    
    print("Negative nli example nums : ",len(all_negative_examples))
    print("bad example nums : ",bad_examples)
    return all_negative_examples


def get_pseudo_pos_examples(all_nli_examples,interval_distance=3):
    '''
    对于每一个正样本来说，假如该样本的文本特别长，那么完全可以删除距离答案句子间隔有3个句子以上的句子
    '''
    all_pseudo_examples=[]
    bad_examples=0
    for example_index,example in enumerate(all_nli_examples):
        context=example[0]
        question=example[1]
        answer=example[2]
        
        sentence_list=context.split('。')
        ans_pos_sentence=-1
        for i,sentence in enumerate(sentence_list):
            if sentence.find(answer)!=-1:
                ans_pos_sentence=i
                break
        if(ans_pos_sentence<0):
            bad_examples+=1
            continue
        sentence_nums=len(sentence_list)
        #现在我们知道答案位置在整个context中位于ans_pos_sentence，而且整个context有sentence_nums个句子
        if sentence_nums>3 and sentence_nums<=5:
            #说明这个context has 4-5 sentence
            if ans_pos_sentence<=1:
                #答案所在的句子位于前半部分，那我们就删除后半部分的两个句子
                temp_list=deepcopy(sentence_list)
                del temp_list[-1]# delete last sentence 
                all_pseudo_examples.append(["".join(temp_list),question,answer,1])
            if ans_pos_sentence>=4:
                temp_list=deepcopy(sentence_list)
                del temp_list[0]#delete first sentence
                all_pseudo_examples.append(["".join(temp_list),question,answer,1])

                
        if sentence_nums>5 and abs(ans_pos_sentence-sentence_nums)<interval_distance:
            #说明这个context至少有6个句子，而且答案所在的句子要么是前3个句子中的某一个，要么是后三个句子中的某一个
            if ans_pos_sentence<sentence_nums//2-1:
                #答案所在的句子位于前半部分，那我们就删除后半部分的两个句子
                temp_list=deepcopy(sentence_list)
                del temp_list[-2:]
                all_pseudo_examples.append(["".join(temp_list),question,answer,1])
            elif ans_pos_sentence>sentence_nums//2+1:
                temp_list=deepcopy(sentence_list)
                del temp_list[:2]#删除first two sentence
                all_pseudo_examples.append(["".join(temp_list),question,answer,1])
            else:
                temp_list=deepcopy(sentence_list)
                del temp_list[0]
                del temp_list[-1]#del first and last sentence when answer in the center of context
                all_pseudo_examples.append(["".join(temp_list),question,answer,1])                
                
    for i,each_pseudo_example in enumerate(all_pseudo_examples):
        context=each_pseudo_example[0]
        answer=each_pseudo_example[2]
        if context.find(answer)==-1:
            del all_pseudo_examples[i]
    
    print("Pesudo example nums : ",len(all_pseudo_examples))
    print("bad example nums : ",bad_examples)
    return all_pseudo_examples


def main(nli_file):
    all_nli_examples=get_examples()
    all_negative_examples=construct_neg_examples(all_positive_examples=deepcopy(all_nli_examples))
    all_pesudo_examples=get_pseudo_pos_examples(deepcopy(all_nli_examples))
    
    all_examples=all_nli_examples+all_pesudo_examples+all_negative_examples
    f=open(nli_file,"w",encoding="utf-8")
    for example in all_examples:
        context=re.sub("[\t\r\n]","",example[0])
        question=re.sub("[\t\r\n]","",example[1])
        answer=example[2]
        label=str(example[3])
        f.write(context+"\t"+question+"\t"+label+"\n")
    f.close()

main(nli_file="./textmatch.txt")
