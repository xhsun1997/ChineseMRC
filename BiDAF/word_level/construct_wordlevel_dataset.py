import os
import random
import json

import jieba
def get_word_level_example(example):
    #example.keys()=={"context","question","answer":{"text","start_position"}}
    context=example["context"]
    question=example["question"]
    answer=str(example["answer"]["text"])
    start_position=example["answer"]["start_position"]
    assert start_position==context.find(answer)
    end_position=start_position+len(answer)#注意，不需要加１
    context_word_list=list(jieba.cut(context))

    word_to_char_index=[]
    for i,word in enumerate(context_word_list):
        for char in word:
            word_to_char_index.append(i)#记录的是每一个字符是属于哪一个单词的

    answer_word_list=list(jieba.cut(answer))
    word_start_pos=word_to_char_index[start_position]
    word_end_pos=word_to_char_index[end_position-1]+1#不减1会出现IndexError，再加１的目的是定位到结束位置单词的后一个
    if answer_word_list!=context_word_list[word_start_pos:word_end_pos]:
        if answer in "".join(context_word_list[word_start_pos-1:word_end_pos]):
            word_start_pos=word_start_pos-1
        elif answer in "".join(context_word_list[word_start_pos:word_end_pos+1]):
            word_end_pos=word_end_pos+1
        elif answer in "".join(context_word_list[word_start_pos-1:word_end_pos+1]):
            word_start_pos-=1
            word_end_pos+=1
        else:
            #print(answer_word_list,context_word_list[word_start_pos:word_end_pos])
            return None
    return {"context":context_word_list,"question":list(jieba.cut(question)),
            "answer":{"text":answer_word_list,"start_position":word_start_pos,"end_position":word_end_pos,"original_text":answer}}

train_file="./data/train.json"
test_file="./data/test.json"

def get_examples(file_):
    examples=[]
    bad_examples=0
    with open(file_,encoding="utf-8") as f:
        lines=f.readlines()
    for line in lines:
        example=json.loads(line.strip())
        if example["answer"]["start_position"]==-1:
            continue
        processed_example=get_word_level_example(example)
        if processed_example!=None:
            examples.append(processed_example)
        else:
            bad_examples+=1
            continue
    print("bad exampple nums : ",bad_examples)
    print("total example nums : ",len(examples))
    return examples


train_examples=get_examples(file_=train_file)
test_examples=get_examples(file_=test_file)

print(len(train_examples),len(test_examples))
with open("./data/train_wordlevel.json","w") as f:
    for example in train_examples:
        f.write(json.dumps(example,ensure_ascii=False)+"\n")

print("Train wordlevel dataset has been constructed!")
with open("./data/test_wordlevel.json","w") as f:
    for example in test_examples:
        f.write(json.dumps(example,ensure_ascii=False)+"\n")
print("Test wordlevel dataset has been constructed!")