import os
import random
import json

import jieba
import json
import tqdm
def get_word_level_example(file_):
    examples=[]
    data=json.load(open(file_))
    for example in tqdm.tqdm(data):
        context=example["context_text"]
        qas=example["qas"]
        context_word_list=list(jieba.cut(context))
        word_to_char_index=[]
        for i,word in enumerate(context_word_list):
            for char in word:
                word_to_char_index.append(i)#
        for qa in qas:
            question=qa["query_text"]
            answer=str(qa["answers"][0])
            assert context.find(answer)!=-1
            start_position=context.find(answer)
            end_position=start_position+len(answer)
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
                    continue
            examples.append({"context":context_word_list,"question":list(jieba.cut(question)),
                        "answer":{"text":answer_word_list,"start_position":word_start_pos,
                                "end_position":word_end_pos,"original_text":answer}})
    return examples

train_file="./data/cmrc2018_train.json"
test_file="./data/cmrc2018_dev.json"


train_examples=get_word_level_example(file_=train_file)
test_examples=get_word_level_example(file_=test_file)

print(len(train_examples),len(test_examples))
with open("./data/train_wordlevel.json","w") as f:
    for example in train_examples:
        f.write(json.dumps(example,ensure_ascii=False)+"\n")

print("Train wordlevel dataset has been constructed!")
with open("./data/test_wordlevel.json","w") as f:
    for example in test_examples:
        f.write(json.dumps(example,ensure_ascii=False)+"\n")
print("Test wordlevel dataset has been constructed!")
