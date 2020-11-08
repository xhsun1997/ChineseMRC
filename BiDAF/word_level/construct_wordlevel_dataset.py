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
    return {"context":context_word_list,"question":list(jieba.cut(question)),"answer":{"text":answer_word_list,"start_position":word_start_pos,"end_position"}}

examples=[]
data_folder="./data"
files_name=["my_cmrc2018.json","my_dureader.json","my_military.json","my_cail2019.json"]
bad_examples=0
for file_name in files_name:
    file_=os.path.join(data_folder,file_name)
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

random.shuffle(examples)
all_example_nums=len(examples)
print(all_example_nums)
train_examples=examples[:int(all_example_nums*0.8)]
test_examples=examples[int(all_example_nums*0.8):]

print(len(train_examples),len(test_examples))
with open("./data/train.json","w") as f:
    for example in train_examples:
        f.write(json.dumps(example,ensure_ascii=False)+"\n")

print("Train dataset has been constructed!")
with open("./data/test.json","w") as f:
    for example in test_examples:
        f.write(json.dumps(example,ensure_ascii=False)+"\n")
print("Test dataset has been constructed!")