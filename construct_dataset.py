import os
import random
import json

examples=[]
data_folder="./data"
files_name=["my_cmrc2018.json","my_dureader.json","my_military.json","my_cail2019.json"]
for file_name in files_name:
    file_=os.path.join(data_folder,file_name)
    with open(file_,encoding="utf-8") as f:
        lines=f.readlines()
    for line in lines:
        examples.append(json.loads(line.strip()))


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