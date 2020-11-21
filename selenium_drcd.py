from selenium import webdriver
import json,os
import win32clipboard
import win32con
import time
from selenium.webdriver.common.keys import Keys
browser=webdriver.Chrome()
browser.get("http://www.fantiz5.com/")
browser.maximize_window()

win32clipboard.OpenClipboard()
win32clipboard.CloseClipboard()#保证剪贴板中的内容为空

input_window=browser.find_element_by_id("in")#获得输入框
change_to_simple_button=browser.find_element_by_xpath('//*[@id="bu"][2]')#获得繁体字转简体字按钮


def get_text(text):
    input_window.clear()#清空输入框内容
    try:
        input_window.send_keys(text)#发送繁体字
        change_to_simple_button.click()#转换为简体字
        input_window.send_keys(Keys.CONTROL,'a')#全选输入框内容
        input_window.send_keys(Keys.CONTROL,"c")#复制
        time.sleep(1)
        win32clipboard.OpenClipboard()#打开剪贴板
        simple_text=win32clipboard.GetClipboardData(win32con.CF_TEXT)#获得剪贴板中复制的内容
        win32clipboard.CloseClipboard()
        return simple_text.decode("GBK")
    except:
        print("出现异常")
        return None


def get_simple_DRCD(file_path,all_json_file,write_path):
    bad_context=0
    total_context=0
    except_nums=0
    f_write=open(write_path,"w",encoding="utf-8")
    for each_json_file in all_json_file:
        with open(os.path.join(file_path,each_json_file),encoding="utf-8") as f:
            test_data=json.load(f)["data"]
        simple_text_examples=[]
        for each_example in test_data:
            paragraphs=each_example["paragraphs"]
            for each_qas in paragraphs:
                context=each_qas["context"]
                simple_context=get_text(text=context)
                if simple_context==None:
                    except_nums+=1
                    continue
                qas=each_qas["qas"]
                for each_qa in qas:
                    question=each_qa["question"]
                    simple_question=get_text(text=question)
                    if simple_question==None:
                        except_nums+=1
                        continue
                    answer=each_qa["answers"][0]
                    answer_text=answer["text"]
                    simple_answer=get_text(text=answer_text)
                    if simple_answer==None:
                        except_nums+=1
                        continue
                    answer_start=answer["answer_start"]
                    total_context+=1
                    try:
                        assert simple_context.find(simple_answer)==answer_start
                    except:
                        #现在出现了context中有多个位置出现了答案中的单词，但是这些位置都不是答案真正应该在的位置
                        bad_context+=1
                        continue
                    each_example={"context":simple_context,"question":simple_question,
                                                 "answer":{"text":simple_answer,"start_position":answer_start}}
                    f_write.write(json.dumps(each_example,ensure_ascii=False)+"\n")
    f_write.close()

        
    print("total examples get from DRCD : %d , bad examples in DRCD : %d "%(total_context,bad_context))
    print("Exception example nums : ",except_nums)
    
if __name__=="__main__":
    file_path="C:\\Users\\Tony Sun\\Desktop\\Chinese MRC Data\\drcd"
    all_json_file=os.listdir(file_path)
    write_path="C:\\Users\\Tony Sun\\Desktop\\Chinese MRC Data\\my_drcd.json"
    get_simple_DRCD(file_path=file_path,all_json_file=all_json_file,write_path=write_path)
    
    