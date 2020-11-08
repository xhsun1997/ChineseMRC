from collections import Counter

def get_metrics(start_pred_ids,end_pred_ids,batch_qa_ids,eval_examples):
    '''
    将start_logits和end_logits经过log_softmax后，start_logits在最后扩维度然后与
    end_probs相加，得到的score，形状为(batch_size,context_len,context_len)
    score的第一个context_len指的是起始位置的概率，第二个context_len指的是终止位置的概率
    然后在score的下三角部分加上-inf，此时score的对角线以及上三角部分代表每一个单词作为答案的概率

    对score的第二个context_len上取最大，相当于找到每一行的最大值，那么该值对应的下标就是起始位置
    对score的第一个context_len取最大，相当于找到每一列的最大值，该值对应的下标就是终止位置

    于是得到了start_pred_ids,end_pred_ids，这两个都是list，长度为(batch_size,)

    batch_qa_ids代表的是这batch_size个样本的独有的qa_id，根据这个qa_id，就能从eval_examples[qa_id]
    中找到该样本的example和标签
    '''
    answer_dict={}
    em_values=[]
    f1_scores=[]
    for i,qa_id in enumerate(batch_qa_ids.tolist()):
        #这个qa_id对应的样本就是start_pred_ids[i]和end_pred_ids[i]对应的样本
        #迭代batch_size个样本的每一个example
        example=eval_examples[str(qa_id)]
        context=example["context"]#list
        spans=example["spans"]
        answer_text=example["answer_text"]#list
        start_ids,end_ids=spans#这个是真实标签

        predict_answer_text=context[start_pred_ids[i]:end_pred_ids[i]]#预测的list
        #这里需要注意的是，example中的终止位置=start_position+len(answer_text)，所以end_position这个标签已经是答案的后一个单词了
        #所以我们预测出来的end_pred_ids不应该加1
        #根据真实答案文本和预测答案文本就可以计算em和f1了
        em_value=get_EM(predict_answer_text,answer_text)
        f1_score=get_F1(predict_answer_text,answer_text)
        em_values.append(em_value)
        f1_scores.append(f1_score)
    return em_values,f1_scores#em_values和f1_scores分别记录着这batch_size样本的每一个的em值和f1分数

def get_F1(predict_answer_text,answer_text):
    predict_tokens=list(predict_answer_text)
    golden_tokens=list(answer_text)

    common=Counter(predict_tokens)&Counter(golden_tokens)
    num_same=sum(common.values())

    if num_same==0:
        return 0
    precision=1.0*num_same/len(predict_tokens)
    recall=1.0*num_same/len(golden_tokens)

    return (2*precision*recall)/(precision+recall)

def get_EM(predict_answer_text,answer_text):
    return list(predict_answer_text)==list(answer_text)

