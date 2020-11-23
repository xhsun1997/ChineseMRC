import utils
from output import write_predictions
from evaluate import get_eval
from transformers.tokenization_bert import BertTokenizer
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from preprocess import json2features
from transformers.modeling_bert import BertModel,PreTrainedModel,BertPreTrainedModel

import os,copy,json,math,logging,tqdm,argparse,collections
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import transformers
import random
import numpy as np
from optimization import BertAdam, warmup_linear
from tensorboardX import SummaryWriter

class BertForQA(BertPreTrainedModel):
    def __init__(self,config):
        super(BertForQA,self).__init__(config)
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs=nn.Linear(in_features=config.hidden_size,2)
        self.config=config
        self.init_weights()#init weights inherited from BertPreTrainedModel
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,start_positions=None,end_positions=None):
        (sequence_output,pooler_output)=self.bert(input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)
        sequence_output=self.dropout(sequence_output)
        logits=self.qa_outputs(sequence_output)
        start_logits,end_logits=logits.split(1,dim=2)
        start_logits=start_logits.squeeze(2)
        end_logits=end_logits.squeeze(2)
        if start_positions is not None:
            if len(start_positions.size())>1:
                start_positions=start_positions.squeeze(1)
                assert end_positions is not None and len(end_positions.size())>1
                end_positions=end_positions.squeeze(1)
            ignore_index=start_logits.size(1)
            start_positions.clamp_(0,ignore_index)
            end_positions.clamp_(0,ignore_index)
            loss_function=nn.CrossEntropyLoss(ignore_index=ignore_index)
            start_loss=loss_function(input=start_logits,target=start_positions)
            end_loss=loss_function(input=end_logits,target=end_positions)
            loss=(start_loss+end_loss)/2
            return loss
        else:
            return start_logits,end_logits


def predict(model, args, eval_examples, eval_features, device, global_steps, best_f1, best_em, best_f1_em):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.checkpoint_dir,
                                          "predictions_steps" + str(global_steps) + ".json")
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids=input_ids, attention_mask=input_mask,token_type_ids=segment_ids)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(args.checkpoint_dir,eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(args.dev_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    with open(args.log_file, 'a') as aw:
        aw.write(json.dumps(tmp_result) + '\n')
    print(tmp_result)

    if float(tmp_result['F1']) > best_f1:
        best_f1 = float(tmp_result['F1'])

    if float(tmp_result['EM']) > best_em:
        best_em = float(tmp_result['EM'])

    if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
        best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
        utils.torch_save_model(model, args.checkpoint_dir,
                               {'f1': float(tmp_result['F1']), 'em': float(tmp_result['EM'])}, max_save_num=1)

    model.train()

    return best_f1, best_em, best_f1_em


def train_model(args,model,optimizer,train_features,dev_examples,dev_features,device,t_total):
    writer=SummaryWriter(log_dir=args.log_dir)

    all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)
    seq_len = all_input_ids.shape[1]
    # 样本长度不能超过bert的长度限制
    assert seq_len <= args.max_seq_len
    all_start_positions = torch.tensor([f['start_position'] for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f['end_position'] for f in train_features], dtype=torch.long)

    train_data = torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions)
    #train_dataloader = DataLoader(train_data, batch_size=args.n_batch, shuffle=True)
    train_sampler=torch.utils.data.RandomSampler(train_data)
    train_dataloader=torch.utils.data.DataLoader(train_data,sampler=train_sampler,batch_size=args.train_batch_size)
    
    epoch=0
    global_step=0
    current_lr=0.0
    model.train()
    print('***** Training *****')
    model.train()
    global_steps = 1
    best_em = 0
    best_f1 = 0
    F1s = []
    EMs = []
    # 存一个全局最优的模型
    best_f1_em = 0

    for epoch in range(int(args.num_epochs)):
        print('Starting epoch %d' % (epoch+1))
        total_loss = 0
        iteration = 1
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, 
            				start_positions=start_positions, end_positions=end_positions)
            # if n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item()
            if (step+1)%args.print_loss_step==0:
                print("step: {} , average loss: {}, current learning rate : {}".format(step,total_loss/args.print_loss_step,current_lr))
                writer.add_scalar("loss/train",total_loss/args.print_loss_step,(step+1)//args.print_loss_step)
                total_loss=0.0


            if (step+1)%args.gradient_accumulation_steps==0:
                lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group["lr"]=lr_this_step
                current_lr=lr_this_step
                #print("learning rate this step : ",lr_this_step)
                optimizer.step()
                optimizer.zero_grad()
                global_step+=1

            if (step+1)%args.predict_step== 0:
                best_f1, best_em, best_f1_em = predict(model, args, dev_examples, dev_features, device,
                                                        global_steps, best_f1, best_em, best_f1_em)
                writer.add_scalar("f1/test",best_f1,(step+1)//args.predict_step)
                writer.add_scalar("em/test",best_em,(step+1)//args.predict_step)
                F1s.append(best_f1)
                EMs.append(best_em)

    # release the memory
    del model
    del optimizer
    torch.cuda.empty_cache()

    print('Mean F1:', np.mean(F1s), 'Mean EM:', np.mean(EMs))
    print('Best F1:', np.max(F1s), 'Best EM:', np.max(EMs))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--seed', type=list, default=[123, 456, 789, 556, 977])
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=21128)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=2, type=int, help="Total batch size for predictions.")
    parser.add_argument("--print_loss_step", default=300, type=int, help="Total batch size for predictions.")
    parser.add_argument("--predict_step", default=1000, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")


    # data dir
    parser.add_argument('--train_dir', type=str,
                        default='/home/xhsun/Desktop/ChineseMRC/cmrc/data/train_features_bert512.json')
    parser.add_argument('--dev_dir1', type=str,
                        default='/home/xhsun/Desktop/ChineseMRC/cmrc/data/dev_examples_bert512.json')
    parser.add_argument('--dev_dir2', type=str,
                        default='/home/xhsun/Desktop/ChineseMRC/cmrc/data/dev_features_bert512.json')
    parser.add_argument('--train_file', type=str,
                        default='/home/xhsun/Desktop/ChineseMRC/cmrc/data/cmrc2018_train.json')
    parser.add_argument('--dev_file', type=str,
                        default='/home/xhsun/Desktop/ChineseMRC/cmrc/data/cmrc2018_dev.json')

    parser.add_argument('--bert_config_file', type=str,
                        default='/home/xhsun/Desktop/chinese_pytorch/bert_config.json')
    parser.add_argument('--bert_vocab_file', type=str,
                        default='/home/xhsun/Desktop/chinese_pytorch')
    parser.add_argument('--init_restore_dir', type=str,
                        default='check_points/pretrain_models/roberta_wwm_ext_base/pytorch_model.pth')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/xhsun/Desktop/chinese_pytorch')
    parser.add_argument('--log_file', type=str, default='log.txt')

    # use some global vars for convenience
    args = parser.parse_args()
    args.checkpoint_dir += ('/epoch{}_batch{}_lr{}_warmup{}_anslen{}/'
                            .format(args.train_epochs, args.n_batch, args.lr, args.warmup_rate, args.max_ans_length))
    args = utils.check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = 0
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))


    # load data
    print('loading data...')
    tokenizer=BertTokenizer.from_pretrained(args.checkpoint_dir)
    assert args.vocab_size == len(tokenizer.vocab)
    if not os.path.exists(args.train_dir):
        json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'), args.train_dir],
                      tokenizer, is_training=True,
                      max_seq_length=bert_config.max_position_embeddings)

    if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
        json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False,
                      max_seq_length=bert_config.max_position_embeddings)

    train_features = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_features = json.load(open(args.dev_dir2, 'r'))
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    num_train_steps=int(len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    t_total=num_train_steps
    print("t_total : ",t_total)
    print("num train features : ",len(train_features))
    print("num test features : ",len(dev_features))

    for seed_ in args.seed:
        with open(args.log_file, 'a') as aw:
            aw.write('===================================' +
                     'SEED:' + str(seed_)
                     + '===================================' + '\n')
        print('SEED:', seed_)

        random.seed(seed_)
        np.random.seed(seed_)
        torch.manual_seed(seed_)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed_)

        # init model
        print('init model...')
        model=BertForQA.from_pretrained(args.checkpoint_dir)
        #utils.torch_show_all_params(model)
        #utils.torch_init_model(model, args.init_restore_dir)
        model.to(device)
        #if n_gpu > 1:
        #    model = torch.nn.DataParallel(model)
        #optimizer=torch.optim.Adam(params=model.parameters(),lr=args.learning_rate)
        ########################################################################################
        param_optimizer=list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters=[
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        optimizer=BertAdam(optimizer_grouped_parameters,lr=args.learning_rate,warmup=args.warmup_proportion,t_total=t_total)
        #######################################################################################

        if args.do_train:
            train_model(args,model,optimizer,train_features,dev_examples,dev_features,device,t_total)
        # release the memory
            del model
            del optimizer
            torch.cuda.empty_cache()

