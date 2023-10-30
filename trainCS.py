import os
import random
import numpy as np
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader

from model import ERCModelCS
from dataset import ERCDatasetCS
from loss import FocalLoss
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = CONFIG['ptmlr']
    '''
    frozen_params = []
    frozen_layers = [3,4,5,6,7,8]
    for layer_idx in frozen_layers:
        frozen_params.extend(
            list(map(id, model.context_encoder.encoder.layer[layer_idx].parameters()))
        )
    '''
    bert_params = list(map(id, model.context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        # if id(param) in frozen_params:
        #     continue
        lr = CONFIG['lr']
        weight_decay = 0
        if id(param) in bert_params:
            lr = pre_train_lr
        if not any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append(
            {
                'params': param,
                'lr': lr,
                'weight_decay': weight_decay
            }
        )
        # warmup的时候不考虑bert
        warmup_params.append(
            {
                'params': param,
                'lr': 0 if id(param) in bert_params else lr,
                'weight_decay': weight_decay
            }
        )
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'], reverse=True)
    return params


def train_epoch(model, optimizer, dataloader, epoch_num=0, max_step=-1):

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    class_weights = torch.Tensor([1.29822955, 1.96171587, 8.37204724, 2.06857977, 0.66619674,
       0.27200051, 1.90546595, 2.41099773])
    
    # criterion = FocalLoss(gamma=0, alpha=class_weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.cuda())
    tq_train = tqdm(total=len(dataloader), position=1)
    accumulation_steps = CONFIG['accumulation_steps']

    for batch_id, batch_data in enumerate(dataloader):
        batch_data = [x.to(model.device()) for x in batch_data]
        input_ids = batch_data[0]
        attention_mask = batch_data[1]
        cs_feats = batch_data[2]
        emotion_idxs = batch_data[3]
        
     
        _,logits = model(input_ids, attention_mask, cs_feats)
        
        # loss += loss_func(outputs[3], sentiment_idxs)
        loss = criterion(logits, emotion_idxs)
        tq_train.set_description('loss is {:.2f}'.format(loss.item()))
        tq_train.update()
        loss = loss / accumulation_steps
        loss.backward()
        if batch_id % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # torch.cuda.empty_cache()
    tq_train.close()



def test(model, dataloader, emotions, is_val=False):

    y_true_list, y_pred_list = [], []
    model.eval()
    tq_test = tqdm(total=len(dataloader), desc="testing", position=2)
    for batch_id, batch_data in enumerate(dataloader):
        
        batch_data = [x.to(model.device()) for x in batch_data]
        input_ids = batch_data[0]
        attention_mask = batch_data[1]
        cs_feats = batch_data[2]
        emotion_idxs = batch_data[3]
        
        _,logits = model(input_ids, attention_mask, cs_feats)
        preds = torch.argmax(logits, axis=-1)
        
        y_true_list.extend(emotion_idxs.tolist())
        y_pred_list.extend(preds.tolist())
        tq_test.update()
        
    f1 = f1_score(y_true=y_true_list, y_pred=y_pred_list, average='weighted')
    acc = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
    
    print("*"*30)
    print("[INFO] Classification report")
    print(classification_report(y_true_list, y_pred_list, target_names=emotions))
    print("*"*30)
    
    phase = "Validation" if is_val else "Test"
    
    print("*"*30)
    print(f"[INFO] {phase} accuracy: {acc:.4f}")
    print(f"[INFO] {phase} F1: {f1:.4f}")
    print("*"*30)
    model.train()
    return f1


def train(model, train_loader, val_loader, test_loader, emotions, config):

    # warmup
    optimizer = torch.optim.AdamW(get_paramsgroup(model, warmup=True))
    
    for epoch in range(CONFIG['wp']):
        train_epoch(model, optimizer, train_loader, epoch_num=epoch)
        torch.cuda.empty_cache()
        f1 = test(model, val_loader, emotions, is_val=True)
        torch.cuda.empty_cache()
        print('f1 on dev @ warmup epoch {} is {:.4f}'.format(
            epoch, f1), flush=True)
    # train
    optimizer = torch.optim.AdamW(get_paramsgroup(model))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)
    
    best_f1 = -1
    tq_epoch = tqdm(total=CONFIG['epochs'], position=0)
    for epoch in range(CONFIG['epochs']):
        tq_epoch.set_description('training on epoch {}'.format(epoch))
        tq_epoch.update()
        train_epoch(model, optimizer, train_loader, epoch_num=epoch)
        torch.cuda.empty_cache()
        f1 = test(model, val_loader, emotions, is_val=True)
        torch.cuda.empty_cache()
        print('[INFO] F1 on dev @ epoch {} is {:.4f}'.format(epoch, f1), flush=True)
        # '''
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model,
                       os.path.join(config['save_dir'], 'f1_{:.4f}_@epoch{}.pkl')
                       .format(best_f1, epoch))
        if lr_scheduler.get_last_lr()[0] > 1e-5:
            lr_scheduler.step()
        f1 = test(model, test_loader, emotions, is_val=False)
        print('[INFO] F1 on test @ epoch {} is {:.4f}'.format(epoch, f1), flush=True)
        # f1 = test(model, test_on_trainset)
        # print('f1 on train @ epoch {} is {:.4f}'.format(epoch, f1), flush=True)
        # '''
    tq_epoch.close()
    lst = os.listdir(config['save_dir'])
    lst = list(filter(lambda item: item.endswith('.pkl'), lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join(config['save_dir'], x)))
    model = torch.load(os.path.join(config['save_dir'], lst[-1]))
    f1 = test(model, test_loader, emotions, is_val=False)
    print('best f1 on test is {:.4f}'.format(f1), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-te', '--test', action='store_true',
                        help='run test', default=False)
    parser.add_argument('-tr', '--train', action='store_true',
                        help='run train', default=False)

    args = parser.parse_args()
    
    CONFIG = {}
    CONFIG['data_path'] = './custom_dataset/'
    CONFIG['device'] = torch.device('cuda')
    CONFIG['lr'] = 1e-4
    CONFIG['ptmlr'] = 1e-5
    CONFIG['num_classes'] = 8
    CONFIG['epochs'] = 20
    CONFIG['model_name'] = 'roberta-base'
    CONFIG['batch_size'] = 1
    CONFIG['dropout'] = 0.3
    CONFIG['wp'] = 0
    CONFIG['accumulation_steps'] =  8
    CONFIG['save_dir'] =  './robertaCS'
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    seed_everything(13)
    
    train_data_path = os.path.join(CONFIG['data_path'], 'train_sent_emo_cs.csv')
    train_feats = os.path.join(CONFIG['data_path'], 'train_new_df.p')
    
    test_data_path = os.path.join(CONFIG['data_path'], 'test_sent_emo_cs.csv')
    test_feats = os.path.join(CONFIG['data_path'], 'test_new_df.p')
    
    val_data_path = os.path.join(CONFIG['data_path'], 'val_sent_emo_cs.csv')
    val_feats = os.path.join(CONFIG['data_path'], 'dev_new_df.p')

    train_dataset = ERCDatasetCS(
        csv_path=train_data_path,
        feats_path=train_feats,
        model_name=CONFIG['model_name'],
        num_past_utterances=5,
        num_future_utterances=0
    )
    val_dataset = ERCDatasetCS(
        csv_path=val_data_path,
        feats_path=val_feats,
        model_name=CONFIG['model_name'],
        num_past_utterances=5,
        num_future_utterances=0
    )
    test_dataset = ERCDatasetCS(
        csv_path=test_data_path,
        feats_path=test_feats,
        model_name=CONFIG['model_name'],
        num_past_utterances=5,
        num_future_utterances=0
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    
    emotions = test_dataset.get_emotions()
    print(emotions)
    
    device = CONFIG['device']

    if args.train:
        # train(model, train_loader, val_loader, test_loader)
        model = ERCModelCS(CONFIG)
    
        
        model.to(device)

        train(model, train_loader, val_loader, test_loader, emotions, CONFIG)
    if args.test:
        lst = os.listdir(CONFIG['save_dir'])
        lst = list(filter(lambda item: item.endswith('.pkl'), lst))
        lst = sorted(lst, key=lambda x: float(x.split("_")[1]), reverse=True)
        model = torch.load(os.path.join(CONFIG['save_dir'], lst[0]))
        model.to(device)
        best_f1 = test(model, test_loader, emotions, is_val=False)

# python train.py -tr -wp 0 -bsz 1 -acc_step 8 -lr 1e-4 -ptmlr 1e-5 -dpt 0.3 >> output.log 0.6505
