import json
import logging
import os
import random
import pickle
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd


class ERCDataset(Dataset):
    
    def __init__(self, csv_path, 
                model_name,
                num_past_utterances, 
                num_future_utterances,
                max_seq_length=512,
                 batch_size=32,
                ):
        
        self.df = pd.read_csv(csv_path)
        self.df['speaker_utterance'] = self.df.apply(lambda x: f"{x['Speaker']}: {x['Utterance']}", axis=1)
        
        self.emotions = np.unique(self.df['Emotion'])
        self.emotions.sort()
        
        self.label2id = {label:idx for idx, label in enumerate(self.emotions)}
        self.id2label = {idx:label for idx, label in enumerate(self.emotions)}
        
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.get_features()
        
    def get_emotions(self):
        return self.emotions
        
    def __len__(self):
        return len(self.input_ids_arr)
        
    def __getitem__(self, idx):
        
        input_ids, attn_mask = self.input_ids_arr[idx], self.attn_mask_arr[idx]
        label = self.labels_arr[idx]

        
        return (
            torch.Tensor(input_ids).long(),
            torch.Tensor(attn_mask).long(),
            label,
        )
        
    def get_features(self):
        
        input_texts = []
        input_labels = []

        dialogue_ids = np.unique(self.df['Dialogue_ID']).tolist()
        
        for dialogue_id in tqdm(dialogue_ids):

            dial_data = self.df[self.df['Dialogue_ID'] == dialogue_id].reset_index(drop="first")

            for utterance_idx in range(0, len(dial_data)):

                future_idx = min(len(dial_data), utterance_idx + self.num_future_utterances)
                past_idx = max(0,  utterance_idx - self.num_past_utterances)
                emotion_label = dial_data.iloc[utterance_idx]['Emotion']

                past_data = dial_data.loc[past_idx:utterance_idx-1]
                future_data = dial_data.loc[utterance_idx + 1:future_idx]

                past_utterances = ";".join(past_data['speaker_utterance'])
                future_utterances = ";".join(future_data['speaker_utterance'])
                current_utterance = dial_data.iloc[utterance_idx]['speaker_utterance']
                # input_utterances = (
                #     past_utterances + "</s></s>" + current_utterance + "</s></s>" + future_utterances
                # )
                input_utterances = (
                    past_utterances + "</s></s>" + current_utterance 
                )

                input_texts.append(input_utterances)
                input_labels.append(emotion_label)
                
        input_ids_list, attn_mask_list = [], []


        
        for batch_start_idx in tqdm(range(0, len(input_texts), self.batch_size)):

            batch_end_idx = batch_start_idx + self.batch_size

            batch_input_texts = input_texts[batch_start_idx: batch_end_idx]

            batch_encoding = self.tokenizer(
                    batch_input_texts,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors='np'
                )

            batch_input_ids = batch_encoding.input_ids
            batch_attn_mask = batch_encoding.attention_mask

            input_ids_list.append(batch_input_ids)
            attn_mask_list.append(batch_attn_mask)


        self.labels_arr = np.array([self.label2id[label] for label in input_labels])
        self.input_ids_arr = np.concatenate(input_ids_list, axis=0)
        self.attn_mask_arr = np.concatenate(attn_mask_list, axis=0)

class ERCDatasetCS(Dataset):
    
    def __init__(self,
                csv_path,
                feats_path,
                model_name,
                num_past_utterances, 
                num_future_utterances,
                max_seq_length=512,
                 batch_size=32,
                ):
        
        self.df = pd.read_csv(csv_path)
        self.df['speaker_utterance'] = self.df.apply(lambda x: f"{x['Speaker']}: {x['Utterance']}", axis=1)
        
        with open(feats_path,"rb") as f:
            self.feats = pickle.load(f)
#         print(self.feats.head())
#         self.feats = pd.read_csv(feats_path)
#         self.cs_feats = self.feats['all_xw']
        
        self.emotions = np.unique(self.df['Emotion'])
        self.emotions.sort()
        
        self.label2id = {label:idx for idx, label in enumerate(self.emotions)}
        self.id2label = {idx:label for idx, label in enumerate(self.emotions)}
        
        self.num_past_utterances = num_past_utterances
        self.num_future_utterances = num_future_utterances
        
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left', truncation_side = 'left',)
#         self.cs_tokenizer = AutoTokenizer.from_pretrained(cs_model_name)
        
        self.get_features()
        
    def get_emotions(self):
        return self.emotions
        
    def __len__(self):
        return len(self.input_ids_arr)
        
    def __getitem__(self, idx):
        
        input_ids, attn_mask = self.input_ids_arr[idx], self.attn_mask_arr[idx]
        cs_feats = self.input_cs[idx]
#         cs_input_ids, cs_attn_mask = self.cs_input_ids_arr[idx], self.cs_attn_mask_arr[idx]
        label = self.labels_arr[idx]

        
        return (
            torch.Tensor(input_ids).long(),
            torch.Tensor(attn_mask).long(),
            cs_feats,
            label,
        )
        
    def get_features(self):
        input_texts = []
        input_labels = []
        self.input_cs = []

        dialogue_ids = np.unique(self.df['Dialogue_ID']).tolist()
        
        for dialogue_id in tqdm(dialogue_ids):
            dial_data = self.df[self.df['Dialogue_ID'] == dialogue_id].reset_index(drop="first")

            for utterance_idx in range(0, len(dial_data)):

                future_idx = min(len(dial_data), utterance_idx + self.num_future_utterances)
                past_idx = max(0,  utterance_idx - self.num_past_utterances)
                emotion_label = dial_data.iloc[utterance_idx]['Emotion']

                past_data = dial_data.loc[past_idx:utterance_idx-1]
                future_data = dial_data.loc[utterance_idx + 1:future_idx]

                past_utterances = ";".join(past_data['speaker_utterance'])
                future_utterances = ";".join(future_data['speaker_utterance'])
                current_utterance = dial_data.iloc[utterance_idx]['speaker_utterance']
                # input_utterances = (
                #     past_utterances + "</s></s>" + current_utterance + "</s></s>" + future_utterances
                # )
                input_utterances = (
                    past_utterances + "</s></s>" + current_utterance 
                )
                
#                 cs_data = dial_data.iloc[utterance_idx]['oWant'] + dial_data.iloc[utterance_idx]['xAttr']
                cs_data = self.feats.iloc[utterance_idx]['xAttr'] + self.feats.iloc[utterance_idx]['xWant']

                input_texts.append(input_utterances)
                input_labels.append(emotion_label)
                self.input_cs.append(cs_data)
                
        input_ids_list, attn_mask_list = [], []
#         print(self.input_cs)
#         self.input_cs = np.array(self.input_cs)
#         print(self.input_cs)
#         cs_input_ids_list, cs_attn_mask_list =[], []
        
        for batch_start_idx in tqdm(range(0, len(input_texts), self.batch_size)):
            batch_end_idx = batch_start_idx + self.batch_size
            batch_input_texts = input_texts[batch_start_idx: batch_end_idx]

            batch_encoding = self.tokenizer(
                    batch_input_texts,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors='np'
                )

            batch_input_ids = batch_encoding.input_ids
            batch_attn_mask = batch_encoding.attention_mask

            input_ids_list.append(batch_input_ids)
            attn_mask_list.append(batch_attn_mask)
        
#         for cs in input_cs:
#             cs_batch_encoding = self.cs_tokenizer(
#                 cs,
#                 max_length=self.max_seq_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_attention_mask=True,
#                 add_special_tokens=True,
#                 return_tensors='np'
#             )
#             cs_input_ids_list.append(cs_batch_encoding.input_ids)
#             cs_attn_mask_list.append(cs_batch_encoding.attention_mask)
                
        self.labels_arr = np.array([self.label2id[label] for label in input_labels])
        
#         print("input_ids_list -> ", input_ids_list.shape)
        self.input_ids_arr = np.concatenate(input_ids_list, axis=0)
        self.attn_mask_arr = np.concatenate(attn_mask_list, axis=0)
        
#         print("cs_input_ids_list -> ", cs_input_ids_list)
#         self.cs_input_ids_arr = np.squeeze(np.array(cs_input_ids_list), axis=1)
#         self.cs_attn_mask_arr = np.squeeze(np.array(cs_attn_mask_list), axis=1)
        
#         print(self.labels_arr.shape, self.input_ids_arr.shape, self.attn_mask_arr.shape, self.cs_input_ids_arr.shape, self.cs_attn_mask_arr.shape)
        print(self.labels_arr.shape, self.input_ids_arr.shape, self.attn_mask_arr.shape)
        
    
    
if __name__ == "__main__":
    
    input_file = '/workspace/projects/nlp/efr/baselines/erc_baselines/bert_erc/custom_dataset/dev_sent_emo.csv'
    erc_dataset = ERCDataset(
        csv_path=input_file,
        model_name='bert-base-uncased',
        num_past_utterances=5,
        num_future_utterances=0
    )
    
    batch_size = 32
    dataloader = DataLoader(erc_dataset, batch_size=batch_size)
    
    for batch in tqdm(iter(dataloader)):
        pass