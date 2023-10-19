import torch
import torch.nn as nn
from transformers import AutoModel

from context_aware_attention import ContextAwareAttention

class ERCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']

        self.context_encoder = AutoModel.from_pretrained(config['model_name'])
        self.dim = 768
        self.fc = nn.Linear(self.dim, self.num_classes)
        
    def device(self):
        return self.context_encoder.device
    
    def forward(self, input_ids, attention_mask):
        '''
        PASS
        '''
    
        ## For Bert, Roberta
#         utterance_encoded = self.context_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )['pooler_output']

        ## For mBERT, muril
        utterance_encoded = self.context_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )['last_hidden_state'].mean(dim=1)#['pooler_output']
#         print("utterance_encoded -> ", utterance_encoded.size())

        logits = self.fc(utterance_encoded)
        
        return logits
    
class ERCModelCS(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']

        self.context_encoder = AutoModel.from_pretrained(config['model_name'])
#         self.cs_encoder = AutoModel.from_pretrained(config['cs_model_name'])
        self.dim = 768
        
        self.caa =  ContextAwareAttention(dim_model = self.dim, dim_context = self.dim)
        
        self.merge_gate = nn.Linear(2*self.dim, self.dim)
        
        self.layer_norm = nn.LayerNorm(self.dim, self.dim)
        
        self.fc = nn.Linear(self.dim, self.num_classes)
        
    def device(self):
        return self.context_encoder.device
    
    def forward(self, input_ids, attention_mask, cs_feats):
        '''
        PASS
        '''
    
        utterance_encoded = self.context_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )['pooler_output'].unsqueeze(1)
                
        cs_encoded = cs_feats
        
        caa_output = self.caa(
            q = utterance_encoded,
            k = utterance_encoded,
            v = utterance_encoded,
            context = cs_encoded
        )
                
        weight_cs = torch.sigmoid(self.merge_gate(torch.cat((utterance_encoded, caa_output), dim=-1)))
        
        output = self.layer_norm(utterance_encoded + weight_cs * caa_output)

        logits = self.fc(output).squeeze(1)
        
        return output, logits