## import library
import pandas as pd 
import numpy as np 
import os
import streamlit as st 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import TFBertModel
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel

## define the bert
class BertClassifier(nn.Module):
    
    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                
            labels=None):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1] # batch, hidden
        cls_output = self.classifier(cls_output) # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels)
        return loss, cls_output
## set the GPU
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

## load our model
model_path = 'C:\\Users\\admin\\Desktop\\capstone\\project\\bert_orginal.pkl'
model = torch.load(model_path)

## Title 
st.title("Toxic comment detection") 
    
## here we define some of the front end elements of the web page like the font and background color, the padding and the text to be displayed 
html_temp = """ 
<div style ="background-color:gray;padding:13px"> 
<h4 style ="color:black;text-align:center;">Using the powerful BERT to detect toxic comment </h4> 
<h4 style ="color:black;text-align:center;">Warning: it might be a little slow  </h4> 
</div> 
"""
    
## this line allows us to display the front end aspects we have defined in the above code 
st.markdown(html_temp, unsafe_allow_html = True) 
    
## enter the text for predict
text = st.text_input("Please paste the text you want to detext:", "Type Here") 
result ="" 
    
## when the predict button was click, it will start to predict with BERT and ouput the table of probility in each class
if st.button("Predict"):
    st.write("BERT is working")
    ## predict
    texts = []
    bert_model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    max_seq_len = 128 
    text = tokenizer.encode(text, add_special_tokens=True, max_length = max_seq_len, truncation=True)
    texts.append(torch.LongTensor(text))
    x = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    mask = (x != tokenizer.pad_token_id).float().to(device)
    with torch.no_grad():
        _, outputs = model(x, attention_mask=mask)
    outputs = outputs.cpu().numpy()
    ## set the table
    df = pd.DataFrame({
        "Class": ['Probility'],
        "toxic": [round(outputs[0][0], 2)],
        "severe_toxic": [round(outputs[0][1], 2)],
        "obscene": [round(outputs[0][2], 2)],
        "threat": [round(outputs[0][3], 2)],
        "insult": [round(outputs[0][4], 2)],
        "identity_hate": [round(outputs[0][5], 2)]
    })
    ## show the table
    st.table(df)


