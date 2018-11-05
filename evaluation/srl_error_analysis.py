
# coding: utf-8

# In[73]:


import pandas as pd
import re
import json
import ast


# In[74]:


EXP1_VAL_PREDS_FILE = 'evaluation/101_val_preds.json'


# In[75]:


all_preds = []


# In[76]:


with open(EXP1_VAL_PREDS_FILE) as fp:
    labels_flag = 0
    ex_map = {}
    ctr = 0
    for line in fp:
        if labels_flag == 1:
            clean = re.sub("[\[\],\s]","",line)
            splitted = re.split("[\'\"]",clean)
            result = [s for s in splitted if s != '']
            if 'labels' not in ex_map:
                ex_map['labels'] = [x for x in result if len(x) > 0]
            else:
                ex_map['labels']+= [x for x in result if len(x) > 0]
            if ']' in line:
                labels_flag = 0   
        elif 'tags:' in line:
            labels_flag = 1
        elif 'prediction:' in line:
            print("PREDICTION LINE")
            preds_map = line.split(':',1)
            if len(preds_map) < 2:
                print(line)
            preds = json.loads(preds_map[1])
            ex_map['predictions'] = preds['tags']
            ex_map['verb'] = preds['verb']
            ex_map['words'] = preds['words']
            print(ex_map)
            print(len(ex_map['predictions']))
            print( len(ex_map['labels']))
            if(len(ex_map['predictions']) == len(ex_map['labels'])):
                all_preds.append(ex_map)
                
            ctr += 1
            print("%%%%%%%%%%%%%%%%%%")
            ex_map = {}
            
        
        
        
    


# In[77]:


ctr


# In[78]:


len(all_preds)

