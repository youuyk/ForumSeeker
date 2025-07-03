# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import numpy as np 
import torch    
import multiprocessing
from tqdm import tqdm, trange
from tree_sitter import Language, Parser
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from torch.utils.data import Dataset 

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser 

class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

class TextDataset_code(Dataset):
    
    def __init__(self, args, tokenizer, datatype, data):
        
        self.args = args
        self.examples = []
        
        dataList = [] 
        for d_idx, d in enumerate(data):
            # d: data, datatype: code or description 
            dataList.append((d, tokenizer, args, datatype))
        
        cpu_count =10
        pool = multiprocessing.Pool(processes= cpu_count) 
        # list of codes (code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg)        
        self.examples = pool.map(convert_examples_to_features, tqdm(dataList, total = len(dataList)))
        pool.close()
        pool.join() 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        self.args.code_length = 256 
        self.args.data_flow_length = 64
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                            self.args.code_length + self.args.data_flow_length), dtype = bool)

        node_index = sum([i>1 for i in self.examples[item].position_idx])
        max_length = sum([i!=1 for i in self.examples[item].position_idx])
        attn_mask[:node_index,:node_index] = True 
        
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx:max_length] = True 
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx + node_index, a:b] = True 
                attn_mask[a:b, idx+node_index] = True   
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a +node_index <len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a +node_index] = True  
        
        return (torch.tensor(self.examples[item].code_ids), 
                torch.tensor(attn_mask), 
                torch.tensor(self.examples[item].position_idx)) 
    
    def __part__(self):
        part_example = self.examples[:100]
        return part_example
    
class TextDataset_nl(Dataset):
    
    def __init__(self, args, tokenizer, datatype, data):
        
        self.args = args
        self.examples = []
        
        dataList = [] 
        for d_idx, d in enumerate(data):
            # d: data, datatype: code or description 
            dataList.append((d, tokenizer, args, datatype))
        
        cpu_count =10
        pool = multiprocessing.Pool(processes= cpu_count) 
        # list of codes (code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg)        
        self.examples = pool.map(convert_examples_to_features, tqdm(dataList, total = len(dataList)))
        pool.close()
        pool.join() 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].nl_ids))  
    
    def __part__(self):
        part_example = self.examples[:100]
        return part_example

                
class InputFeatures_code(object):
    
    def __init__(self, code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg):
        
        self.code_tokens = code_tokens
        self.code_ids = code_ids 
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code 
        self.dfg_to_dfg = dfg_to_dfg 
        
class InputFeature_nl(object):
    
    def __init__(self, nl_tokens, nl_ids):
        self.nl_tokens = nl_tokens 
        self.nl_ids = nl_ids 

def extract_dataflow(code, parser, lang):
    
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass 
    
    try:
        tree = parser[0].parse(bytes(code, 'utf-8'))
        root_node = tree.root_node 
        tokens_index= tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[] 
        code_tokens=[]
        
    return code_tokens, dfg       
            
# item: data from stackoverflow    
def convert_examples_to_features(item):
    
    d, tokenizer, args, datatype = item
    if datatype == "code":
        lang= "python"
        code_length = 256
        data_flow_length = 64
        
        parser = parsers[lang]
        code_tokens, dfg = extract_dataflow(d, parser, lang)
        code_tokens = [tokenizer.tokenize('@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
        ori2cur_pos= {}
        ori2cur_pos[-1] = (0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        #truncating
        code_tokens=code_tokens[:code_length+data_flow_length-2-min(len(dfg),data_flow_length)]
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:code_length+data_flow_length-len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        code_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=code_length+data_flow_length-len(code_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        code_ids+=[tokenizer.pad_token_id]*padding_length    
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 
        inputFeatures = InputFeatures_code(code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg)
    
    elif datatype == "alldata" or datatype == "description" or datatype == "title" or datatype== "description_sep" or datatype == "log" or datatype == "console":
        
        nl_length = 128 
        nl = d 
        nl_tokens = tokenizer.tokenize(nl)[:nl_length - 2]
        nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] 
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length 

        inputFeatures = InputFeature_nl(nl_tokens, nl_ids)
    
    return inputFeatures
