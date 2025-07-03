import pickle 
import argparse
import os
import numpy as np 
from itertools import repeat 
import tensorflow as tf 
import torch 
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import sys 
from collections import OrderedDict
import nltk 
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
import utils 
import logging 
logging.disable(logging.WARNING)

# 1. load questionPair (use key as user-side) 
# 2. load stackoverflow post of questionPair (user-side)
# 3. load stackoverflow post of post-side (tagged with language (e.g. python))
# 4. make input of model (user-side nl + post-side code) 

def return_posValue(prob):
    return prob[1]

# description = list 
# code: string 
def compute_relevance(args,forumsideDataType, usersideDataType, device, model, tokenizer, qID, userside, forumside):
    
    userList, forumList, keyList = [], [], [] 

    for sidx, userData in enumerate(userside):
        userside_index = f"{qID}_{sidx}"
        for q_number, (cidx, forumData) in enumerate(forumside.items()):   
            # cidx = (question index of forum-side) + (order of code, 1st, 2nd,...)
            dictKey = f"{userside_index}_{cidx}"
            userList.append(userData) 
            forumList.append(forumData)
            keyList.append(dictKey)

    print(f"Total number of data: {len(userList)}, {len(forumList)}, {len(keyList)}")
    result = [] 
    batch_size = args.batch_size 
    batch_start, batch_end = 0, 0
    batch_round = 0 
    while True:
        
        if batch_start >= len(keyList):
            break    
        batch_end = batch_start + batch_size
        if batch_end > len(keyList):
            batch_end = len(keyList)
        
        batch_round += 1
        if batch_round % 100 == 0:
            print(f"Now running ({qID}): {batch_start} -> {batch_end}")
        with torch.no_grad():
            softmax = tf.keras.layers.Softmax()
            batch_key = keyList[batch_start:batch_end]
            batch_user = userList[batch_start:batch_end]
            batch_forum = forumList[batch_start:batch_end]
            # depending on forumside and userside datatype 
            if forumsideDataType == "code":
                tokenizer_output = tokenizer(batch_user, batch_forum, truncation = True, padding = 'max_length', max_length = args.length_limit_code, return_tensors = 'pt')
            if usersideDataType == "code":
                tokenizer_output = tokenizer(batch_forum, batch_user, truncation = True, padding = 'max_length', max_length = args.length_limit_code, return_tensors = 'pt')

            tokenizer_output.to(device)
            logits = model(**tokenizer_output).logits 
            logits = logits.detach().cpu().numpy()
            prob = softmax(logits)
            prob = np.array(prob)
            prob = list(map(return_posValue, prob))
            result.extend(prob)
            batch_start = batch_end 
    print(f"Number of Key: {len(keyList)}, Number of result: {len(result)}")
    return keyList, result                     

def strip(w):
    return w.strip()

def get_gtData(forumside, gt_list):
    
    gt_data = [] 
    for gt in gt_list:
        gt_data.append(forumside[gt])
    return gt_data

# compare user-side description & post-side code 
def main(args, qID, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_type_index, model_name_or_path, device_id, datasetPath):
    
    print("=" * 50)
    print(f"{modelType}({model_name_or_path}), {usersideDataType}({userside_preprocessing_type}), {forumsideDataType} ({forumside_preprocessing_type})")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"0,1,2,3,4"
    device = f"cuda:{args.device_id}"
    args.device = device 
    
    tokenizer_name = "roberta-base"
    model_name_or_path_tmp = "yykimyykim/forumdr-CodeBERT-code-description"
    # load tokenizer and model 
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name_or_path_tmp)
    model.to(args.device)
    model.eval()

    # loading forumside data 
    stackoverflowDataPath = args.stackoverflowDataPath
    stackoverflowDataPath_tmp = stackoverflowDataPath.split("/")[-1].split(".")[0]
    if forumsideDataType == "code":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_code}_codebert.pickle"
    if forumsideDataType == "log" or forumsideDataType == "console":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_output}_codebert.pickle"
    if forumsideDataType == "description_sep":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_des}_codebert.pickle"
    if not os.path.exists(args.datasetPath):
        os.makedirs(args.datasetPath)

    forumside, forumside_gt, saveFlag = utils.load_data_for_inference_codebert(args, forumsideDataPath, forumsideDataType)
    userside = utils.load_data_for_userside(args, usersideDataType, qID)
    if len(userside) == 0:
        return 
    
    print(f"Number of {forumsideDataType} in questions: {len(forumside)}")
    print(f"Userside {(usersideDataType)}: {userside}")
    
     
    keyList, relevance_value = compute_relevance(args, forumsideDataType, usersideDataType, device, model, tokenizer, qID, userside, forumside)
    resultDict = utils.make_result_dict_crossencoder(keyList, relevance_value)
    utils.saveResult(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, resultDict)
    utils.saveData(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, userside, forumside)