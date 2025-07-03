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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import utils 

# 1. load questionPair (use key as user-side) 
# 2. load stackoverflow post of questionPair (user-side)
# 3. load stackoverflow post of post-side (tagged with language (e.g. python))
# 4. make input of model (user-side nl + post-side code) 

def encode(model_name_or_path, data):
    
    model = SentenceTransformer(model_name_or_path)
    embed = model.encode(data)
    print(f"Shape of embed: {embed.shape}")
    return embed 
    
def compute_relevance(args, userside_embed, forumside_embed):
    
    print(f"Shape of embedding(userside, forumside): {userside_embed.shape}, {forumside_embed.shape}")

    #userside_embed = np.reshape(userside_embed, (1, len(userside_embed))) 
    #forumside_embed = np.reshape(forumside_embed, (1, len(forumside_embed)))
    #print(f"Shape of embed (after reshape): {userside_embed.shape}, {forumside_embed.shape}")
    similarities = cosine_similarity(userside_embed, forumside_embed)   
    print(f"Shape of similarity: {similarities.shape}")
    return similarities
        
def strip(word):
    return word.strip()     


# compare user-side description & post-side code 
def main(args, qID, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, device_id, datasetPath):
    
    stopword_list = stopwords.words('english') 
    man_stopPath = "stopword.txt"
    with open(man_stopPath, "r") as fr:
        man_stop = fr.readlines()
    man_stop = list(map(strip, man_stop))
    stopword_list.extend(man_stop)
        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"0,1,2,3,4"
    device = f"cuda:{args.device_id}"
    args.device = device 
    
    if "/" in model_name_or_path:
        model_name_or_path_tmp = "_".join(model_name_or_path.split("/"))
    else:
        model_name_or_path_tmp = model_name_or_path
        
    stackoverflowDataPath = args.stackoverflowDataPath 
    stackoverflowDataPath_tmp = stackoverflowDataPath.split("/")[-1].split(".")[0] 
    if forumsideDataType == "code":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_code}.pickle"
    if forumsideDataType == "log" or forumsideDataType == "console":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_output}.pickle"
    if forumsideDataType == "description_sep":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_des}.pickle"
    if forumsideDataType == "post":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}.pickle"
    forumside, _, _ = utils.load_data_for_inference(args, forumsideDataPath, forumsideDataType)
    if len(forumside) == 0:
        return 
    forumsideKey = list(forumside.keys())
    forumsideData = list(forumside.values())

    forumsideDataEmbedPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_{modelType}_{model_name_or_path_tmp}_embed.pickle"
    if os.path.exists(forumsideDataEmbedPath):
        print("Loading pre-saved embedding")
        forumside_embed = utils.open_pickle(forumsideDataEmbedPath)
    else:
        print(f"Number of forumside Data: {len(forumsideData)}")
        print(f"Example of forumside ({forumsideDataType}): {forumsideKey[0]}, {forumsideData[0]}")
        forumside_embed = encode(model_name_or_path, forumsideData)
        utils.save_pickle(forumsideDataEmbedPath, forumside_embed)      
        
    userside = utils.load_data_for_userside(args, usersideDataType, qID)    
    userside_embed = encode(model_name_or_path, userside)
    relevance_value = compute_relevance(args, userside_embed, forumside_embed)
    
    print(f"Userside {qID} {usersideDataType}: {userside}") 
    print(f"Number of userside Data: {len(userside_embed)}")
    relevance_value = compute_relevance(args, userside_embed, forumside_embed) 
    
    if "/" in model_name_or_path:
        model_name_or_path = "_".join(model_name_or_path.split("/")) 
        
    resultDict = utils.make_result_dict_biencoder(qID, userside, forumsideKey, relevance_value)
    
    utils.saveResult(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType,model_name_or_path,  qID, resultDict)
    utils.saveData(args,usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, userside, forumside)
