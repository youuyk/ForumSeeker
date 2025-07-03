import argparse
import pickle 
import os
import numpy as np 
from itertools import repeat 
import tensorflow as tf 
import torch 
from transformers import AutoTokenizer, AutoModel
import sys 
from collections import OrderedDict
import nltk 
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
from torch.utils.data import Dataset, DataLoader
import utils 

def remove_punc(w):
    punc = [",", ":", ";", "'", "\"", "<", ">", "{", "}", "(", ")", "[", "]"]
    for p in punc:
        w = w.replace(p, "")
    return w

def strip(w):
    return w.strip()

def get_feats(model, tokenizer, dataloader, max_length, device):
    
    embeds = []     
    for data_idx, data in enumerate(dataloader):
        print(f"Loading data: {data_idx}, max_length: {max_length}")
        with torch.no_grad():
            input = tokenizer(data, padding= 'max_length', truncation= True, max_length= max_length, return_tensors = 'pt').to(device)
            embed = model(input.input_ids, attention_mask = input.attention_mask)
            print(f"Shape of embed: {embed.shape}")
        embeds.append(embed)
    embeds = torch.cat(embeds, dim= 0)
    return embeds

def contrast_evaluation(text_embed, code_embed):
    
    #print(text_embed.shape, code_embed.shape)    
    score = text_embed @ code_embed.t()
    #print(score.shape)
    return score 

def create_loader(args, data):
    
    loader = DataLoader(
        data, 
        batch_size = args.batch_size, 
        shuffle = False, 
    )
    return loader

def embed(args, model, tokenizer, device, data, max_length, dataType):

    loader = create_loader(args, data)
    print(f"Data to Loader({dataType}): {len(data)}, {len(loader)}")
    embeds = get_feats(model, tokenizer, loader, max_length, device)
    return embeds 

def compute_relevance(userside_embed, forumside_embed):
        
    scoring = contrast_evaluation(userside_embed, forumside_embed)
    print(f"Shape of scoring: {scoring.shape}")
    return scoring     

def get_gtData(forumside, gt_list):
    
    gt_data = [] 
    for gt in gt_list:
        gt_data.append(forumside[gt])
    return gt_data

def main(args, qID, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, device_id, datasetPath):
    
    # make list of stopwords 
    stopword_list = stopwords.words('english') 
    man_stopPath = "stopword.txt"
    with open(man_stopPath, "r") as fr:
        man_stop = fr.readlines()
    man_stop = list(map(strip, man_stop))
    stopword_list.extend(man_stop)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    device = f"cuda:{args.device_id}"
    args.device = device 
    torch.cuda.empty_cache()
    
    model_name_or_path = "Salesforce/codet5p-110m-embedding"
    tokenizer= AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code = True)
        
    model= model.to(device)
    model.eval()
    print(f"Loaded {model_name_or_path}, model (#para={model.num_parameters()})")

    if "/" in model_name_or_path:
        model_name_or_path_tmp = "_".join(model_name_or_path.split("/"))
    else:
        model_name_or_path_tmp = model_name_or_path
        
    # load post-side dataset (stackoverflow dataset)
    stackoverflowDataPath = args.stackoverflowDataPath  
    stackoverflowDataPath_tmp = stackoverflowDataPath.split("/")[-1].split(".")[0]
    if forumsideDataType == "code":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_code}.pickle"
    if forumsideDataType == "log" or forumsideDataType == "console":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_output}.pickle"
    if forumsideDataType == "description_sep":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_des}.pickle"
    print(f"forumside data: {forumsideDataPath}")
    forumside, _, _ = utils.load_data_for_inference(args, forumsideDataPath, forumsideDataType)
    if len(forumside) == 0:
        return 
    forumsideData = list(forumside.values())
    forumsideKey = list(forumside.keys())
    
    if forumsideDataType == "code":
        forumsideDataEmbedPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_{modelType}_{model_name_or_path_tmp}_embed_limit_{args.length_limit_code}.pickle"
    if forumsideDataType == "log" or forumsideDataType == "console":
        forumsideDataEmbedPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_{modelType}_{model_name_or_path_tmp}_embed_limit_{args.length_limit_output}.pickle"   
    if forumsideDataType == "description_sep":
        forumsideDataEmbedPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_{modelType}_{model_name_or_path_tmp}_embed_limit_{args.length_limit_des}.pickle"

    
    if os.path.exists(forumsideDataEmbedPath):
        print("Loading pre-saved embedding")
        forumside_embed = utils.open_pickle(forumsideDataEmbedPath)
    else:
        print("Creating new embed")
        if forumsideDataType == "code":
            forumside_embed = embed(args, model, tokenizer, device, forumsideData, args.length_limit_code, "code")
        else:
            forumside_embed = embed(args, model, tokenizer, device, forumsideData, args.length_limit_des, forumsideDataType)
        utils.save_pickle(forumsideDataEmbedPath, forumside_embed)
        
    forumside_device = forumside_embed.get_device()
    if str(forumside_device) != args.device_id:
        forumside_embed.detach().cpu()
        forumside_embed.to(device)
    
    print(f"Number of forumside Data({forumsideDataType}): {len(forumsideData)}")
    
    userside = utils.load_data_for_userside(args, usersideDataType, qID)
    if usersideDataType == "code":
        userside_embed = embed(args, model, tokenizer, device, userside, args.length_limit_code, "code")
    else:
        userside_embed = embed(args, model, tokenizer, device, userside, args.length_limit_des, usersideDataType)
    print(f"Data in userside: {userside}")
    print(f"Data in forumside ({forumsideKey[0]}): {forumsideData[0]}")
    # highest related question in stackoverflow 
    if "/" in model_name_or_path:
        model_name_or_path = "_".join(model_name_or_path.split("/"))
    
    userside_embed.to(device)
    forumside_embed.to(device)
    relevance_value = compute_relevance(userside_embed, forumside_embed)
    resultDict = utils.make_result_dict_biencoder(qID, userside, forumsideKey, relevance_value)
        
    utils.saveResult(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, resultDict)
    utils.saveData(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, userside, forumside)