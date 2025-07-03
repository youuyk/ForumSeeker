import pickle, sys, os  
import argparse
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from model import Model, TextDataset_code, TextDataset_nl
import utils 
import torch 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np 

def compute_relevance(userside, forumside):
    
    userside = np.concatenate(userside, 0)
    forumside = np.concatenate(forumside, 0)
    
    scores = np.matmul(userside, forumside.T)
    print(f"Shape of score: {scores.shape}")
    return scores 

def encode(args, model, tokenizer, datatype, dataset):
    
    if datatype == "code":
        rawDataset = TextDataset_code(args, tokenizer, datatype, dataset)
    elif datatype == "alldata" or datatype == "title" or datatype == "description_sep" or datatype == "log" or datatype == "console":
        rawDataset = TextDataset_nl(args, tokenizer, datatype, dataset)
    rawDataset_sampler = SequentialSampler(rawDataset)
    rawDataset_dataloader = DataLoader(rawDataset, sampler = rawDataset_sampler, batch_size = args.batch_size, num_workers = 4)
    
    model.eval()
    
    vecs = [] 
    if datatype == "code":
        print(f"Length of Dataloader: {len(rawDataset_dataloader)}")
        for idx, batch in enumerate(rawDataset_dataloader):
            code_inputs = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            with torch.no_grad():
                code_vec = model(code_inputs= code_inputs, attn_mask = attn_mask, position_idx = position_idx)
                vecs.append(code_vec.cpu().numpy())
        
    elif datatype == "alldata" or datatype == "description_sep" or datatype == "title" or datatype == "log" or datatype == "console":
        
        for idx, batch in enumerate(rawDataset_dataloader):
            nl_ids = batch.to(args.device)
            with torch.no_grad():
                nl_vec = model(nl_inputs = nl_ids)
                vecs.append(nl_vec.cpu().numpy())
                
    return vecs 

def strip(w):
    return w.strip()

def main(args, qID, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, device_id, datasetPath):
    
    stopword_list = stopwords.words('english')
    man_stopPath = "stopword.txt"
    with open(man_stopPath, "r") as fr:
        man_stop = fr.readlines()
    man_stop = list(map(strip, man_stop))
    stopword_list.extend(man_stop)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = f"cuda:{args.device_id}"
    args.device = device
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
    model = Model(model)

    # load fine-tuned model weight 
    graphcodebert_name = "../forumdr-GraphCodeBERT-code-output/model.bin"
    model.load_state_dict(torch.load(graphcodebert_name, weights_only = False, map_location='cuda:0')) 
    model.to(device)    
    model.eval()

    stackoverflowDataPath = args.stackoverflowDataPath
    stackoverflowDataPath_tmp = stackoverflowDataPath.split("/")[-1].split(".")[0] 
    if forumsideDataType == "code":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_code}.pickle"
    if forumsideDataType == "log" or forumsideDataType == "console":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_output}.pickle"
    if forumsideDataType == "description_sep":
        forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_limit_{args.length_limit_des}.pickle"
    forumside, _, _ = utils.load_data_for_inference(args, forumsideDataPath, forumsideDataType)
    userside = utils.load_data_for_userside(args, usersideDataType, qID)
    
    forumsideData = list(forumside.values())
    forumsideKey = list(forumside.keys())
    print(f"Number of {forumsideDataType} in questions: {len(forumsideKey)}")

    if "/" in model_name_or_path:
        model_name_or_path_tmp = "_".join(model_name_or_path.split("/"))
    else:
        model_name_or_path_tmp = model_name_or_path
        
    if forumsideDataType == "code":
        forumsideDataEmbedPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_{modelType}_{model_name_or_path_tmp}_embed_limit_{args.length_limit_code}.pickle"
    if forumsideDataType == "log" or forumsideDataType == "console":
        forumsideDataEmbedPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_{modelType}_{model_name_or_path_tmp}_embed_limit_{args.length_limit_output}.pickle"   
    if forumsideDataType == "description_sep":
        forumsideDataEmbedPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}_{modelType}_{model_name_or_path_tmp}_embed_limit_{args.length_limit_des}.pickle"

    print("Create new embed")
    forumside_embed = encode(args, model, tokenizer, forumsideDataType, forumsideData)
    utils.save_pickle(forumsideDataEmbedPath, forumside_embed)

    userside_embed = encode(args,model, tokenizer, usersideDataType, userside)
    
    relevance_value = compute_relevance(userside_embed, forumside_embed)
    resultDict = utils.make_result_dict_biencoder(qID, userside, forumsideKey, relevance_value)        
    
    if "/" in model_name_or_path:
        model_name_or_path = "_".join(model_name_or_path.split("/"))
        
    utils.saveResult(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID,resultDict)
    utils.saveData(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, userside, forumside)
         
