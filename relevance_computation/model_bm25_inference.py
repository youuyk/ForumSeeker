import pickle, sys, os  
import argparse 
import utils 
from itertools import repeat 
from rank_bm25 import BM25Okapi
import nltk 
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 

def compute_relevance(userside, bm25, forumsideKey, forumsideData):
    
    userside_tokenized = userside.split(" ")
    scoring = bm25.get_scores(userside_tokenized)
    #print(f"Shape of scoring , example: {scoring.shape}, {scoring[0]}")
    #print(f"scoring key, data: {forumsideKey[0]}, {forumsideData[0]}")
    return scoring 

def get_resultDict(qID, relevance_value, userside, forumside_key):
    
    resultDict = {}
    for idx, d in enumerate(userside):
        k = f"{qID}_{idx}"
        rel_val = relevance_value[idx]
        for f_idx, f_key in enumerate(forumside_key):
            tot_key = f"{k}_{f_key}"
            resultDict[tot_key] = rel_val[f_idx]
    
    return resultDict 

def strip(w):
    return w.strip()

def main(args, qID, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, datasetPath):
    
    stopword_list = stopwords.words('english') 
    man_stopPath = "stopword.txt"
    with open(man_stopPath, "r") as fr:
        man_stop = fr.readlines()
    man_stop = list(map(strip, man_stop))
    stopword_list.extend(man_stop)
    
    stackoverflowDataPath = args.stackoverflowDataPath 
    stackoverflowDataPath_tmp = stackoverflowDataPath.split("/")[-1].split(".")[0] 
    forumsideDataPath = f"{datasetPath}{stackoverflowDataPath_tmp}_{forumsideDataType}_{forumside_preprocessing_type}.pickle"
        
    forumside, _, _ = utils.load_data_for_inference(args, forumsideDataPath, forumsideDataType)
    
    userside = utils.load_data_for_userside(args, usersideDataType, qID)
    
    if len(forumside) == 0:
        return 
        
    if len(userside) == 0:
        print(f"This question has no {usersideDataType}!")
        return False 
  
    forumsideKey = list(forumside.keys())
    forumsideData= list(forumside.values())
    
    print(f"Number of forumside data : {len(forumside)}")
    print(f"Example of forumside {forumsideDataType}: {forumsideData[0]}")
    forumside_data_tokenized = [d.split(" ") for d in forumsideData]
    bm25 = BM25Okapi(forumside_data_tokenized)
       
    relevance_value = list(map(compute_relevance, userside, repeat(bm25), repeat(forumsideKey), repeat(forumsideData)))
    print(f"Number of relevance_value: {len(relevance_value)}, {len(relevance_value[0])}")
    
    resultDict = utils.make_result_dict_biencoder(qID, userside, forumsideKey, relevance_value)
    utils.saveResult(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, resultDict)
    utils.saveData(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, userside, forumside)
        
