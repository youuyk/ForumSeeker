import utils 
from itertools import repeat 
import model_bm25_inference, model_codebert_inference, model_sentencebert_inference, model_graphcodebert_inference, model_codeT5p_inference

def run_model(args, modelInfo, stackoverflowData):
    
    userside = modelInfo['userside']
    forumside = modelInfo['forumside']
    userside_preprocessing_type = int(modelInfo['userside_preprocessing_type'])
    forumside_preprocessing_type = int(modelInfo['forumside_preprocessing_type'])
    modelType = modelInfo['modelType']
    model_name_or_path = modelInfo['model_name_or_path']
    
    print("-" * 30)        
    print(modelType)
    print(args.qID , modelInfo)
    
    # code 
    if modelType == "CodeBERT":
        print("Running CodeBERT")
        if forumside == "log" or forumside == "console" or userside == "log" or userside == "console":
            model_type_index = 1
        else:
            #if forumside or userside is 'description' or 'description_sep' 
            model_type_index = 0 
        model_codebert_inference.main(args, args.qID, userside, forumside, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_type_index, model_name_or_path, args.device_id, args.datasetPath)
    if modelType == "CodeT5+":
        print("Running CodeT5+")
        model_codeT5p_inference.main(args, args.qID, userside, forumside, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, args.device_id, args.datasetPath)
    if modelType == "GraphCodeBERT":
        print("Running GraphCodeBERT")
        model_graphcodebert_inference.main(args, args.qID, userside, forumside, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, args.device_id, args.datasetPath)
    # description
    if modelType == "SentenceBERT":
        model_sentencebert_inference.main(args, args.qID, userside, forumside, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, args.device_id, args.datasetPath)
    if modelType == "BM25":
        model_bm25_inference.main(args, args.qID, userside, forumside, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, args.datasetPath)
       
def check_type(keyName, userData_qID, modelDict):
    
    if keyName not in userData_qID:
        return None     
    dataList = userData_qID[keyName]
    if len(dataList) == 0:
        return None 
    run_model = [] 
    
    for k, v in modelDict.items():
        userside = v.split("_")[0]
        if keyName in userside:
            run_model.append(k)
    return run_model 

def main(args,qID, keyName):
    
    #qID = args.qID
    modelResultPathType = args.modelResultPathType 
    stackoverflowDataPath = args.stackoverflowDataPath 
    userSideDataPath = args.userSideDataPath
 
    model_list, modelResultPathDict, modelResultPathType = utils.get_model(modelResultPathType)

    stackoverflowData = utils.open_pickle(stackoverflowDataPath)
    stackoverflowKey = list(stackoverflowData.keys())
    
    userSideData_all = utils.open_pickle(userSideDataPath)
    userSideData = userSideData_all[qID] 
    modelType = list(map(check_type, keyName, repeat(userSideData), repeat(modelResultPathDict)))
    modelType = utils.extend_list(modelType)
    
    if len(modelType) == 0:
        return
    
    modelInfo = utils.get_model_info(modelType, modelResultPathDict) 
    
    print(modelType)
     
    list(map(run_model, repeat(args), modelInfo, stackoverflowData))

    
    