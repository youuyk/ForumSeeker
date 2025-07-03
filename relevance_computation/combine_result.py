import pickle, sys, os  
import modelResultPath, weightResultPath
import argparse
from itertools import repeat 
import utils
from collections import OrderedDict
from scipy.stats import gmean 
from rank_bm25 import BM25Okapi
from itertools import islice
import ranking_metric
import multiprocessing

def make_sorted_result_dict_sum(args, resultDict, qID):
    
    sortedDict = {}       
    stID_set = set()
    qID_dataIndex_set = set()
    for k, v in resultDict.items():
        qID_dataIndex = int(k.split("_")[1])
        qID_dataIndex_set.add(qID_dataIndex)
        stID = int(k.split("_")[2])
        stID_set.add(stID)
        if qID_dataIndex not in sortedDict:
            sortedDict[qID_dataIndex] = {}
        tmp = sortedDict[qID_dataIndex]
        if stID not in tmp:
            tmp[stID] = v 
        else:
            if tmp[stID] < v:
                tmp[stID] = v  
        sortedDict[qID_dataIndex] = tmp    

    # dataIndex: index of data in qID 
    # question_dict: relevance dict between dataIndex and stData (max was chosen)
    
    fDict = {}
    for dataIndex, question_dict in sortedDict.items():
        for stID, v in question_dict.items():
            if stID not in fDict:
                fDict[stID] = v 
            else:
                tmp = fDict[stID] + v 
                fDict[stID] = tmp

    fDict = OrderedDict(sorted(fDict.items(), key = lambda item:item[1], reverse = True)) 

    return fDict  

def make_ranking_result_dict(resultDict):
    
    ranking_dict= {}
    ranking = 0 
    for url, value in resultDict.items():
        if ranking == 0:
            last_value = value 
            ranking = ranking + 1 
            n = 1 
        else:
            if last_value != value:
                ranking = n + 1 
                n = ranking 
                last_value = value 
            else:
                n = n + 1 
        ranking_dict[url] = ranking 
        
    return ranking_dict



def make_result_dict_scoring_type(args, rPath, modelPath, qID):

    resultPath = f"{rPath}{modelPath}{qID}_sum.pickle"
    if os.path.exists(resultPath):
        print(f"{resultPath} already exists!")
        with open(resultPath, "rb") as fr:
            resultDict = pickle.load(fr)  
    else:
        print(f"Generating result file: {modelPath}")
        resultPath_tmp = f"{rPath}{modelPath}{qID}.pickle"
        if not os.path.exists(resultPath_tmp):
            return None  
        with open(resultPath_tmp, "rb") as fr:
            resultDict = pickle.load(fr)
        if len(resultDict) == 0:
            return None  
        resultDict = make_sorted_result_dict_sum(args, resultDict, qID)
        utils.save_pickle(resultPath, resultDict)
            
    return resultDict

def get_combined_ranking(ranking_resultDict, weightDict, st_key):

    ranking_list, weight_list = [], [] 
    ranking_weight_dict = {}
    total_weight = 0
    st_key = int(st_key)    
    for idx, item in enumerate(ranking_resultDict):
        modelIndex = item['modelIndex']
        rankingDict = item['rankingDict']

        model_weight = weightDict[modelIndex]
        if rankingDict == None:
            ranking_list.append(-1) 
            weight_list.append(0)
            continue  
        total_weight += model_weight 
        if st_key not in rankingDict:
            ranking_list.append(-1)
            weight_list.append(0)
            continue 
        ranking = rankingDict[st_key]
        ranking_list.append(ranking)
        weight_list.append(model_weight)
        ranking_weight_dict[modelIndex] = ranking
            
    return {'stID': st_key, 'rankingList':ranking_list, 'weightList': weight_list, 'rankingDict': ranking_weight_dict, 'total_weight': total_weight} 

def remove_value(rankingList, weight_list):
    
    n_list, n_weight = [], []
    for idx, v in enumerate(rankingList):
        if v == -1:
            continue  
        if weight_list[idx] == 0:
            continue  
        n_list.append(v)
        n_weight.append(weight_list[idx])
    return n_list, n_weight


def get_gmean_ranking(rankingList_total):
    
    resultDict, rankingDict, weightDict = {}, {}, {}
    for idx, k in enumerate(rankingList_total):
        key = k['stID']
        rankingList = k['rankingList']
        weightList = k['weightList']
        rankings, weights = remove_value(rankingList, weightList)
        if len(rankings) == 0 and len(weights) == 0:
            # this case, the forumside data is not computed with userside because of the length limit 
            continue  
        total_weight = k['total_weight']
        weight_sum = sum(weights)
        gmean_ranking = gmean(rankings, weights = weights)
        gmean_ranking = gmean_ranking * (total_weight / weight_sum)
        resultDict[key] = gmean_ranking
        rankingDict[key] = rankingList
        weightDict[key] = weights
    # the lower the better (key: ID of question, value: gmean value)
    resultDict = OrderedDict(sorted(resultDict.items(), key = lambda item:item[1], reverse = False))
    return resultDict, rankingDict, weightDict
    
def get_dict(k, rankingList):
    
    resultDict ={}
    for i, key in enumerate(k):
        resultDict[key] = rankingList[i]
        
    return resultDict 

def return_text(stackoverflowData, key):
    return stackoverflowData[key]['text']

def get_result_dict(args, qID, stackoverflowKey, scoring):
    
    resultDict = {}
    for k_idx, key in enumerate(stackoverflowKey):
        if args.using_reproduced_data == False:
            if key == qID:
                continue  
        resultDict[key] = scoring[k_idx]
    resultDict = OrderedDict(sorted(resultDict.items(), key = lambda item:item[1], reverse = True))
    return resultDict   


def get_modelPath(modelIndex, resultDict):
    if resultDict != None:
        print(f"Number of question in this result({modelIndex}): {len(resultDict)}")
    return {'modelIndex': modelIndex, 'resultDict': resultDict}

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

def extend_list(d_list):
    
    a = [] 
    for k in d_list:
        if k == None:
            continue  
        for data in k:
            if data == None:
                continue  
            a.append(data) 
    return a 

def get_number_of_model(rDict):
    
    nlist =[] 
    for idx, r in enumerate(rDict):
        if r == None:
            continue  
        nlist.append(idx + 1)
    nlist.append(14)
    print(nlist)
    return len(nlist) 

def get_model_weight(args):
    
    weightResultPathType = args.weightResultPathType
    weightResultPathDict = getattr(weightResultPath, f'weightResultPath_{weightResultPathType}')()
    weight_name = list(weightResultPathDict.values())
    return weight_name, weightResultPathDict, weightResultPathType 

def get_model(args):
    
    modelResultPathType = args.modelResultPathType 
    modelResultPathDict = getattr(modelResultPath, f"modelResultPath_{modelResultPathType}")()
    model_name =list(modelResultPathDict.values())
    return model_name, modelResultPathDict, modelResultPathType

    
def main(args):
     
    model_name, modelResultPathDict, modelResultPathType = get_model(args)
    weight_name, weightDict, weightResultPathType = get_model_weight(args)

    # path of each model 
    modelPath = list(modelResultPathDict.values())
    modelPath_index = list(modelResultPathDict.keys())
    # resultDict to rankingDict 
    
    #print(f"Path of stackoverflow Data : {stackoverflowDataPath}")
    stackoverflowData = utils.open_pickle(args.stackoverflowDataPath)
    stackoverflowKey = list(stackoverflowData.keys())
    
    ranking_dirc = f"{args.rankingPath}{args.modelResultPathType}/"
    ranking_path = f"{ranking_dirc}{args.qID}.pickle"
    
    if os.path.exists(ranking_path):
        ranking_resultDict = utils.open_pickle(ranking_path)
    else:
        os.makedirs(ranking_dirc, exist_ok=True)
        result_resultDict = list(map(make_result_dict_scoring_type, repeat(args), repeat(args.resultPath), modelPath, repeat(args.qID)))             
        result_resultDict = list(map(get_modelPath, modelPath_index, result_resultDict))
        ranking_resultDict = list(map(getattr(ranking_metric.make_ranking_dict, f"make_ranking_dict_type"), result_resultDict))
        utils.save_pickle(ranking_path, ranking_resultDict)
            
    rDict_list = []
    for rDict in ranking_resultDict:
        if rDict['rankingDict'] == None:
            rDict_list.append(0) 
        else:
            rDict_list.append(1)


    cpu_count = 40 
    pool = multiprocessing.Pool(processes=cpu_count)
    # list of ranking for each qID 
    rankingList = pool.starmap(get_combined_ranking, zip(repeat(ranking_resultDict), repeat(weightDict), stackoverflowKey))
    pool.close()
    pool.join()
    
    finalResultDict, rankingList_dict, weightDict = get_gmean_ranking(rankingList)
    rankingDict = make_ranking_result_dict(finalResultDict)
    
    print(len(rankingDict))
    print("******Relevant question for this user-side error******")
    for idx, (k, v) in enumerate(rankingDict.items()):
        if idx == 10:
            break 
        print(f"Rank {idx + 1}: https://stackoverflow.com/questions/{k}")
    
