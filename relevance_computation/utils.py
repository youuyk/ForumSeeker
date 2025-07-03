import pickle 
import modelResultPath
import re 
import os, sys 
from nltk.corpus import stopwords 
from itertools import repeat 
from collections import OrderedDict

def strip(x):
    return x.strip()

def open_pickle(path):
    with open(path, "rb") as fr:
        d = pickle.load(fr)
    return d 

def save_pickle(path, data):
    with open(path, "wb") as fw:
        pickle.dump(data, fw)
        
        
def remove_blank(dataList):
    
    nList =[]
    for d in dataList:
        if len(d) == 0:
            continue  
        else:
            nList.append(d)
    return nList

def get_model(modelResultPathType):
    
    modelResultPathDict = getattr(modelResultPath, f"modelResultPath_{modelResultPathType}")()
    model_name =list(modelResultPathDict.values())
    return model_name, modelResultPathDict, modelResultPathType


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

def get_model_info(modelType, modelDict):
    
    p = re.compile('\(([^)]+)')
    
    modelInfo = [] 
    for modelIndex in modelType:
        modelPath = modelDict[modelIndex]
        dataType = modelPath.split("/")[0]
        userside = dataType.split("_")[0]
        if userside == "description" and dataType.split("_")[1] == "sep":
            userside = "description_sep"
        forumside = dataType.split("_")[-1]
        if forumside == "sep":
            forumside = "description_sep"
        
        prepType = modelPath.split("/")[1]
        userside_prepType = prepType.split("_")[0]
        forumside_prepType = prepType.split("_")[1]
        
        modelName = modelPath.split("/")[2]
        modelName_Type = p.findall(modelName)[0]
        modelName = modelName.split("(")[0]

        modelInfoDict = {'userside': userside, 'forumside': forumside, 'userside_preprocessing_type': userside_prepType, 'forumside_preprocessing_type': forumside_prepType, 'modelType': modelName_Type, 'model_name_or_path': modelName}
        modelInfo.append(modelInfoDict)
    return modelInfo

def load_data_for_inference_codebert(args, dataPath, dataType):
    
    if os.path.exists(dataPath):
        print("load pre-saved dataset")
        data_gt= {}
        data = open_pickle(dataPath)
        flag = False
        return data, data_gt, flag
    else:
        print("Create new dataset")
        stopword_list = stopwords.words('english') 
        man_stopPath = "stopword.txt"
        with open(man_stopPath, "r") as fr:
            man_stop = fr.readlines()
        man_stop = list(map(strip, man_stop))
        stopword_list.extend(man_stop)

        data, data_gt = getattr(preprocess_module_codebert, f"get_forumside_{dataType}")(args, stopword_list)
        save_pickle(dataPath, data)
        flag = True
        
        return data, data_gt, flag
    
def load_data_for_inference(args, dataPath, dataType):
    
    if os.path.exists(dataPath):
        print("load pre-saved dataset")
        data_gt= {}
        data = open_pickle(dataPath)
        flag = False
        return data, data_gt, flag
    else:
        print("Create new dataset")
        stopword_list = stopwords.words('english') 
        man_stopPath = "stopword.txt"
        with open(man_stopPath, "r") as fr:
            man_stop = fr.readlines()
        man_stop = list(map(strip, man_stop))
        stopword_list.extend(man_stop)

        data, data_gt = getattr(preprocess_module, f"get_forumside_{dataType}")(args, stopword_list)
        save_pickle(dataPath, data)
        flag = True
        
        return data, data_gt, flag


def load_data_for_userside(args, datatype, qID):

    userside = getattr(preprocess_module, f"get_userside_{datatype}")(args, datatype, qID)
    return userside


    
def remove_punc(word):
    punc_list = ["(", ")", "_", "-", "|", "[", "]", ">", "<", ":", "\\", "*", "{", "}", "=", ";", "rn"]
    for punc in punc_list:
        word = word.replace(punc, "")
    return word 

def preprocess_text(textList, stopword_list):    
    
    nList = [] 
    for w in textList:
        w = w.lower()
        w = w.split(" ")
        wordList = [] 
        for word in w:
            # add remove_punc 
            word = remove_punc(word)
            if word in stopword_list:
                continue   
            else:
                wordList.append(word)

        w = " ".join(wordList)        
        nList.append(w)
    return nList

def preprocess_output(output):

    # remove word in ''
    output = re.sub("'([^']*)'", "", output)
        
    new_word = [] 
    output = output.split(" ")
    for word in output:
        word = word.lower()
        word = remove_punc(word) 
        word = re.sub(r'\d', '', word)
        word = word.split(" ")
        for w in word:
            if len(w) <= 1:
                continue  
            new_word.append(w)  
    output= " ".join(new_word)
    return output 

def preprocessing_code(code, stopwords):
        
    # 1. split camel case 
    code = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', code)
    code = [m.group(0) for m in code]
    # 2. remove stopwords 
    new_code = [] 
    for word in code:
        if word in stopwords:
            continue  
        else:
            new_code.append(word)  
    new_code = " ".join(new_code)
    return new_code 



class preprocess_module_codebert:
    
    print("preprocessing data for codebert")
    
    def get_forumside_description_sep(args, stopword_list):
        
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)
        print(len(stackoverflowData))
        
        forumside ={}
        forumside_gt = {}
        for forumside_idx, forumside_data in stackoverflowData.items():
            
            if 'description' not in forumside_data:
                continue 
            
            forumside_description = forumside_data['description']
            forumside_description = preprocess_text(forumside_description, stopword_list)             

            d_list = [] 
            for d_id, description in enumerate(forumside_description):
                tot_idx = f"{forumside_idx}_{d_id}"

                d_list.append(description) 
                forumside[tot_idx] = description
                forumside_gt[forumside_idx] = d_list 
                
        return forumside, forumside_gt
    
    def get_forumside_console(args, stopword):

        print("Preprocessing console!")
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)

        forumside = {}
        forumside_gt = {}
        for forumside_qidx, forumside_data in stackoverflowData.items():
            if 'console' not in forumside_data:
                continue 
            forumside_output = forumside_data['console']
            
            d_list = [] 
            for o_idx, output in enumerate(forumside_output):
                tot_key = f"{forumside_qidx}_{o_idx}"
                output = preprocess_output(output)

                d = output.split(" ")
                d_list.append(output) 
                forumside[tot_key] = output 
            forumside_gt[forumside_qidx] = d_list 
        return forumside, forumside_gt

    def get_forumside_log(args, stopword):

        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)

        forumside = {}
        forumside_gt = {}
        for forumside_qidx, forumside_data in stackoverflowData.items():
            if 'log' not in forumside_data:
                continue  
            forumside_output = forumside_data['log']
            cnt= 0 
            d_list = [] 
            for o_idx, output in enumerate(forumside_output):
                tot_key = f"{forumside_qidx}_{o_idx}"
                output = preprocess_output(output)
                d_list.append(output) 
                forumside[tot_key] = output 
            forumside_gt[forumside_qidx] = d_list 
        return forumside, forumside_gt
    
    def get_forumside_command(args, stopword):
        
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData = open_pickle(stackoverflowDataPath)
        
        cmd_dict, cmd_gt = {}, {}
        for qidx, data in stackoverflowData.items():
            if 'command' not in data:
                continue  
            command = data['command']
            for idx, cmd in enumerate(command):
                cmd_key = f"{qidx}_{idx}"
                cmd = remove_punc(cmd)        
                cmd_dict[cmd_key] = cmd  
        return cmd_dict, cmd_gt 

    def get_forumside_code(args, stopword):
        
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)
        codeDict = {}
        forumside_gt = {}
        for qidx, data in stackoverflowData.items():
            if 'code' not in data:
                continue 
            codePart = data['code']
            if len(codePart)== 0:
                continue
            c_list = [] 
            for idx, code in enumerate(codePart):
                 
                # question_index: index of question in stackoverflow 
                # idx: order of code  
                codeDict_key = f"{qidx}_{idx}" 
                code = preprocessing_code(code, stopword)
                c_list.append(code)
                codeDict[codeDict_key] = code
            forumside_gt[qidx] = c_list   
        return codeDict, forumside_gt

class preprocess_module:
    
    def get_forumside_description_sep(args, stopword_list):
        
        print("preprocess data for description")
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)
        print(len(stackoverflowData))
        
        forumside ={}
        forumside_gt = {}
        for forumside_idx, forumside_data in stackoverflowData.items():
            
            if 'description' not in forumside_data:
                continue 
            forumside_description = forumside_data['description']
            forumside_description = preprocess_text(forumside_description, stopword_list)
                        
            d_list = [] 
            
            for d_id, description in enumerate(forumside_description):
                tot_idx = f"{forumside_idx}_{d_id}"
                d_list.append(description) 
                forumside[tot_idx] = description
                forumside_gt[forumside_idx] = d_list 
                
        return forumside, forumside_gt
    
    def get_forumside_console(args, stopword):

        print("Preprocessing console!")
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)

        forumside = {}
        forumside_gt = {}
        for forumside_qidx, forumside_data in stackoverflowData.items():
            if 'console' not in forumside_data:
                continue 
            forumside_output = forumside_data['console']
            cnt= 0 
            d_list = [] 
            for o_idx, output in enumerate(forumside_output):
                tot_key = f"{forumside_qidx}_{o_idx}"
                output = preprocess_output(output)
                d_list.append(output) 
                forumside[tot_key] = output 
            forumside_gt[forumside_qidx] = d_list 
        return forumside, forumside_gt

    def get_forumside_log(args, stopword):

        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)

        forumside = {}
        forumside_gt = {}
        for forumside_qidx, forumside_data in stackoverflowData.items():
            if 'log' not in forumside_data:
                continue  
            forumside_output = forumside_data['log']
            d_list = [] 
            for o_idx, output in enumerate(forumside_output):
                tot_key = f"{forumside_qidx}_{o_idx}"
                output = preprocess_output(output)
                d_list.append(output) 
                forumside[tot_key] = output 
            forumside_gt[forumside_qidx] = d_list 
        return forumside, forumside_gt
    
    def get_forumside_command(args,  stopword):
        
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData = open_pickle(stackoverflowDataPath)
        
        cmd_dict, cmd_gt = {}, {}
        for qidx, data in stackoverflowData.items():
            if 'command' not in data:
                continue  
            command = data['command']
            for idx, cmd in enumerate(command):
                cmd_key = f"{qidx}_{idx}"
                cmd = remove_punc(cmd)        
                cmd_dict[cmd_key] = cmd  
        return cmd_dict, cmd_gt 

    def get_forumside_code(args, stopword):
        
        stackoverflowDataPath = args.stackoverflowDataPath 
        stackoverflowData=open_pickle(stackoverflowDataPath)
        codeDict = {}
        forumside_gt = {}
        for qidx, data in stackoverflowData.items():
            if 'code' not in data:
                continue 
            # codePart = list of code 
            codePart = data['code']
            if len(codePart)== 0:
                continue
            c_list = [] 
            for idx, code in enumerate(codePart):
                codeDict_key = f"{qidx}_{idx}" 
                code = preprocessing_code(code, stopword)
                c_list.append(code)
                codeDict[codeDict_key] = code
            forumside_gt[qidx] = c_list   
        return codeDict, forumside_gt    

    def get_userside_console(args, datatype,  qID):
        
        print("Preprocessing userside console")     
        userData=open_pickle(args.userSideDataPath)
        userside_output = userData[qID][datatype]
        userside_output = list(map(preprocess_output, userside_output))
        userside_output= remove_blank(userside_output)
        return userside_output 
    
    def get_userside_command(args, datatype, qID):
        
        userData = open_pickle(args.userSideDataPath)
        userside_command = userData[qID]['command'] 
        userside_command = list(map(remove_punc, userside_command))
        return userside_command 
    
    def get_userside_log(args, datatype, qID):
            
        userData=open_pickle(args.userSideDataPath)
        userside_output = userData[qID][datatype]
        userside_output = list(map(preprocess_output, userside_output))
        
        userside_output= remove_blank(userside_output)
        return userside_output 

    def get_userside_code(args, datatype, qID):
             
        userData = open_pickle(args.userSideDataPath)
        userside_code = userData[qID]['code']

        stopword_list = stopwords.words('english') 
        
        man_stopPath = "stopword.txt"
        with open(man_stopPath, "r") as fr:
            man_stop = fr.readlines()
        man_stop = list(map(strip, man_stop))
        stopword_list.extend(man_stop)

        print(f"Load data for userside {qID}")   
        userside_code = list(map(preprocessing_code, userside_code, repeat(stopword_list)))     
        userside_code= remove_blank(userside_code)
        
        if len(userside_code) <= 5:
            return userside_code
        
        u_code = []
        for c in userside_code:
            if len(c.split(" ")) <= 5:
                continue  
            else:
                u_code.append(c)            
        return u_code
    

    def get_userside_description_sep(args, datatype, qID):

        stopword_list = stopwords.words('english') 
        man_stopPath = "stopword.txt"
        with open(man_stopPath, "r") as fr:
            man_stop = fr.readlines()
        man_stop = list(map(strip, man_stop))
        stopword_list.extend(man_stop)

        userData = open_pickle(args.userSideDataPath)
        userside_description = userData[qID]['description']
        userside_description = userside_description.split(".")
        userside_description = preprocess_text(userside_description, stopword_list)    
        userside_description= remove_blank(userside_description)
        return userside_description


#make result dict for crossencoder, biencoder 

def make_result_dict_crossencoder(keyList, relevance_value):
    
    result = {}
    for kidx, key in enumerate(keyList):
        result[key] = relevance_value[kidx].item()
    return result 

def make_result_dict_biencoder(qID, userside, forumsideKey, relevance_value):
    
    resultDict = {}
    for sidx, sent in enumerate(userside):
        qID_index = f"{qID}_{sidx}"
        relVal = relevance_value[sidx]
        print(f"Number of relevance value: {len(relVal)}")
        for fidx, forumsideKey_key in enumerate(forumsideKey):
            tot_index = f"{qID_index}_{forumsideKey_key}"
            resultDict[tot_index] = relVal[fidx].item()
    return resultDict


def saveResult(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, resultDict):
        
    print(f"Total number of result: {len(resultDict)}")
    resultDirc = f"{args.resultPath}{usersideDataType}_{forumsideDataType}/{userside_preprocessing_type}_{forumside_preprocessing_type}/{model_name_or_path}({modelType})/"

    print("Sorting result") 
    # highest value should be go first 
    resultDict = OrderedDict(sorted(resultDict.items(), key = lambda item:item[1], reverse = True))
    
            
    os.makedirs(resultDirc, exist_ok=True)
    resultPath = f"{resultDirc}{qID}.pickle"
    with open(resultPath, "wb") as fw:
        pickle.dump(resultDict, fw)  
            
def saveData(args, usersideDataType, forumsideDataType, userside_preprocessing_type, forumside_preprocessing_type, modelType, model_name_or_path, qID, userside, forumside):
    
    saveDataDirc = f"{args.usedDataPath}{usersideDataType}_{forumsideDataType}/{userside_preprocessing_type}_{forumside_preprocessing_type}/{model_name_or_path}({modelType})/"
    
    os.makedirs(saveDataDirc, exist_ok=True)
    
    usersideDict = {}
    for idx, d in enumerate(userside):
        key = f"{qID}_{idx}"
        usersideDict[key] = d 
    saveDataPath =f"{saveDataDirc}{qID}_user.pickle"
    with open(saveDataPath, "wb") as fw:
        pickle.dump(usersideDict, fw)
            
    saveDataPath = f"{saveDataDirc}{qID}_forum.pickle"
    with open(saveDataPath, "wb") as fw:
        pickle.dump(forumside, fw)
