import pickle, sys, os 
import argparse

def main(args):
    
    dataType= ["code","log","console", "description", "command"]
    dataDict = {}
    usersideDataPath = args.path
    dircs = os.listdir(usersideDataPath)

    for dirc in dircs:
        files = os.listdir(f"{usersideDataPath}{dirc}")
        dataDict[dirc] = {}
        for file in files:
            filepath = f"{usersideDataPath}{dirc}/{file}"
            with open(filepath, "r") as fr:
                data = fr.readlines()
            if file == "description":
                data = data[0]
            dataDict[dirc][file] = data  
        for t in dataType:
            if t not in dataDict[dirc]:
                dataDict[dirc][t] = [] 

    with open(f'./data_benchmark/{args.system}_manual.pickle', "wb") as fw:
        pickle.dump(dataDict, fw)

parser = argparse.ArgumentParser()
parser.add_argument("--system", default = 'openstack')
parser.add_argument("--path", default = "./data_userside/")
parser.add_argument("--savePath", default = "./data_benchmark/")
args = parser.parse_args()

main(args)
