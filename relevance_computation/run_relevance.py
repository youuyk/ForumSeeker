import argparse
import run_relevance_model 
import combine_result 
import utils 
import os 
from itertools import repeat 

parser = argparse.ArgumentParser()
parser.add_argument("--system", default = "openstack")
parser.add_argument("--device_id", default=0, type = int)
parser.add_argument("--qID", type= int)
# stackoverflow datapath is depending on question id
parser.add_argument("--stackoverflowDataPath", default = "../data_stackoverflow/openstack/combined_qID.pickle")
# path of reproduced dataset 
parser.add_argument("--userSideDataPath", default = "../data_benchmark/openstack.pickle")
parser.add_argument("--datasetPath", default = "../data_inference/")
parser.add_argument("--resultPath", default = "../result_relevance/")
parser.add_argument("--usedDataPath", default = "../result_usedData/")
parser.add_argument("--rankingPath", default = "../ranking/")
parser.add_argument("--modelResultPathType", type = str, default="fdd_code_code_added")
parser.add_argument("--weightDict", default="fdd_code_code_added")
parser.add_argument("--weightResultPathType", type = str, default='fdd_code_code_added')
parser.add_argument("--batch_size", default=256, type = int)
parser.add_argument("--length_limit_code", default = 360, type = int)
parser.add_argument("--length_limit_output", default = 64, type = int)
parser.add_argument("--length_limit_des", default = 512, type = int)
args = parser.parse_args()

system = args.system 

keyName = ["code", "description", "log", "console", "command"]
userSideData = utils.open_pickle(args.userSideDataPath) 
st_DataPath = args.stackoverflowDataPath
stData= utils.open_pickle(st_DataPath)
stackoverflowKey = list(stData.keys())

for idx, (qID, _) in enumerate(userSideData.items()):

    #args.stackoverflowDataPath = st_DataPath
    args.qID= qID
    run_relevance_model.main(args, qID, keyName)
    combine_result.main(args)
    