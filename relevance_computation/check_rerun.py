import pickle, os, sys
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--path", default="../benchmark_linkso/questionPair_code_output_python.pickle")
args = parser.parse_args()

path = args.path 
with open(path, "rb") as fr:
    data = pickle.load(fr)

fw = open('rerun.txt', "w")
for k, v in data.items():
    p = f"../bm25_rerank_stackoverflowData/{k}.pickle"
    if not os.path.exists(p):
        fw.write(str(k))
        fw.write("\n")