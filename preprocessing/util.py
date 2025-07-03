import pickle 

def save_pickle(path, data):
    with open(path, "wb") as fw:
        pickle.dump(data, fw)

def load_pickle(path):
    with open(path, "rb") as fr:
        data = pickle.load(fr)
    
    return data 

def save_json(path, data):
    with open(path, "w") as fw:
        fw.write(data)