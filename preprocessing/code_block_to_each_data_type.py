import pickle
import re
import json
from preprocessing_text import mining_code_block_in_question, mining_text_in_question, preprocessing_execution_output, preprocessing_user_operation
import argparse
import util
import os 

def combine_as_questionID(new_datas):
    
    questionDict = {}
    for k, v in new_datas.items():
        # k: url of post 
        for item in v:
            url = item['url']
            text = item['text']
            id = url.split("/")[-1]
            if id not in questionDict:
                questionDict[id] = {}
            if k == "description":
                text = text.split(".")
                questionDict[id][k] = text
            else:
                if k not in questionDict[id]:
                    questionDict[id][k] = []
                tmp = questionDict[id][k] 
                tmp.append(text)
                questionDict[id][k] = tmp 
                
    return questionDict
    
def preprocess(txt):

    txt = txt.replace('[CODE]', '').replace('[/CODE]', '').replace('[QUOTE]', '').split()    
    
    '''reg = r"[^a-zA-Z\s]"
    t_arr = []
    for t in txt:
        if len(t) == 0:
            continue
        #new_str = re.sub(reg, " ", t).strip()
        #print(new_str)
        #print(new_str)
        if len(new_str) != 0:
            t_arr.append(new_str)
    result_text = ' '.join(t_arr).lower()
    return ' '.join(result_text.split())'''
    txt = " ".join(txt)
    txt = txt.lower()
    return txt


def main(args):
    
    system = args.system 

    code_reg_s = '\s*(|(\/\/|\/\*|#+|@|[\(%\w])[\S]*([\r\n]>*[ \t]*))([\w .\/\'\"\<\>]+[\[\w\]]*[ \t]*=[\s]*\w[\w\W]+|([\w]*\w+(\.\w+)*\([\w\W]*\)[\w\W]*)|(try|public|private|class|void|int|double|float|def|while|for)[^\n]*?( [^ \r\n\(\)]*|)*(\([\w]*?\)|)[^\n]*?[\{\:] *\n[\w\W]*|(|\()(|from [\w\W]+\s+)import [\w\W]+)'
    command_reg_s = '\s*\**(|(|\[) *[\S]+\@[\S ]+(|\])|[\w\\:]*)(|\#|\$|>)\s*((mongosh|mongod|start-all|stop-all|nova-hypervisor-list)\s*[\w\.\-\*\&\@\\\r\n\s\|>]*|(sudo|su|curl|nc|nohup|hadoop|mongo|python(|3)|ssh|wget|java|[\/\w\.]*(ansible-inventory|kafka-consumer-groups|kafka-server-start)(|.sh|.bat)|openstack|[\/\w\.]*spark-submit|docker|nova|neutron|juju|kolla-ansible|cinder|systemctl|mongorestore|mongoimport|mongostore)\s+[\w\.\-\*\/\:\"\'\&\@\\\r\n\s\|>]+)'
    #codeex_reg_s = '[\s]*File [\w\W]+\, line [0-9]+\,[\w\W]+'
    codeex_reg_s = '[\s]*(File [\w\W]+\, line [0-9]+\,|[\w\.]+Caused by:)[\w\W]+'
    codeex2_reg_s = '(export \w+=[\w\W]+|[A-Z\_]+=[\w\W]+|[A-Z\_]+=[\w\W]+|\[\w+\]\n|[\w\W]*Error:[\w\W]*)'
    log_reg_s = '[\w\W]*([0-9]{2}:[0-9]{2}:[0-9]{2})[^\n\r]*(DEBUG|debug|Debug|INFO|info|Info|WARN|warn|Warn|ERROR|error|Error|FATAL|fatal|Fatal|TRACE|Trace|trace)[\w\W]+'
    msg_reg_s = '[\w\W]*(Traceback \(most recent call last\):|Can\'t validate|CRITICAL|Exception in thread|Failed:|exception|Exception[\s]*:|Exception[\s]*at|fatal[\s]*:|Fatal[\s]*:|FATAL[\s]*:|\/\'err\/\'|\/\'error\/\'|error[\s]*:|Error[\s]*:|ERROR[\s]*:|cannot find symbol[\s]*:|missing:|ERRNO[:=]|Errno[:=]|errno[:=]|unreachable|Access denied|Unable to|not authorized)[\w\W]*'
    msg2_reg_s = '[\w\W]*\(HTTP [4-5][0-9]{2}\)[\w\W]*'

    code_p = re.compile(code_reg_s)
    codeex_p = re.compile(codeex_reg_s)
    codeex2_p = re.compile(codeex2_reg_s)
    command_p = re.compile(command_reg_s)
    log_p = re.compile(log_reg_s)
    msg_p = re.compile(msg_reg_s)
    msg2_p = re.compile(msg2_reg_s)
    
    
    datas = util.load_pickle(args.input_path)
    print(f"System: {system}, Load data and classify into each data type!")
    new_datas = {'code': [], 'command': [], 'log': [], 'console': [], 'multi': [], 'description': []}
    for data in datas:
        # load list of code block 
        title = data['title']
        code_list = mining_code_block_in_question(data['text'])
        text_without_code =  mining_text_in_question(data['text'])
        if args.preprocess == True:
            text_without_code = preprocess(text_without_code)
            title = preprocess(title)

        for i, c in enumerate(code_list):
            count = 0
            if code_p.match(c) and not codeex_p.search(c) and not codeex2_p.search(c):
                if args.preprocess== True:
                    c = preprocessing_user_operation(c)
                new_datas['code'].append({'url': data['url'], 'text': c})
                count += 1
            if command_p.match(c):
                if args.preprocess== True:
                    c = preprocessing_user_operation(c)
                new_datas['command'].append({'url': data['url'], 'text': c})
                count += 2
            if log_p.match(c):
                if args.preprocess== True:
                    c = preprocessing_execution_output(c)
                new_datas['log'].append({'url': data['url'], 'text': c})
                count += 4
            elif (msg_p.match(c) or msg2_p.match(c) or codeex_p.match(c)) and 'def ' not in c:
                if args.preprocess== True:
                    c = preprocessing_execution_output(c)
                new_datas['console'].append({'url': data['url'], 'text': c})
                count += 8

            if count != 0 and count != 1 and count != 2 and count != 4 and count != 8:
                new_datas['multi'].append({'url': data['url'], 'text': c, 'where': count})
        text_without_code = title + ". " + text_without_code 
        new_datas['description'].append({'url': data['url'], 'text': text_without_code})

        #print(new_datas)

    codej = json.dumps(new_datas['code'], indent=4)
    comj = json.dumps(new_datas['command'], indent=4)
    logj = json.dumps(new_datas['log'], indent=4)
    msgj = json.dumps(new_datas['console'], indent=4)
    multij = json.dumps(new_datas['multi'], indent=4)
    tndj = json.dumps(new_datas['description'], indent = 4)

    # save as json 
    json_save_path = f"{args.output_path}_json"
    os.makedirs(json_save_path, exist_ok = True)
    util.save_json(f"{json_save_path}/cod.json", codej)
    util.save_json(f"{json_save_path}/cmd.json", comj)
    util.save_json(f"{json_save_path}/log.json", logj)
    util.save_json(f"{json_save_path}/cns.json", msgj)
    util.save_json(f"{json_save_path}/multi.json", multij)
    util.save_json(f"{json_save_path}/tnd.json", tndj)

    pickle_save_path = f"{args.output_path}"
    os.makedirs(pickle_save_path, exist_ok = True)
    util.save_pickle(f"{pickle_save_path}/cod.pickle", new_datas['code'])
    util.save_pickle(f"{pickle_save_path}/cmd.pickle", new_datas['command'])
    util.save_pickle(f"{pickle_save_path}/log.pickle", new_datas['log'])
    util.save_pickle(f"{pickle_save_path}/cns.pickle", new_datas['console'])
    util.save_pickle(f"{pickle_save_path}/tnd.pickle", new_datas['description'])
    
    questionDict = combine_as_questionID(new_datas)
    util.save_pickle(f"{pickle_save_path}/combined_qID.pickle", questionDict)

parser = argparse.ArgumentParser()
parser.add_argument("--system", required = True)
parser.add_argument("--input_path", required = True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--preprocess", action = 'store_true')
parser.add_argument("--not_preprocess", action = 'store_false', dest = "preprocess")
parser.set_defaults(preprocess = True)
args = parser.parse_args()

main(args)
