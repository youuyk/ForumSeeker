
def modelResultPath_fdd_code_code_added():
    
    modelResultPath = {
        1: "code_description_sep/4_3/microsoft_codebert-base_t1(CodeBERT)/", 
        2: "code_log/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/",
        3: "code_console/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/", 
        4: "description_sep_code/4_4/microsoft_codebert-base_t1(CodeBERT)/",
        5: "log_code/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/",
        6: "console_code/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/", 
        7: "log_log/4_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        8: "log_description_sep/4_3/all-MiniLM-L6-v2(SentenceBERT)/", 
        9: "description_sep_log/3_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        10:"description_sep_description_sep/3_3/all-MiniLM-L6-v2(SentenceBERT)/", 
        11:"description_sep_console/3_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        12:"console_description_sep/4_3/all-MiniLM-L6-v2(SentenceBERT)/", 
        13:"console_console/4_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        14:"description_sep_command/4_4/BM25Okapi(BM25)/", 
        15:"command_description_sep/4_4/BM25Okapi(BM25)/", 
        16:"command_command/4_4/BM25Okapi(BM25)/",
        17:"code_code/5_5/Salesforce_codet5p-110m-embedding(CodeT5+)/",
    }
    return modelResultPath


# change codebert, graphcodebert to codet5p
def modelResultPath_fdd_codet5p_graphcodebert_codebert_final():
    
    modelResultPath = {
        1: "code_description_sep/4_3/microsoft_codebert-base_t1(CodeBERT)/", 
        2: "code_log/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/",
        3: "code_console/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/", 
        4: "description_sep_code/5_5/Salesforce_codet5p-110m-embedding(CodeT5+)/",
        5: "log_code/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/",
        6: "console_code/4_4/microsoft_graphcodebert-base_t1(GraphCodeBERT)/", 
        7: "log_log/4_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        8: "log_description_sep/4_3/all-MiniLM-L6-v2(SentenceBERT)/", 
        9: "description_sep_log/3_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        10:"description_sep_description_sep/3_3/all-MiniLM-L6-v2(SentenceBERT)/", 
        11:"description_sep_console/3_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        12:"console_description_sep/4_3/all-MiniLM-L6-v2(SentenceBERT)/", 
        13:"console_console/4_4/all-MiniLM-L6-v2(SentenceBERT)/", 
        14:"description_sep_command/4_4/BM25Okapi(BM25)/", 
        15:"command_description_sep/4_4/BM25Okapi(BM25)/", 
        16:"command_command/4_4/BM25Okapi(BM25)/",
        17:"code_code/5_5/Salesforce_codet5p-110m-embedding(CodeT5+)/",
    }
    return modelResultPath

