# preprocessing 
# 1) remove variables in ''
# 2) regex for date, time 
# 3) variable including numbers 
# 4) remove hyperlinks 
import regex as re 
from string import ascii_lowercase
import nltk 
from nltk import sent_tokenize
    

def remove_blank(text):
    
    word = text.split(" ")
    n_word = []
    for w in word:
        if len(w) == 0:
            continue   
        else:
            n_word.append(w)
    return n_word

def only_remain_alpha(text):
    
    pos_punc = [".", ",", ":", "!", "?", ";", '"', "{", "}", "[", "]", "(", ")", "<", ">", "_", "-", "'", "’", "‘", "/"]
    flag_punc = [".", ",", ":", "!", "?", ";", "_"]
    words = text.split()
    word_list = []
    for word in words:
        n_word = word 
        flag = False 
        for punc in pos_punc:
            if punc in word:
                if punc in flag_punc:
                    flag = True
                n_word = n_word.replace(punc, " ")
                #check word isalpha after removing pos_punc 
            #word just including alpha and pos_punc
        n_word = remove_blank(n_word)
        not_word = []
        for idx, w in enumerate(n_word):
            if not w.isalpha():
                not_word.append(idx)
        #if word including numbers or unnecessary characters 
        n_word = word 
        if flag == True:
            for punc in pos_punc:
                if punc in flag_punc:
                    continue   
                else:
                    n_word = n_word.replace(punc, " ")
        else:
            for punc in pos_punc:
                n_word = n_word.replace(punc, " ")
        n_word = remove_blank(n_word)
        for s_idx, w in enumerate(n_word):
            if s_idx in not_word:
                continue 
            word_list.append(w)
                
    text = ' '.join(word_list)
    return text 

def remove_path(text):
    regex_list = ["\/[\S]+\/[\S]+"]
    for regex in regex_list:
        text = re.sub(regex, "", text)
    return text 

def length_of_text(text):
    words = text.split()
    return len(words)

def check_not_path(text):
    
    word_list = []
    words = text.split()
    for word in words:
        if len(word) < 2:
            continue 
        if word[0] == "/":
            if word[1] == "'":
                word_list.append(word)
        else:
            word_list.append(word)
        
    text = ' '.join(word_list)
    return text  

def remove_word(text):
    word_list = []
    un_word_list = ["req", "info", "error", "warning", "default", "pid"]
    word = text.split()
    for w in word:
        if w in un_word_list:
            continue   
        else:
            word_list.append(w)
    text = ' '.join(word_list)
    return text 
# when whole lines of execution output is given 
# text is concatenated 
def preprocessing_execution_output(text):
    
    text = quote_mark_to_quotation_mark(text)
    text = text.lower()
    text = remove_path(text)
    text = remove_hyperlink(text)
    text = remove_date_and_time(text)
    text = only_remain_alpha(text)
    length = length_of_text(text)
    text = remove_word(text)
    return text

def preprocessing_user_operation(text):
    text = quote_mark_to_quotation_mark(text) 
    text = text.lower()
    text = check_not_path(text)
    #text = remove_path(text)
    text = remove_hyperlink(text) 
    text = remove_date_and_time(text)
    text = only_remain_alpha(text) 
    text = remove_word(text)
    return text 

def tokenize_into_sentence(input_part):
    sent_list = []
    for input in input_part:
        sent = sent_tokenize(input)
        for s in sent:
            sent_list.append(s)
    return sent_list 

#whole code block as text 
def datetime_in_codeblock_then_split(text):
    #find date time in codeblock 
    date_regex = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
    date_idx_list = re.finditer(date_regex, text)
    #if date_regex is in text 
    text_list = []
    if date_idx_list != None:
        s_idx = 0 
        for d_idx, date_idx in enumerate(date_idx_list):
            start = date_idx.start()
            end = date_idx.end() 
            code_text = text[s_idx: start]
            s_idx = end + 1 
            text_list.append(code_text)
        code_text = text[s_idx: len(text)]
        text_list.append(code_text)
    else:
        text_list.append(text)      
    return text_list   

# if "." or ":" in words, do not remove this word (in system output, word such as "keystone.auth." or "found:" can be exists)
# remove word with another characters, then remove (e.g. "_", "-", "!", ...)
def text_number_variable(text):

    possible_punc = [".", ":"]
    text_list = []    
    for sent in text:
        sent_list = []
        word_list = sent.split(" ")
        for word in word_list:
            for pos in possible_punc:
                if pos in word:
                    sent_list.append(word)
                    break 
                continue 
            if not word.isalnum():
                continue    
            sent_list.append(word)
        sent = ' '.join(sent_list)
        text_list.append(sent)
    return text_list

def text_number_variable_log(sent):
    
    possible_punc = [".", ":"]
    sent_list = []
    word_list = sent.split(" ")
    for word in word_list:
        for pos in possible_punc:
            if pos in word:
                sent_list.append(word)
                break 
            continue 
        if not word.isalnum():
            continue    
        sent_list.append(word)
    sent = ' '.join(sent_list)
    return sent


#one code block => tokenize into multiple sentences
def one_code_block_tokenize_into_sentences(text):
    all_sent_list = []
    for sent in text:
        sent_list = []
        sent = sent.split("\n")
        for s in sent:
            '''if "at" in s:
                at_s = s.split(" at")
                for at in at_s:
                    sent_list.append(at.strip())
                continue   '''
            '''if "-" in s:
                at_s = s.split("-")
                for at in at_s:
                    sent_list.append(at.strip())
                continue '''
            sent_list.append(s)
        all_sent_list.extend(sent_list)
    return all_sent_list

def remove_hyperlink_in_sentence(text):
    sent_list = []
    for sentence in text:
        word_list = sentence.split(" ")
        text_list = []
        for word in word_list:
            if "https" in word or "http" in word:
                continue
            text_list.append(word)
        text = ' '.join(text_list)
        sent_list.append(text)
    return sent_list 

#text : list of sentence in one code block 
def remove_variable_in_parenthesis_and_date_remain_without_req(text):
    sent_list = []
    regex_list = ["\'\[\w\W]*?\'", "\"\[\w\W]*?\"", "\([\w\W]*?\)", "\(host[\w\W]*?\)", "\‘[\w\W]*?\’", "´[\w\W]*?\´", "\[[\w\W]*?\]", "\<[\w\W]*?\>", "\"[\w\W]*?\"", "&lt;[\w\W]*?&gt;", "[0-9]{4}-[0-9]{2}-[0-9]{2}", "[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]*", "https:[\w\W]*? ", "http:[\w\W]^? ", "urllib[\w\W]*?:", "\/[\S]+\/[\S]+"]
    #for each sentence in code block 
    for sent in text:
        for regex in regex_list:
            sent = re.sub(regex, '', sent)
        sent = sent.replace("[QUOTE]", "")
        sent_list.append(sent)
    return sent_list 

def remove_variable_in_parenthesis_and_date_remain_without_req_by_sent(sent):
    regex_list = ["\'\[\w\W]*?\'", "\"\[\w\W]*?\"", "\([\w\W]*?\)", "\(host[\w\W]*?\)", "\‘[\w\W]*?\’", "´[\w\W]*?\´", "\[[\w\W]*?\]", "\<[\w\W]*?\>", "\"[\w\W]*?\"", "&lt;[\w\W]*?&gt;", "[0-9]{4}-[0-9]{2}-[0-9]{2}", "[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]*", "https:[\w\W]*? ", "http:[\w\W]^? ", "urllib[\w\W]*?:", "\/[\S]+\/[\S]+"]
    #for each sentence in code block 
    for regex in regex_list:
        sent = re.sub(regex, '', sent)
        sent = sent.replace("[QUOTE]", "")
    return sent


def remove_number_words(text):
    punc_list = ".!@#$%^&*()[]<>,/\/{/}_-=+:?`\"\'’‘|;´"
    sent_list = []
    #sentence in text part 
    for sent in text:
        #split into words 
        text = sent.split()
        word_list = []
        # word in text
        for word in text:
            w = word 
            # replace each punc to blank
            for punc in punc_list:
                w = w.replace(punc, "")
            if w.isdigit() or "zip" in word:
                continue     
            else:
                word_list.append(word)
        new_word = ' '.join(word_list)
        sent_list.append(new_word)
    
    return sent_list 
                
#one code block is given as text 
#multiple code block can be in one question 
#return preprocessed sentence as list 
def preprocessing_code_block_to_sentence(text):
    
    text = datetime_in_codeblock_then_split(text)
    text = one_code_block_tokenize_into_sentences(text)
    text = tokenize_into_sentence(text)
    text = remove_hyperlink_in_sentence(text)
    text = remove_variable_in_parenthesis_and_date_remain_without_req(text)
    text = text_number_variable(text)
    text = remove_number_words(text)
    
    #list of sentence tokenized 
    return text 
    
def remove_log_with_traceback(text):
    check_list = [".py", "/", "*args", "*kwargs"]

    if "traceback" in text:
        word_list = []
        text = text.split(" ")
        for word in text:
            for c in check_list:
                if c in word:
                    continue    
            word_list.append(word)
        text = ' '.join(word_list)
    return text                         
    
#preprocessing for log 
def preprocessing_logs(text):
    
    text = remove_hyperlink(text)
    text = remove_date_and_time(text)

    text = remove_variable_in_parenthesis_and_date_remain_without_req_by_sent(text)
    text = text_number_variable_log(text)
    text = remove_variables_including_numbers(text)
    text = to_lower_case(text)
    text = remove_variables_only_un_character(text)
    #text = remove_log_with_traceback(text)
    return text 

def mining_text_in_question(text):
    
    regex_list = ["\[CODE\][\w\W]*?\[\/CODE\]"]
    for regex in regex_list:
        result = re.sub(regex, " ", text)
        text = result 
        text = remove_hyperlink(text)
        text = remove_date_and_time(text)
        text = remove_variable_in_parenthesis_and_date_remain_without_req_by_sent(text)
        text = remove_variables_including_numbers(text)
        text = remove_variables_only_un_character(text)
    return text 
        
def mining_code_block_in_question(text):
    code_list = []
    regex_list = ["\[CODE\][\w\W]*?\[\/CODE\]"]
    for regex in regex_list:
        code_block = re.finditer(regex, text)
        if code_block != None:
            for code in code_block:
                start = code.start()
                end = code.end()
                c = text[start:end]
                c = code_mark_remove(c)
                c = quote_mark_to_quotation_mark(c)
                code_list.append(c)
    return code_list 

def code_mark_remove(text):
    code = "[CODE]"
    r_code = "[/CODE]"
    if code in text:
        text = text.replace(code, "")
    if r_code in text:
        text = text.replace(r_code, "")
    return text 

def quote_mark_to_quotation_mark(text):
    quote = "[QUOTE]"
    if quote in text:
        text = text.replace(quote, "/'")
    return text 

def preprocessing_classifier_dataset(text):
    #text = remove_variables_in_quotation_mark(text)
    text = remove_hyperlink(text)
    text = remove_date_and_time(text)
    text = remove_variables_including_numbers(text)
    text = to_lower_case(text)
    text = remove_variables_only_un_character(text)
    return text 

# remove variables in ' ' or " "
def remove_variables_in_quotation_mark(text):
    regex_list = ["\'[\w\W]*?\'", "\"[\w\W]*?\""]
    for regex in regex_list:
        text = re.sub(regex, '', text)
    return text 

# remove hyperlinks in text 
def remove_hyperlink(text):
    text_word = text.split()
    word_list = []
    for word in text_word:
        if "http:" in word or "https:" in word:
            continue
        word_list.append(word)
    
    cleaned_text = ' '.join(word_list)
    return cleaned_text

#remove date and time in text 
def remove_date_and_time(text):
    # regex for date (e.g. 2022-02-10), time (e.g. 10:10:10)
    regex_list = ["[0-9]{4}-[0-9]{2}-[0-9]{2}", "[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]*", "[0-9]{2}:[0-9]{2}:[0-9]{2}"]
    for regex in regex_list:
        text = re.sub(regex, '', text)
    return text 

def convert_to_string(data):
    return str(data)

#if word is including numbers -> then remove 
def remove_variables_including_numbers(text):
    text_word = text.split()
    number_list = [str(i) for i in range(1, 10)]
    word_list = []
    for word in text_word:
        flag = False 
        if "keystoneauth1" in word:
            word_list.append(word)
        for number in number_list:
            if number in word:
                flag = True 
                break 
        if flag == False:
            word_list.append(word)
        
    cleaned_text = ' '.join(word_list)
    return cleaned_text 

def to_lower_case(text):
    text = text.lower()
    return text 
    
def remove_variables_only_un_character(text):
    text_word = text.split()
    word_list = []
    number_list = [str(i) for i in range(1, 10)]
    alpha_list = list(ascii_lowercase)
    for word in text_word:
        flag = False 
        for number in number_list:
            if number in word:
                flag = True 
                break 
        for alpha in alpha_list:
            if alpha in word:
                flag = True 
                break 
        if flag == True:
            word_list.append(word)
    cleaned_text = ' '.join(word_list)
    return cleaned_text
