# ForumDR
ForumDR: Curating Online Forum Knowledge as Troubleshooting Dataset for Generative AI Using Fusion Retrieval 

### Organization
---
```
|--failure_case                         #Directory to save failure case data.
|--data_stackoverflow                   #Example stackoverflow data is stored. 
|--preprocessing                        #Directory to preprocess stackoverflow data. 
|--relevance_computation                #Computing relevance between classified stackoverflow data 
```

### 🖥 Running ForumDR
---
```
git clone https://gitlab.com/dbdid0064/forumdr.git
cd forumdr 
pip install -r requirements.txt 
git clone https://huggingface.co/yykimyykim/forumdr-GraphCodeBERT-code-output
```
#### Step1. Preprocessing 
The example data is stored in ```./data_stackoverflow/``` directory. 

With this step, the input stackoverflow data will be classified into one of the data type ['code', 'log', 'command', 'console output', 'description]. 

To run this step,
```python 
cd preprocessing
python3 code_block_to_each_data_type.py --system openstack --input_path ../data_stackoverflow/openstack_QAset_tagged.pickle --output_path ../data_stackoverflow/openstack/
```
The outcome:
```
|--data_stackoverflow/
    |--openstack                # Directory for classified stackoverflow data 
        |--_json                # Directory where classified data saved as json format 
        |--cmd.pickle           # Classified "Command" data saved as pickle 
        |--cns.pickle           # Classified "Console output" data saved as pickle 
        |--cod.pickle           # Classified "Code" data saved as pickle 
        |--log.pickle           # Classified "Log" data saved as pickle 
        |--tnd.pickle           # Classified "Description" data saved as pickle 
        |--combined_qID.pickle  # Classified data saved with stackoverflow question id 
``` 

#### Step2. Relevance Computation 

Computing the relevance between stackoverflow data and failure case data. 

The example failure case data is stored in ```./failure_case/``` directory. 

With this step, we compute the relevance between classified stackoverflow data and failure case data from failure system. 

The fine-tuned models stored in huggingface will be loaded in this step. (We fine-tuned the model based on the publicly released code.)

To run this step, 
```python
cd relevance_computation
python3 run_relevance.py --system openstack --stackoverflowDataPath ../data_stackoverflow/openstack/combined_qID.pickle --userSideDataPath ../failure_case/openstack.pickle --batch_size 512 
```

The outcome when computing relevance:  
```
CodeBERT
{'userside: 'code', 'forumside': 'description_sep', 'modelType': 'CodeBERT', 'model_name_or_path': 'microsoft_codebert-base_t1'}
Running CodeBERT 
...
```
The outcome after computing relevance: 
```
******Relevant question for this user-side error******
Rank 1: https://stackoverflow.com/questions/45511739
Rank 2: https://stackoverflow.com/questions/63650069
Rank 3: https://stackoverflow.com/questions/54463543 
...
Rank 10: https://stackoverflow.com/questions/45290219
``` 

### 🔈Running ForumDR with custom data 
---
To run ForumDR with custom data, the data should be saved as pickle. 
First, save user-side data at ```./failure_case/``` directory. 
```
|--failure_case         # Directory to save user-side data from failure system 
    |--manual_failure_case
	|-1            
      	    |-code          # User-side code from failure system.
            |-log           # User-side log from failure system.
            |-console       # User-side console output from failure system. 
            |-command       # User-side command from failure system. 
            |-description   # User-side description from failure system. 
```

Next, run ```failure_case_to_pickle.py```.
```
python3 failure_case_to_pickle.py --system openstack --path ./failure_case/manual_failure_case --savePath ./failure_case/
``` 

The outcome:
```
|--failure_case             # Directory to save user-side data. 
    |--openstack_manual.pickle     # Pickle file for user-side data.
```

After generating file, follow the step in "Running ForumDR" 
