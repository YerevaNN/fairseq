import pandas as pd
import json


with open('/home/gayane/BartLM/fairseq/scripts/datasets.json', 'r+') as f:
    datasets_json = json.load(f)

# dataset = 'pcba'
# dataset_js = datasets_json[dataset]

# for i in range(len(dataset_js["class_index"])):
#     datasets_json[f"{dataset}_{i}"] = datasets_json[dataset]
#     datasets_json[f"{dataset}_{i}"]["class_index"] = [-2]
#     jsonString = json.dumps(datasets_json)
#     jsonFile = open('/home/gayane/BartLM/fairseq/scripts/datasets.json', "w")
#     jsonFile.write(jsonString)
#     jsonFile.close()


# for key in datasets_json.keys():
#     if datasets_json[key]["type"] == "classification":
#         datasets_json[key]["return_logits"] = False
#     else:
#         datasets_json[key]["return_logits"] = True
#     jsonString = json.dumps(datasets_json, sort_keys=True, indent=3)
#     jsonFile = open('/home/gayane/BartLM/fairseq/scripts/datasets.json', "w")
#     jsonFile.write(jsonString)
#     jsonFile.close()