import json
import math
from ujson import load as json_load
import pandas as pd

baseline_result = './save/test/baseline-dev-01/dev_submission.csv'
full_BiDAF = './save/test/full-BiDAF-dev-01/submission.csv'
wiq_BiDAF = './save/test/wiq-BiDAF-dev-01/submission.csv'
self_att_wiq_BiDAF = './save/test/self_att-wiq-BiDAF-dev-01/submission.csv'
dev_results = [baseline_result, full_BiDAF, wiq_BiDAF, self_att_wiq_BiDAF]
dev_eval_file = './data/dev_eval.json'

def qual_analysis():
  with open(dev_eval_file, 'r') as fh:
    gold_dict = json_load(fh)  
  
  question_categories = {'what': 0, 
                         'who': 0, 
                         'when': 0,
                         'where': 0,
                         'why': 0,
                         'which': 0,
                         'how': 0,
                         'other': 0}
  
  question_EMs = [{'what': 0, 'who': 0, 'when': 0, 'where': 0, 'why': 0, 'which': 0, 'how': 0, 'other': 0},
                  {'what': 0, 'who': 0, 'when': 0, 'where': 0, 'why': 0, 'which': 0, 'how': 0, 'other': 0},
                  {'what': 0, 'who': 0, 'when': 0, 'where': 0, 'why': 0, 'which': 0, 'how': 0, 'other': 0},
                  {'what': 0, 'who': 0, 'when': 0, 'where': 0, 'why': 0, 'which': 0, 'how': 0, 'other': 0}]
  
  for idx in gold_dict:
    question = gold_dict[idx]['question'].lower()
    if 'what' in question:
      question_categories['what'] += 1
    elif 'who' in question or 'whose' in question:
      question_categories['who'] += 1
    elif 'when' in question:
      question_categories['when'] += 1
    elif 'where' in question:
      question_categories['where'] += 1
    elif 'why' in question:
      question_categories['why'] += 1
    elif 'which' in question:
      question_categories['which'] += 1
    elif 'how' in question:
      question_categories['how'] += 1
    else:
      question_categories['other'] += 1
  
  for i, result in enumerate(dev_results):  
    df = pd.read_csv(result)
    ids = df['Id'].values.tolist()
    predicted_answer = df['Predicted'].values.tolist()
    for idx in gold_dict:
      question = gold_dict[idx]['question'].lower()
      uuid = gold_dict[idx]['uuid']
      answers = gold_dict[idx]['answers']
      for j in range(len(ids)):
        if type(predicted_answer[j]) == float and math.isnan(float(predicted_answer[j])):
          predicted_answer[j] = []
        
        if ids[j] == uuid:
          if answers == [] and predicted_answer[j] == []:
            if 'what' in question:
              question_EMs[i]['what'] += 1
            elif 'who' in question or 'whose' in question:
              question_EMs[i]['who'] += 1
            elif 'when' in question:
              question_EMs[i]['when'] += 1
            elif 'where' in question:
              question_EMs[i]['where'] += 1
            elif 'why' in question:
              question_EMs[i]['why'] += 1
            elif 'which' in question:
              question_EMs[i]['which'] += 1
            elif 'how' in question:
              question_EMs[i]['how'] += 1
            else:
              question_EMs[i]['other'] += 1
          elif predicted_answer[j] in answers:
            if 'what' in question:
              question_EMs[i]['what'] += 1
            elif 'who' in question or 'whose' in question:
              question_EMs[i]['who'] += 1
            elif 'when' in question:
              question_EMs[i]['when'] += 1
            elif 'where' in question:
              question_EMs[i]['where'] += 1
            elif 'why' in question:
              question_EMs[i]['why'] += 1
            elif 'which' in question:
              question_EMs[i]['which'] += 1
            elif 'how' in question:
              question_EMs[i]['how'] += 1
            else:
              question_EMs[i]['other'] += 1

  for em in question_EMs:
    for category in question_categories:
      em[category] /= em[category]
      em[category] *= 100   

if __name__ == '__main__':
  qual_analysis()