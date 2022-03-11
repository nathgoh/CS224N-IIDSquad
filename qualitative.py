from ujson import load as json_load
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

baseline_result = './save/test/baseline-dev-01/dev_submission.csv'
full_BiDAF = './save/test/full-BiDAF-dev-01/submission.csv'
wiq_BiDAF = './save/test/wiq-BiDAF-dev-01/submission.csv'
self_att_BiDAF = './save/test/self_att-BiDAF-dev-01/dev_submission.csv'
self_att_wiq_BiDAF = './save/test/self_att-wiq-BiDAF-dev-01/submission.csv'
dev_results = [baseline_result, full_BiDAF, wiq_BiDAF, self_att_BiDAF, self_att_wiq_BiDAF]
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
    for category in question_categories.keys():
      em[category] /= question_categories[category]
      em[category] *= 100   
  
  em_data = {'Baseline': [question_EMs[0]['what'], question_EMs[0]['who'], question_EMs[0]['when'], question_EMs[0]['where'], question_EMs[0]['why'], question_EMs[0]['which'], question_EMs[0]['how'], question_EMs[0]['other']], 
             'Char Embeddings': [question_EMs[1]['what'], question_EMs[1]['who'], question_EMs[1]['when'], question_EMs[1]['where'], question_EMs[1]['why'], question_EMs[1]['which'], question_EMs[1]['how'], question_EMs[1]['other']], 
             'Char Embeddings + Wiq': [question_EMs[2]['what'], question_EMs[2]['who'], question_EMs[2]['when'], question_EMs[2]['where'], question_EMs[2]['why'], question_EMs[2]['which'], question_EMs[2]['how'], question_EMs[2]['other']], 
             'Self-Attention': [question_EMs[3]['what'], question_EMs[3]['who'], question_EMs[3]['when'], question_EMs[3]['where'], question_EMs[3]['why'], question_EMs[3]['which'], question_EMs[3]['how'], question_EMs[3]['other']],
             'Char Embeddings + Wiq + Self-Attention': [question_EMs[4]['what'], question_EMs[4]['who'], question_EMs[4]['when'], question_EMs[4]['where'], question_EMs[4]['why'], question_EMs[4]['which'], question_EMs[4]['how'], question_EMs[4]['other']]}
  
  bar_df = pd.DataFrame(em_data, columns=['Baseline', 'Char Embeddings', 'Char Embeddings + Wiq', 'Self-Attention', 'Char Embeddings + Wiq + Self-Attention'],
                        index=['What', 'Who', 'When', 'Where', 'Why', 'Which', 'How', 'Other'])
  bar_df.plot.bar()
  plt.rc('font', size=16)
  plt.xlabel('Question Type')
  plt.ylabel('EM')
  plt.show()

  # question_type = list(question_categories.keys())
  # count = list(question_categories.values())
  
  # plt.bar(range(len(question_categories)), count, tick_label=question_type)
  # plt.xlabel('Question Type')
  # plt.ylabel('Count')
  # plt.show()

if __name__ == '__main__':
  qual_analysis()