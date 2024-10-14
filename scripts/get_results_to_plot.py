import json
import os
import numpy as np

def process_json_files(json_files, item_names, output_path):
  
  data_collect = {name: {'rejection_time': [], 'power': [], 'fpr': []} for name in item_names}
  
  for file in json_files:
      with open(file, 'r') as f:
          data = json.load(f)
          for i, item in enumerate(data):  
              data_collect[item_names[i]]['rejection_time'].append(item['rejection_time'])
              data_collect[item_names[i]]['power'].append(item['power'])
              data_collect[item_names[i]]['fpr'].append(item['fpr'])
  
  # get the average results
  results = []
  for name, metrics in data_collect.items():
      avg_rejection_time = np.mean(metrics['rejection_time'], axis=0).tolist()
      avg_power = np.mean(metrics['power'], axis=0).tolist()
      avg_fpr = np.mean(metrics['fpr'], axis=0).tolist()
      results.append({
          'item_name': name,
          'rejection_time': avg_rejection_time,
          'power': avg_power,
          'fpr': avg_fpr
      })
  
  # save results
  results_json = json.dumps(results, indent=4)
  
  with open(output_path, 'w') as f:
      f.write(results_json)

  print(f'Saved average results to {output_path}')
