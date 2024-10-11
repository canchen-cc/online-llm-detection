import argparse
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from methods_scenario2 import betting_experiment, seq_perm_test_experiment
from fractions import Fraction

def load_data(file_path, data_type):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['predictions'][data_type]  #data_type='real'/'samples'
    except KeyError:
        print(f"Error: {data_type} not found in {file_path}")
        return None
    except IOError:
        print(f"Error: Could not read {file_path}")
        return None

def show_score_function(filepath):
    filename = os.path.basename(filepath)
    part = filename.split('.')[-2]
    return part

def process_files(file1, type1, file2, type2, file3, type3, k, alphas, iters, shift_time):
    data1 = load_data(file1, type1)
    data2 = load_data(file2, type2)
    data3 = load_data(file3, type3)
    
    if data1 is None or data2 is None:
        return None
        
    #every time we test one two audits
    min_len = min(len(data1), len(data2), len(data3))
    y1 = data1[:min_len] 
    y2 = data2[:min_len]
    y3 = data3[:min_len]
    z1 = y1
    z2 = y3

 
    perm_k_tau, perm_k_tpr = seq_perm_test_experiment(y1, y2, alphas, iters, k=k, bonferroni=False)
    _, perm_k_fdr = seq_perm_test_experiment(z1, z2, alphas, iters, k=k, bonferroni=False)
    
    part=show_score_function(file1)
    
    return {
        "score_function": part,
        "batch_size": k,
        "rejection_time": np.ceil(np.mean(perm_k_tau, axis=0)), #rejection time
        "power": np.mean(perm_k_tpr, axis=0),#true=1/false=0
        "fpr": np.mean(perm_k_fdr, axis=0) #type-1 error
    }



def convert_np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    elif isinstance(obj, dict):
        return {key: convert_np_to_list(value) for key, value in obj.items()}  
    elif isinstance(obj, list):
        return [convert_np_to_list(item) for item in obj]  
    else:
        return obj  

def fraction_type(x):
    try:
        return float(Fraction(x))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid fraction value: {x}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--type1', type=str, required=True, choices=['real', 'samples'])
    parser.add_argument('--file2', type=str, required=True)
    parser.add_argument('--type2', type=str, required=True, choices=['real', 'samples'])
    parser.add_argument('--file3', type=str, required=True)
    parser.add_argument('--type3', type=str, required=True, choices=['real', 'samples'])
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--alphas', type=str, default="0.005,0.1,20")  
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--shift_time', type=str, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    # Convert alphas parameter from string to numpy linspace array
    start, stop, num = map(float, args.alphas.split(','))
    alphas = np.linspace(start, stop, int(num))

    results = process_files(args.file1, args.type1, args.file2, args.type2, args.file3, args.type3, args.k, alphas, args.iters, args.shift_time)
    
 
   
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [] 
            except json.JSONDecodeError:
                data = []  
    else:
        data = []
    
 
    converted_results = convert_np_to_list(results)
    data.append(converted_results)
    

    with open(args.output_file, 'w') as file:
        json.dump(data, file, indent=4)

     

if __name__ == "__main__":
    main()
