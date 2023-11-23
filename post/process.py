import json
import numpy as np
import os
import pandas as pd

folderpath = r"C:\Users\mgime\Documents\FINAL RDDL RESULTS"

    
def collect_json_to_csv():
    
    # combine JSON data
    lines = []
    for (path, _, files) in os.walk(folderpath):
        for file in files:
            if file.endswith('.json'):
                name = os.path.splitext(file)[0]
                domain, *_, instance, method, online, time = name.split('_')
                instance = instance.rjust(2, '0')
                task = f'{domain}_{instance}'
                with open(os.path.join(path, file)) as contents:
                    stats = json.load(contents)
                    mean = stats['mean']
                    std = stats['std']
                    lines.append([task, method, mean, std])
    df = pd.DataFrame(lines, columns=['task', 'method', 'mean', 'std'])
    df = df.sort_values(by=['task', 'method'])
    df = df.pivot(index='task', columns='method')
    df.to_csv('final_raw.csv')
    
    # normalize performance
    low = np.maximum(df['mean', 'noop'], df['mean', 'random'])
    high = df['mean'].max(axis=1, skipna=True)
    normalizer = np.maximum(high - low, 1e-12)
    for col in df['mean'].columns:
        df['mean', col] = np.clip((df['mean', col] - low) / normalizer, 0., 1.)
        df['std', col] = df['std', col] / normalizer
    df.to_csv('final_normalized.csv')
    
if __name__ == '__main__':
    collect_json_to_csv()
    
