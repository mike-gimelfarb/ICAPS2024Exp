import pandas as pd
import matplotlib.pyplot as plt

   
if __name__ == '__main__':
    df = pd.read_csv('final_raw.csv')
    df.columns = df.iloc[0]
    df.drop(df.index[0])
    df[['domain', 'instance']] = df['method'].str.split('_', n=1, expand=True)
    del df['method']
    df = df.iloc[2:, [-2, -1, 0, 1, 2, 3, 5, 8, 4, 15]]
    del df['noop']
    del df['random']
    nplots = len(df.groupby('domain'))
    fig, axs = plt.subplots(nplots, figsize=(10, 6 * nplots))
    idx = 0
    for domain, dfd in df.groupby('domain'):
        del dfd['domain']
        dfd = dfd.set_index('instance')
        dfd = dfd.astype(float)    
        dfd.plot.bar(ax=axs[idx])
        axs[idx].title.set_text(domain)
        idx += 1
    plt.tight_layout()
    plt.savefig('final_plots.pdf')