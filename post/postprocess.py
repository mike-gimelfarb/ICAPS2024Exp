import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

SMALL_SIZE = 24
MEDIUM_SIZE = 30
BIGGER_SIZE = 36

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

NEW_DOMAINS = {'HVAC', 'MarsRover', 'MountainCar', 'PowerGen', 'RaceCar', 'Reservoir', 'UAV', 'RecSim'}
OLD_BASELINES = ['gurobiplan', 'jaxplan', 'prost']
NEW_BASELINES = ['gurobiplan', 'jaxplan', 'disprod']
PRINT_NAMES = {
    'mean ': '',
    'gurobiplan': 'GurobiPlan',
    'jaxplan': 'JaxPlan',
    'disprod': 'DiSProD',
    'prost': 'PROST',
    'noop': 'NoOp',
    'random': 'Random'
}


def plot_individual(time_ps=3):
    data = pd.read_csv('final_raw.csv')
    data[['domain', 'instance']] = data['task'].str.split('_', n=1, expand=True)
    for (domain, df_domain) in data.groupby('domain', sort=True):
        print(f'plotting {domain}')
        baselines = NEW_BASELINES if domain in NEW_DOMAINS else OLD_BASELINES
        means = [f'mean {baseline}_{time_ps}' for baseline in baselines] + ['mean noop', 'mean random']
        stds = [f'std {baseline}_{time_ps}' for baseline in baselines] + ['std noop', 'std random']
        df = df_domain[['instance'] + means]
        new_columns = []
        for col in df.columns:
            value = col.split('_')[0]
            for rule in PRINT_NAMES.items():
                value = value.replace(*rule)
            new_columns.append(value)
        df.columns = new_columns
        df.set_index('instance', drop=True, inplace=True)
        ax = df.plot.bar(rot=0, figsize=(10, 4), legend=False)
        
        mins, maxs = [], []
        for mean, std in zip(means, stds):
            for inst in range(len(df.index)):
                mean_value = df_domain[mean].iloc[inst]
                std_value = df_domain[std].iloc[inst]
                mins.append(mean_value - 1.96 * std_value / np.sqrt(20))
                maxs.append(mean_value + 1.96 * std_value / np.sqrt(20))
        assert len(mins) == len(maxs) == len(ax.patches)
        for (minp, maxp, p) in zip(mins, maxs, ax.patches):
            x = p.get_x()
            w = p.get_width()
            plt.vlines(x + w / 2, minp, maxp, color='k')
            
        ax.set_title(domain)
        ax.set_xlabel(r'$\mathrm{Instance}$')
        ax.set_ylabel(r'$\mathrm{Return}$')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        if domain in ['SkillTeaching', 'HVAC']:
            plt.legend(ncol=3)
        plt.tight_layout()
        if domain in NEW_DOMAINS:
            plt.savefig(f'plots/ippc2011and2014/{domain}_{time_ps}.pdf')
        else:
            plt.savefig(f'plots/ippc2023/{domain}_{time_ps}.pdf')


def tables_normalized(times_ps=[1, 3, 5]):
    old_latex, new_latex = '', ''
    old_latex += '\\begin{table*}\n\\centering\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{' + '|l|' + \
        '|'.join('l' * len(times_ps) for _ in OLD_BASELINES) + '|}\n'
    new_latex += '\\begin{table*}\n\\centering\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{' + '|l|' + \
        '|'.join('l' * len(times_ps) for _ in NEW_BASELINES) + '|}\n'
    old_latex += '\\hline\n\\multirow{2}{*}{Domain} & ' + \
        ' & '.join('\\multicolumn{' + str(len(OLD_BASELINES)) + '}{c}{' + PRINT_NAMES[base] + '}' 
                   for base in OLD_BASELINES) + '\\\\\n'
    new_latex += '\\hline\n\\multirow{2}{*}{Domain} & ' + \
        ' & '.join('\\multicolumn{' + str(len(NEW_BASELINES)) + '}{c}{' + PRINT_NAMES[base] + '}' 
                   for base in NEW_BASELINES) + '\\\\\n'
    old_latex += ' & ' + ' & '.join(map(str, times_ps * len(OLD_BASELINES))) + '\\\\\n'
    new_latex += ' & ' + ' & '.join(map(str, times_ps * len(NEW_BASELINES))) + '\\\\\n'
    old_latex += '\\hline\n'
    new_latex += '\\hline\n'
    
    data = pd.read_csv('final_normalized.csv')
    data[['domain', 'instance']] = data['task'].str.split('_', n=1, expand=True)
    for (domain, df_domain) in data.groupby('domain', sort=True):
        baselines = NEW_BASELINES if domain in NEW_DOMAINS else OLD_BASELINES
        means, ses = [], []
        for baseline in baselines:
            for time_ps in times_ps:
                if f'mean {baseline}_{time_ps}' in df_domain:
                    mean = df_domain[f'mean {baseline}_{time_ps}'].mean()
                    se = 1.96 * df_domain[f'mean {baseline}_{time_ps}'].std() / np.sqrt(len(df_domain.index)) 
                else:
                    mean = -np.inf
                    se = -np.inf
                means.append(mean)
                ses.append(se)                    
        i_best = np.nanargmax(means)
        means = list(map('{:.2f}'.format, means))
        ses = list(map('{:.2f}'.format, ses))
        row = domain
        for i, (m, s) in enumerate(zip(means, ses)):
            if m == '-inf':
                row += ' & -'
            elif m >= means[i_best]:
                row += ' & $\\mathbf{' + str(m) + ' \\pm ' + str(s) + '}$'
            else:
                row += ' & $' + str(m) + ' \\pm ' + str(s) + '$'
        if domain in NEW_DOMAINS:
            new_latex += row + '\\\\\n'
        else:
            old_latex += row + '\\\\\n'
 
    old_latex += '\\hline\n\\end{tabular}%\n}\n\\end{table*}'
    new_latex += '\\hline\n\\end{tabular}%\n}\n\\end{table*}'
    
    print(old_latex)
    print()
    print(new_latex)
    

def win_rate_by_instance(time_ps):
    data = pd.read_csv('final_raw.csv')
    data[['domain', 'instance']] = data['task'].str.split('_', n=1, expand=True)
    
    win_rates_old = {base: [0] * 10 for base in OLD_BASELINES}
    win_rates_new = {base: [0] * 5 for base in NEW_BASELINES}
    
    for (domain, df_domain) in data.groupby('domain', sort=True):
        baselines = NEW_BASELINES if domain in NEW_DOMAINS else OLD_BASELINES
        for i in range(len(df_domain.index)):
            inst = int(df_domain['instance'].iloc[i]) - 1
            scores = np.asarray([df_domain[f'mean {base}_{time_ps}'].iloc[i] 
                                 for base in baselines])
            winners = np.flatnonzero(scores == np.max(scores))
            for i_winner in winners: 
                if domain in NEW_DOMAINS:
                    win_rates_new[baselines[i_winner]][inst] += 1
                else:
                    win_rates_old[baselines[i_winner]][inst] += 1
    
    df_old = pd.DataFrame({PRINT_NAMES[base]: v for base, v in win_rates_old.items()}, index=range(1, 10 + 1))
    df_new = pd.DataFrame({PRINT_NAMES[base]: v for base, v in win_rates_new.items()}, index=range(1, 5 + 1))
    df_old = df_old.div(df_old.sum(axis=1), axis=0)
    df_new = df_new.div(df_new.sum(axis=1), axis=0)
    
    for idx, df in enumerate([df_old, df_new]):
        ax = df.plot(kind='bar', stacked=True, figsize=(10, 5))
        ax.set_xlabel(r'$\mathrm{Instance}$')
        ax.set_ylabel(r'$\mathrm{Win \ Rate}$')
        ax.legend(ncol=3)
        plt.tight_layout()
        if idx == 0:
            plt.savefig(f'plots/ippc2011_win_rate_by_instance_{time_ps}_seconds.pdf')
        else:
            plt.savefig(f'plots/ippc2023_win_rate_by_instance_{time_ps}_seconds.pdf')


def win_rate_by_time(times_ps=[1, 3, 5]):
    data = pd.read_csv('final_raw.csv')
    data[['domain', 'instance']] = data['task'].str.split('_', n=1, expand=True)
    
    win_rates_old = {base: [0] * len(times_ps) for base in OLD_BASELINES}
    win_rates_new = {base: [0] * len(times_ps) for base in NEW_BASELINES}
    
    for (domain, df_domain) in data.groupby('domain', sort=True):
        baselines = NEW_BASELINES if domain in NEW_DOMAINS else OLD_BASELINES
        for i in range(len(df_domain.index)):
            for i_time, time_ps in enumerate(times_ps):
                scores = np.asarray([df_domain[f'mean {base}_{time_ps}'].iloc[i] 
                                    for base in baselines])
                winners = np.flatnonzero(scores == np.max(scores))
                for i_winner in winners: 
                    if domain in NEW_DOMAINS:
                        win_rates_new[baselines[i_winner]][i_time] += 1
                    else:
                        win_rates_old[baselines[i_winner]][i_time] += 1
    
    df_old = pd.DataFrame({PRINT_NAMES[base]: v for base, v in win_rates_old.items()}, index=times_ps)
    df_new = pd.DataFrame({PRINT_NAMES[base]: v for base, v in win_rates_new.items()}, index=times_ps)
    df_old = df_old.div(df_old.sum(axis=1), axis=0)
    df_new = df_new.div(df_new.sum(axis=1), axis=0)
    
    for idx, df in enumerate([df_old, df_new]):
        ax = df.plot(kind='bar', stacked=True, figsize=(10, 5))
        ax.set_xlabel(r'$\mathrm{Time \ per \ Decision \ (Seconds)}$')
        ax.set_ylabel(r'$\mathrm{Win \ Rate}$')
        ax.legend(ncol=3)
        plt.tight_layout()
        if idx == 0:
            plt.savefig(f'plots/ippc2011_win_rate_by_time.pdf')
        else:
            plt.savefig(f'plots/ippc2023_win_rate_by_time.pdf')

                    
if __name__ == '__main__':
    # tables_normalized()
    # win_rate_by_instance(1)
    # win_rate_by_instance(3)
    # win_rate_by_instance(5)
    # win_rate_by_time([1, 3, 5])
    plot_individual(1)
    plot_individual(3)
    plot_individual(5)
