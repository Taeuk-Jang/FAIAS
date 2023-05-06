import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import os
import math

# methods_orig = ['Baseline', 'Adversarial Debiasing', 'Calibrated Equalized Odds Postprocessing',
# 'Disparate Impact Remover', 'Learning Fair Representations', 'Optimized Preprocessing', 'Reweighing',
# 'Reject Option Classification', 'Prejudice Remover Regularizer', 'FAIAS']

def plot(X_orig, metric_idx, data, setting, method_idx = [0,1,2,3,6,7,9], save = True, show = False):
    methods_orig = ['ABL', 'Adv_Deb', 'CEOP', 'DIR', 'LFR', 'Optimized Preprocessing',
                    'Reweigh', 'Rej_Opt', 'LAFTR', 'FAIAS', 'ABL']

    metrics = ['Classification_Accuracy', 'Balanced_Classification_Accuracy','Abs_(1-Disparate_Impact)',
'Abs_Equal_Opportunity_Difference', 'Abs_Average_Odds_Difference','Abs_Theil_Index',
'Abs_Statistical_Parity_Difference', 'True_Positive_Rate','False_Positive_Rate_Difference']

    # Enter raw data
    methods = ['\n'.join(wrap(methods_orig[i], 12)) for i in method_idx]
    # X_orig = np.random.rand(10, 5)
    X = X_orig[method_idx, :]

    # Create lists for the plot
    x_pos = np.arange(len(method_idx))
    # y_pos = np.arange(0.0,1.0,0.1)
    ymax = math.ceil(np.max(X)*10) / 10.0
    y_pos = np.arange(0.0, ymax, 0.1)

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, np.mean(X, axis=1), yerr=np.std(X, axis=1), align='center', alpha=0.5, ecolor='black', capsize=8)
    ax.set_ylabel(metrics[metric_idx], fontsize=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11, rotation = 15)
    # ax.set_yticks(y_pos)
    ax.set_ylim(ymin=0.0, ymax=ymax)
    ax.tick_params(axis='y', which='major', labelsize=15)
    # ax.tick_params(axis='x', which='major', labelsize=16)
    ax.set_title(metrics[metric_idx] + ': ' + data, fontsize=14)
    ax.yaxis.grid(True)
    # ax.annotate('Fair',xy=(x_pos[-1]+.5,1.))
    # ax.annotate('Bias',xy=(x_pos[-1]+.5,-1.))

    if metric_idx in [2,3,4,5,6]:
        ax2 = ax.twinx()
        ax2.set_yticks(np.arange(2))
        ax2.set_yticklabels(['Fair', 'Bias'], fontsize=15, color='red')

    # Save the figure and show
    plt.tight_layout()
    if not os.path.exists('./plot_fig/'+setting):
        os.makedirs('./plot_fig/'+setting)
    if save:
        plt.savefig('./plot_fig/'+setting+'/'+metrics[metric_idx]+'.png')
    if show:
        plt.show()
    # fig.savefig('test.png')