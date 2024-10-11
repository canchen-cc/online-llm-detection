import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results_to_plot file

with open('exp_main/resluts/results_to_plot/flash.gemma.scenario1.json', 'r') as file:
    items = json.load(file)

df_list = []
for item in items:
    df = pd.DataFrame({
        'rejection_time': item['rejection_time'],
        'power': item['power'],
        'fpr': item['fpr'],
        'name': item['item_name'],  
        'alpha': np.linspace(0.005, 0.1, len(item['fpr']))  
    })
    df_list.append(df)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))  
i=0
markers = ["*", "s", "^", "v", "_", "<", "P", "o", "X", "|"]
for df in df_list:
    # Plot 1: Rejection Time vs Power
    ax[0].plot(df['rejection_time'], df['fpr'],ls='--', lw=3,  marker=markers[i], label=df['name'].iloc[0], markersize=10)
    #ax[0].plot(df['rejection_time'], df['fpr'],ls='', lw=3,  marker=markers[i], label=df['name'].iloc[0], markersize=10)
    # Plot 2: Alpha vs FPR
    ax[1].plot(df['alpha'], df['fpr'], ls='--', lw=3, marker=markers[i], label=df['name'].iloc[0], markersize=10)
    i += 1

ax[0].tick_params(axis='both', labelsize=20, which='major', length=10,  width=2)


#ax[0].set_ylim(-0.005,0.055)
#ax[0].set_yticks(np.arange(0,0.055,0.01))
ax[0].set_ylim(-0.05,0.85)
ax[0].set_yticks(np.arange(0,0.85,0.2))


ax[0].set_xlim(-20,220)
ax[0].set_xticks(np.arange(0,220, 40))
ax[0].set_xlabel(r'Rejection Time ($\tau$)', fontsize=20)

#ax[0].set_ylabel('Power', fontsize=20)
ax[0].set_ylabel('False Positive Rate (FPR)', fontsize=20)
#ax[0].legend(loc='upper right', fontsize=14, framealpha=0.2)
#ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=4, fontsize=14)

x = np.linspace(0, 0.1, 500)
y = x 
ax[1].fill_between(x, 0, y, color='yellow', alpha=0.2, zorder=1)  

ax[1].set_ylim(-0.05,0.85)
ax[1].set_yticks(np.arange(0,0.85,0.2))

ax[1].set_xlim(-0.005,0.105)
ax[1].set_xticks(np.arange(0,0.12,0.02))
ax[1].tick_params(axis='both', labelsize=20, which='major', length=10,  width=2)
ax[1].plot([0, 0.1], [0, 0.1], color='k', ls='--',  lw=3)
ax[1].set_xlabel(r'Significance Level ($\alpha$)', fontsize=20)
ax[1].set_ylabel('False Positive Rate (FPR)', fontsize=20)




handles, labels = next(ax.flat).get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.49, -0.07), fancybox=True, shadow=True,
             ncol=5, fontsize=20, labelspacing=0.1, handletextpad=0.5, handlelength=1)

plt.subplots_adjust(wspace=0.4)  



for axis in ax: 
    for spine in axis.spines.values():
        spine.set_linewidth(2)  

plt.savefig('/.../...', dpi=300, bbox_inches='tight') 
plt.show()
