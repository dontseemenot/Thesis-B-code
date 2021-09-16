
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
X = []
y = []
with pd.HDFStore('E:/HDD documents/University/Thesis/Thesis B code/data/Berlin_no_overlap.h5') as store:
    df = store['IM_01 I']
    # Exclude 'Other' stages
    df = df.loc[(df['Sleep_Stage'] == 'W') | (df['Sleep_Stage'] == 'S1') | (df['Sleep_Stage'] == 'S2') | (df['Sleep_Stage'] == 'S3') | (df['Sleep_Stage'] == 'S4') | (df['Sleep_Stage'] == 'R') ]
    [X.append(row) for row in df.iloc[:, 1:None].to_numpy()]


# %%
epoch = 1
i = 10
j = i + 10
a = X[epoch][128*i:128*j]
plt.rcParams["figure.figsize"] = (20,5)

plt.plot(a)
print(f'Max: {np.max(a)*1e6} uV Min: {np.min(a)*1e6}')
# %%
