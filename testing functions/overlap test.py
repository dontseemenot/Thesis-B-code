# %%
import numpy as np
import matplotlib.pyplot as plt

def overlap(epochData, dataPointsInEpoch):
    extraWindows = 5    # 25s overlapping windows between 30s epochs
    offset = int(dataPointsInEpoch / (extraWindows + 1))
    epochData2 = []
    i = 0
    j = 0
    curStage = epochData[0][0]
    epochData2.append(epochData[0])
    for i in range(1, len(epochData)):
    #for i in range(1, 4):  # debugging
        if epochData[i][0] == curStage:
            # Do overlap if two consecutive epochs are the same stage
            epochCombined = np.array([*epochData[i - 1][1], *epochData[i][1]])
            assert(len(epochCombined) == dataPointsInEpoch*2)
            # Do overlap
            for j in range(1, extraWindows + 1):    # j = 1 to 5
                epochData2.append([epochData[i][0], epochCombined[j*offset: j*offset + dataPointsInEpoch]])
            curStage = epochData[i][0]
            assert(np.all(
                epochCombined[(2*offset):(2*offset + dataPointsInEpoch)] ==
                [*epochData2[-4][1]]))  # Sanity check
        epochData2.append(epochData[i])
        assert(np.all(epochData[i][1] == epochData2[-1][1]))    # Sanity check
        curStage = epochData[i][0]
            
        
    #print(f'len1 {len(epochData)}, len2 {len(epochData2)}')
    return epochData2

size = 6
# W = ['W', np.full(size, 1)]
# S1 = ['S1', np.full(size, 0.9)]
# S2 = ['S2', np.full(size, 0.8)]
# S3 = ['S3', np.full(size, 0.7)]
# S4 = ['S4', np.full(size, 0.6)]
# R = ['R', np.full(size, 0.5)]

a1 = [
    ['W', np.array([1, 2, 3, 4, 5, 6])],
    ['W', np.array([3, 4, 5, 6, 7, 8])],
    ['W', np.array([-3, -4, -5, -6, -7, -8])],
    ['S1', np.array([10, 20, 30, 40, 50, 60])],
    ['S1', np.array([20, 30, 40, 50, 60, 70])],
    ['W', np.array([-2, -1, 0, 1, 2, 3])],
    ['W', np.array([32, 30, 28, 26, 24, 22])]
]




for i, a in enumerate(a1):
    print(i, a)
print("\n")
a2 = overlap(a1, size)
for i, a in enumerate(a2):
    print(i, a)

a2 = np.array(a2)
# %%
plt.plot(a2[:,1])
# %%
n10 = 351 max epochs