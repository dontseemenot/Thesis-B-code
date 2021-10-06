'''
with open(f'{results_dir}timestamp.txt', 'a') as f:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
    f.write(f'\nEnd time: {dt_string}')
# %%
# JUPYTER OUTPUT
with open(f'{results_dir}ipython_output.txt', 'w') as f:
    f.write(cap.stdout)
# %%
#model = keras.models.load_model(os.path.join('./CAP results/balanced data epoch 80 rem models/', f))
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
for m, x in zip(models, range(1, 10)):
    m[0].model.save(f'./CAP results/balanced data epoch 80/{dt_string} fold_{x}.h5')
results = np.array(results)
np.save('./CAP results/overlap epoch 20/results', results)
# %%
import os
#for f in os.listdir('./CAP results/balanced data epoch 80/'):
    
model = keras.models.load_model(f'./CAP results/balanced data epoch 80/28-07-2021 18.45.41 fold_1.h5')
y_pred = model.predict(X_test)
# %%
    unique, counts = np.unique(y_test, return_counts=True)
    d = dict(zip(unique, counts))
    results.append((accuracy_score(y_test, y_pred), d))
    print(results)
    models.append(pipeline)
# %%

res = tf.math.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(models[8][0], X_test, y_test,
                             display_labels={"Control", "Insomnia"},
                             cmap=plt.cm.Blues,
                             normalize= True)
# %%
maxTrainPatients = 8
maxTestPatients = 1
numTrainInsomniaPatients = 0
numTrainGoodPatients = 0
numTestInsomniaPatients = 0
numTestGoodPatients = 0
numGoodTrainEpochs = 0
numGoodTestEpochs = 0
numInsomniaTrainEpochs = 0
numInsomniaTestEpochs = 0

X_train = []
X_test = []
Y_train = []
Y_test = []



# Code for CAP 
# insomniaIDs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# goodIDs = [11, 12, 13, 14, 15, 22, 24, 25, 26]
# IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 22, 24, 25, 26]
# Using inter-patient paradigm (split patients into training and testing set)
## %%
classDict = {'G': 0, 'I': 1}
df = pd.DataFrame(columns = ['All', 'SLEEP-S0', 'SLEEP-S1', 'SLEEP-S2', 'SLEEP-S3', 'SLEEP-S4', 'SLEEP-REM'], index = ['Control', 'Insomnia', 'Total'])
for col in df.columns:
    df[col].values[:] = 0

good_id_to_group = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 22:6, 24:7, 25:8, 26: 9}
data = []
groups = np.array([])
for i in goodIDs:
    f = pd.read_hdf(dataPath, key = str(i))
    for epoch in f:
        data.append([epoch[0], epoch[1], epoch[2], classDict[epoch[3]]])
        groups = np.append(groups, good_id_to_group[epoch[0]])
        df.at['Control', epoch[1]] += 1
        df.at['Total', epoch[1]] += 1
        df.at['Control', 'All'] += 1
        df.at['Total', 'All'] += 1
for i in insomniaIDs:
    f = pd.read_hdf(dataPath, key = str(i))
    for epoch in f:
        data.append([epoch[0], epoch[1], epoch[2], classDict[epoch[3]]])
        groups = np.append(groups, epoch[0])
        df.at['Insomnia', epoch[1]] += 1
        df.at['Total', epoch[1]] += 1
        df.at['Insomnia', 'All'] += 1
        df.at['Total', 'All'] += 1
        
data = np.asarray(data, dtype = object)

X = data[:, 2]
Y = data[:, 3]



lpgo = LeavePGroupsOut(n_groups=1)
lpgo.get_n_splits(X, Y, groups)

lpgo.get_n_splits(groups=groups)  # 'groups' is always required

print(lpgo)

for train_index, test_index in lpgo.split(X, Y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    #print(X_train, X_test, y_train, y_test)

X_train = X_train.tolist()
X_train = np.reshape(X_train, (len(X_train), numEpochDataPoints, 1)).astype('float32')
Y_train = np.asarray(Y_train).astype('uint8')
Y_train = np.reshape(Y_train, (len(Y_train), 1))

X_test = X_test.tolist()
X_test = np.reshape(X_test, (len(X_test), numEpochDataPoints, 1)).astype('float32')
Y_test = np.asarray(Y_test).astype('uint8')
Y_test = np.reshape(Y_test, (len(Y_test), 1))
# %%

train = True
for g in goodIDs:
    print(g)
    data = pd.read_hdf(dataPath, key = str(g))
    if train == True:
        numTrainGoodPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_train.append(epoch[2])
            Y_train.append(epoch[3])
            numGoodTrainEpochs += 1
        if numTrainGoodPatients == maxTrainPatients:
            train = False
    else:
        numTestGoodPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_test.append(epoch[2])
            Y_test.append(epoch[3])
            numGoodTestEpochs += 1
        if numTestGoodPatients == maxTestPatients:
            break

train = True
for i in insomniaIDs:
    print(i)
    data = pd.read_hdf(dataPath, key = str(i))
    if train == True:
        numTrainInsomniaPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_train.append(epoch[2])
            Y_train.append(epoch[3])
            numInsomniaTrainEpochs += 1
        if numTrainInsomniaPatients == maxTrainPatients:
            train = False
    else:
        numTestInsomniaPatients += 1
        for epoch in data:
            #if epoch[1] == 'Rem':
            X_test.append(epoch[2])
            Y_test.append(epoch[3])
            numInsomniaTestEpochs += 1
        if numTestInsomniaPatients == maxTestPatients:
            break


print("Epochs, Good train: {} | In train: {} | Good test: {} | In test: {}".format(numGoodTrainEpochs, numInsomniaTrainEpochs, numGoodTestEpochs, numInsomniaTestEpochs))

X_train = np.asarray(X_train).astype('float32')
X_train = np.reshape(X_train, (len(X_train), numEpochDataPoints, 1))
Y_train = [0 if x == 'G' else 1 for x in Y_train]
Y_train = np.asarray(Y_train).astype('uint8')
Y_train = np.reshape(Y_train, (len(Y_train), 1))

X_test = np.asarray(X_test).astype('float32')
X_test = np.reshape(X_test, (len(X_test), numEpochDataPoints, 1))
Y_test = [0 if x == 'G' else 1 for x in Y_test]
Y_test = np.asarray(Y_test).astype('uint8')
Y_test = np.reshape(Y_test, (len(Y_test), 1))

# %%
# Alexnet model instantializer

# %%
m = 31   # model number
  # Sparse weights make algorithm faster
# %%
history = model.fit(X_train, Y_train, batch_size = 256, epochs = 50, shuffle = True)
np.save('{} history.npy'.format(m),history.history)
# %%
model.evaluate(X_test, Y_test)
# %%
y_prob = model.predict(X_test) 
y_classes = y_prob.argmax(axis=-1)
# %%
# Confusion matrix
cm = confusion_matrix(y_true = Y_test ,y_pred=y_classes)
plot_confusion_matrix(cm = cm, classes = ['0 (control)', '1 (insomnia)'], title = "confusion matrix")
 # %%
plt.rcParams["figure.figsize"] = (10,5)
plt.plot(history.history['sparse_categorical_accuracy'], label = 'accuracy', color = 'g')
plt.plot(history.history['loss'], label = 'loss', color = 'r')
plt.title('Training accuracy/loss')
plt.ylabel('Accuracy/loss')
plt.xlabel('Training iteration')
plt.yticks(np.arange(0, 1, 0.05))
plt.grid()
plt.legend(bbox_to_anchor = (1, 1))
plt.savefig('model{}_accloss.png'.format(m), dpi = 200)
# %%
inter_output_model = keras.Model(model.input, model.get_layer(index = 23).output )
inter_output = inter_output_model.predict(X_test)
plt.rcParams["figure.figsize"] = (10,5)
plt.title('Predicted vs actual output for test data')
#plt.plot(inter_output[:, 0], label = 'Class 0 (Control)')
#plt.plot(inter_output[:, 1], label = 'Class 1 (Insomnia)')
plt.plot(y_classes, label = 'Predicted label')
plt.plot(Y_test, label = 'True label')
#plt.ylim([-2, 2])
plt.legend()
plt.xlabel('Test epoch #')
plt.ylabel('Softmax output')
plt.tight_layout()
plt.grid()

plt.savefig('model{}_output.png'.format(m), dpi = 200)
# %%
model.save("E:\\HDD documents\\University\\Thesis\\Thesis B code\\model{}.h5".format(m))
# %%
# loading stuff
n = -1
model = keras.models.load_model('E:\\HDD documents\\University\\Thesis\\Thesis B code\\model{}.h5'.format(n))
# %%
history=np.load('{} history.npy'.format(n),allow_pickle='TRUE').item()


# %%
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# %%
'''