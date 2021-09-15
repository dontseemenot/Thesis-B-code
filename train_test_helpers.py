# Helper functions for algorithm
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import openpyxl as pxl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

def max_num_epochs(stages):
    max_epochs = 800
    num_ALL = stages['W'] + stages['S1'] + stages['S2'] + stages['S3'] + stages['S4'] + stages['R']
    num_LSS = stages['S1'] + stages['S2']
    num_SWS = stages['S3'] + stages['S4']
    num_REM = stages['R']
    num_BSL = stages['W'] + stages['S1'] + stages['S2'] + stages['R']
    threshold = min(max_epochs, num_ALL, num_LSS, num_SWS, num_REM, num_BSL)
    # print(f"W {stages['W']} S1 {stages['S1']} S2 {stages['S2']} S3 {stages['S3']} S4 {stages['S4']} R {stages['R']}")
    # print(f'ALL {num_ALL}, LSS {num_LSS}, SWS {num_SWS}, REM {num_REM}, BSL {num_BSL}\nChosen threshold {threshold}')
    return threshold

def balance_dataset(df, threshold, subdataset):
    if subdataset == 'ALL':
        data = df.loc[(df['Sleep_Stage'] == 'W') | (df['Sleep_Stage'] == 'S1') | (df['Sleep_Stage'] == 'S2') | (df['Sleep_Stage'] == 'S3') | (df['Sleep_Stage'] == 'S4') | (df['Sleep_Stage'] == 'R') ]
    elif subdataset == 'LSS':
        data = df.loc[(df['Sleep_Stage'] == 'S1') |  (df['Sleep_Stage'] == 'S2')]
    elif subdataset == 'SWS':
        data = df.loc[(df['Sleep_Stage'] == 'S3') |  (df['Sleep_Stage'] == 'S4')]
    elif subdataset == 'REM':
        data = df.loc[(df['Sleep_Stage'] == 'R')]
    elif subdataset == 'BSL':
        data = df.loc[(df['Sleep_Stage'] == 'W') |  (df['Sleep_Stage'] == 'S1') | (df['Sleep_Stage'] == 'S2') | (df['Sleep_Stage'] == 'R')]
    #print(f'data len: {len(data)}')
    data_resampled = resample(data, replace = False, n_samples = threshold, random_state = 42)
    count = data_resampled['Sleep_Stage'].value_counts()
    # print(f'After balancing:\n{count}')
    assert(len(data_resampled) == threshold)
    return data_resampled


def custom_preprocess(X, y, groups):
    X = np.asarray(X)
    y = np.asarray(y)       
    groups = np.asarray(groups)

    scaler = MinMaxScaler(feature_range = (0, 1))
    X = scaler.fit_transform(X)
    
    reshaper = ReshapeToTensor()
    X = reshaper.transform(X)
    y = reshaper.transform(y)
    return X, y, groups

class ReshapeToTensor():
    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X = np.expand_dims(X, -1)
        #y = np.expand_dims(y, -1)
        return X

# A hack to plot confusion matrix with keras model
class MyModelPredict(object):
    def __init__(self, model):
        self._estimator_type = 'classifier'
        self.model = model
        
    def predict(self, X_test):
        m = self.model
        y_pred = m.predict_classes(X_test)
        return y_pred

def create_dirs(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    models_dir = f'{results_dir}/models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    performance_dir = f'{results_dir}/performance_metrics'
    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)
    images_dir = f'{results_dir}/images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    test_pred_dir = f'{results_dir}/test_pred_npy'
    if not os.path.exists(test_pred_dir):
        os.makedirs(test_pred_dir)
    fold_info_dir = f'{results_dir}/fold_info'
    if not os.path.exists(fold_info_dir):
        os.makedirs(fold_info_dir)
    return models_dir, performance_dir, images_dir, test_pred_dir, fold_info_dir

def get_train_test(X, y, groups, train_index, test_index):
    X_train = X[train_index].astype('float32')
    y_train = y[train_index]
    X_test = X[test_index].astype('float32')
    y_test = y[test_index]
    groups_train = groups[train_index]
    groups_test = groups[test_index]
    unique_train_groups = np.unique(groups_train)   # Groups only needed for GroupKFold
    unique_test_groups = np.unique(groups_test)
    unique, counts = np.unique(y_train, return_counts = True)
    train_info = dict(zip(unique, counts))
    unique, counts = np.unique(y_test, return_counts = True)
    test_info = dict(zip(unique, counts))
    info = {
        'Train groups': str(unique_train_groups),   # To fit list into df
        'Test groups': str(unique_test_groups),
        'X_train.shape': str(X_train.shape),
        'y_tarain.shape': str(y_train.shape),
        'Train class count': str(train_info),
        'Test class count': str(test_info)
    }
    return X_train, y_train, X_test, y_test, groups_train, groups_test, info
    

def plot_train_val_acc_loss(i, val_acc, val_loss, train_acc, train_loss, images_dir):
    # Accuracy
    plt.rcParams["figure.figsize"] = (10,5)
    plt.plot(train_acc, label = 'Training accuracy', color = 'darkorange')
    plt.plot(val_acc, label = 'Validation accuracy', color = 'darkgreen')
    plt.title(f'Fold {i} Validation and Training accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training iteration')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid()
    plt.legend(bbox_to_anchor = (1, 1))
    plt.savefig(f'{images_dir}/Fold {i} Training accuracy plot.png', dpi = 200)

    # Validation
    plt.plot(train_loss, label = 'Training loss', color = 'wheat')
    plt.plot(val_loss, label = 'Validation loss', color = 'lawngreen')
    plt.title(f'Fold {i} Validation and Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Training iteration')
    plt.grid()
    plt.legend(bbox_to_anchor = (1, 1))
    plt.savefig(f'{images_dir}/Fold {i} Training loss plot.png', dpi = 200)

def get_excel_writer(spreadsheet_file):
    return pd.ExcelWriter(spreadsheet_file, engine = 'xlsxwriter')

def save_test_info(test_info, spreadsheet_file):
    writer = pd.ExcelWriter(spreadsheet_file, engine='xlsxwriter')   
    workbook=writer.book
    worksheet=workbook.add_worksheet('Test info')
    writer.sheets['Test info'] = worksheet

    df_test_info = pd.DataFrame.from_dict(test_info, orient = 'index')
    df_test_info.to_excel(writer, sheet_name = 'Test info',startrow = 1 , startcol = 0, header = False)
    writer.save()

# def save_fold_info(i, hyp_results, performance_metrics, spreadsheet_file):
def save_fold_info(i, data_info, hyp_results, best_hyp, performance_metrics, spreadsheet_file):
    book = pxl.load_workbook(spreadsheet_file)
    writer = pd.ExcelWriter(spreadsheet_file, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets) 


    # Data info
    df_info = pd.DataFrame(data = data_info, index = [0])
    df_info.to_excel(writer, sheet_name = f'Outer Fold {i}',startrow = 0, startcol = 0,index = False)
    offset = len(df_info) + 2
    
    # All hyperparameters and corresponding losses
    df_fold_info = pd.DataFrame(data = hyp_results['parameters'])
    df_fold_info['Loss'] = hyp_results['loss']
    df_fold_info['Loss std'] = hyp_results['loss_std']
    df_fold_info = df_fold_info.sort_values(by = 'Loss', ascending = True)
    df_fold_info.to_excel(writer, sheet_name = f'Outer Fold {i}',startrow = offset, startcol = 0,index = False)
    offset += len(df_fold_info) + 2

    # Best hyperparameters
    df_best_hyp = pd.DataFrame(data = best_hyp, index = [0])
    df_best_hyp.to_excel(writer, sheet_name = f'Outer Fold {i}',startrow = offset, startcol = 0,index = False)
    offset += len(df_best_hyp) + 2

    # Test performance
    df_performance = pd.DataFrame(data = performance_metrics, index = [0])
    df_performance.to_excel(writer, sheet_name = f'Outer Fold {i}',startrow = offset, startcol = 0,index = False)

    writer.save()

def save_mean_results(performance_metrics_mean, spreadsheet_file):
    book = pxl.load_workbook(spreadsheet_file)
    writer = pd.ExcelWriter(spreadsheet_file, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets) 

    df_mean_results = pd.DataFrame(data = performance_metrics_mean, index = [0])
    df_mean_results.to_excel(writer, sheet_name = 'Test info', startrow = 20 , startcol = 0, index= False)
    writer.save()

def plot_cm(y_pred, y_test, i, images_dir):
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Control (0)', 'Insomnia (1)']).plot(cmap = 'Blues')
    plt.title(f'Fold {i} Confusion Matrix')
    plt.savefig(f'{images_dir}/Fold {i} CM.png', dpi = 100)
    plt.clf()

    cm_norm = confusion_matrix(y_test, y_pred, normalize = 'true')
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_norm, display_labels = ['Control (0)', 'Insomnia (1)']).plot(cmap = 'Blues')
    plt.title(f'Fold {i} Confusion Matrix Normalized')
    plt.savefig(f'{images_dir}/Fold {i} Confusion Matrix Normalized.png', dpi = 100)
    plt.clf()
    return cm

def calculate_performance_metrics(y_test, y_pred, cm):
    performance_metrics = {}
    performance_metrics['accuracy'] = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    performance_metrics['precision'] = tp/(tp + fp)
    performance_metrics['recall'] = tp/(tp + fn)
    performance_metrics['sensitivity'] = tp/(tp + fn)
    performance_metrics['specificity'] = tn/(tn + fp)
    performance_metrics['f1'] = 2 * performance_metrics['precision'] * performance_metrics['recall'] / (performance_metrics['precision'] + performance_metrics['recall'])
    return performance_metrics

def plot_fold_test(y_pred, y_test, i, images_dir):

    plt.plot(y_pred, label = 'pred', linestyle = 'None', markersize = 1.0, marker = '.')
    plt.plot(y_test, label = 'test')
    plt.title(f'Fold {i} Test vs predicted')
    plt.ylabel('Control (0), Insomnia (1)')
    plt.xlabel('Test epoch')
    plt.legend()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.savefig(f'{images_dir}/Fold {i} Test vs predicted.png', dpi = 200)
    plt.clf()