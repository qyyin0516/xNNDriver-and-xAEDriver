import pandas as pd
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def one_hot_coding(train_y):
    """
    Converts a single-column DataFrame of labels into a one-hot encoded NumPy array.
    """
    Y_train_df = pd.get_dummies(train_y.iloc[:, 0])
    Y_train = np.array(Y_train_df).T
    return Y_train


def softmax(x):
    """
    Computes the softmax activation function for a given input array.
    This is used to convert raw output scores (logits) into probabilities.
    """
    x = np.asarray(x)
    x_col_max = x.max(axis=0)
    x_col_max = x_col_max.reshape([1,x.shape[1]])
    x = x - x_col_max
    x_exp = np.exp(x)
    x_exp_col_sum = x_exp.sum(axis=0).reshape([1,x.shape[1]])
    softmax = x_exp / x_exp_col_sum
    return softmax


def get_predictions(output_train, output_test, output_layer):
    """
    Calculates class predictions from the softmax output of a specific layer.
    """
    output_train = softmax(output_train[output_layer])
    output_test = softmax(output_test[output_layer])
    y_train_pred = np.zeros(output_train.shape[1])
    for i in range(output_train.shape[1]):
        y_train_pred[i] = output_train[:, i].argmax()
    y_test_pred = np.zeros(output_test.shape[1])
    for i in range(output_test.shape[1]):
        y_test_pred[i] = output_test[:, i].argmax()
    return y_train_pred, y_test_pred
    
    
def get_driver_potential_score(y_true, y_pred):
    """
    Calculates the average Area Under the ROC Curve (AUC) across all classes.
    This serves as the driver potential score for a gene model.
    """
    classes = list(pd.DataFrame(y_true).iloc[:, 0].unique())
    y1 = label_binarize(y_true, classes=classes)
    y2 = label_binarize(y_pred, classes=classes)
    fpr = {}
    tpr = {}
    roc_auc = []
    if len(classes) == 2:
        n_classes = 1
    else:
        n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y1[:, i], y2[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))
    return sum(roc_auc) / len(roc_auc)


def get_pathway_importance(y_train, activation_output, thr=0.1):
    """
    Identifies and ranks important pathways based on the difference in their 
    average activation values between two classes (e.g., class 0 and 1).
    """
    pathway_importance = {}
    for output_layer in range(2, len(activation_output) + 2):
        pathway_importance_layer = {}
        for j in range(1, output_layer):
            activation_value = activation_output[output_layer][j].T
            activation_average = pd.DataFrame(data=0, index=[0, 1], columns=activation_value.columns)
            activation_0 = activation_value[y_train.iloc[:, 0] == 0]
            activation_1 = activation_value[y_train.iloc[:, 0] == 1]
            activation_average.iloc[0, :] = activation_0.mean()
            activation_average.iloc[1, :] = activation_1.mean()
            pathway_importance_layer[j] = activation_average
        pathway_importance[output_layer] = pathway_importance_layer
    pathways = {}
    for output_layer in range(2, len(activation_output) + 2):
        pathway_01_value = pd.DataFrame(data=None, columns=['pathway', 'value'])
        count = 0
        for j in range(1, output_layer):
            pathway_value = pathway_importance[output_layer][j]
            for i in range(pathway_value.shape[1]):
                pathway_ordered = pathway_value.iloc[:, i].sort_values(ascending=False)
                if abs(pathway_ordered[0] - pathway_ordered[1]) >= thr:
                    pathway_01_value.loc[count] = [pathway_value.columns[i], abs(pathway_ordered[0] - pathway_ordered[1])]
                    count = count + 1
        pathways[output_layer] = pathway_01_value
        pathways[output_layer] = pathways[output_layer].sort_values(by='value', ascending=False)
        for j in range(pathways[output_layer].shape[0]):
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy1') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy2') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy3') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy4') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy5') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy6') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy7') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy8') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
            pathways[output_layer].iloc[j, 0] = re.sub(re.escape('_copy9') + '$', '',
                                                       pathways[output_layer].iloc[j, 0])
        pathways[output_layer] = pathways[output_layer].drop_duplicates(subset='pathway', keep='first')
    if len(activation_output) == 1:
        pathways_final = pathways[2]
    else:
        pathways_final = pathways[2]
        for output_layer in range(3, len(activation_output) + 2):
            pathways_final = pd.concat([pathways_final, pathways[output_layer]], axis=0)
    pathways_final = pathways_final.sort_values(by='value', ascending=False)
    pathways_final = pathways_final.drop_duplicates(subset='pathway', keep='first')
    return pathways_final