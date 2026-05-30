import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import torch 
import torch.nn as nn
from pathlib import Path
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
test_batch_size = 500

def createConfusionMatrix(y_test, y_pred, plt_name, class_names=None, output_dir=".", file_ext="png"):
    if class_names is None:
        class_names = ['Buildings', 'Woods', 'Roads', 'Apples', 'ground', 'Vineyard']
    labels = list(range(len(class_names)))
    df_cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=class_names,
        columns=class_names,
    )
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=0.9)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, cmap="Blues",annot=True,annot_kws={"size": 16}, fmt='g')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f'Cross-HL_{plt_name}.{file_ext}', format=file_ext)
    plt.close()

def AvgAcc_andEachClassAcc(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    class_acc = np.full(len(list_raw_sum), np.nan, dtype=np.float64)
    valid_classes = list_raw_sum > 0
    class_acc[valid_classes] = truediv(
        list_diag[valid_classes], list_raw_sum[valid_classes]
    )
    average_acc = np.nanmean(class_acc) if np.any(valid_classes) else 0.0
    class_acc = np.nan_to_num(class_acc, nan=0.0)
    return class_acc, average_acc

def result_reports(
    xtest,
    xtest2,
    ytest,
    name,
    model,
    iternum,
    device,
    class_names=None,
    output_dir=".",
    save_confusion=True,
):
    y_pred = np.empty((len(ytest)), dtype=np.float32)
    number = len(ytest) // test_batch_size

    model.eval()
    with torch.no_grad():
        for i in range(number):
            temp = xtest[i * test_batch_size:(i + 1) * test_batch_size, :, :].to(device)
            temp1 = xtest2[i * test_batch_size:(i + 1) * test_batch_size, :, :].to(device)
            temp2 = model(temp, temp1)
            if isinstance(temp2, tuple):
                temp2 = temp2[0]
            temp3 = torch.max(temp2, 1)[1].squeeze()
            y_pred[i * test_batch_size:(i + 1) * test_batch_size] = temp3.detach().cpu().numpy()
            del temp, temp1, temp2, temp3

        start = number * test_batch_size
        if start < len(ytest):
            temp = xtest[start:len(ytest), :, :].to(device)
            temp1 = xtest2[start:len(ytest), :, :].to(device)
            temp2 = model(temp, temp1)
            if isinstance(temp2, tuple):
                temp2 = temp2[0]
            temp3 = torch.max(temp2, 1)[1].squeeze()
            y_pred[start:len(ytest)] = temp3.detach().cpu().numpy()
            del temp, temp1, temp2, temp3

    y_pred = torch.from_numpy(y_pred).long()

    overall_acc = accuracy_score(ytest, y_pred)
    confusion_mat = confusion_matrix(ytest, y_pred)
    class_acc, avg_acc = AvgAcc_andEachClassAcc(confusion_mat)
    kappa_score = cohen_kappa_score(ytest, y_pred)
    if save_confusion:
        createConfusionMatrix(
            ytest,
            y_pred,
            str(name)+'_test_'+str(iternum)+'',
            class_names=class_names,
            output_dir=output_dir,
        )

    return confusion_mat, overall_acc*100, class_acc*100, avg_acc*100, kappa_score*100
