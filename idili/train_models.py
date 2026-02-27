import os
import re

import markdown as md
import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import pickle
from imblearn.over_sampling import SMOTE

import click

import matplotlib.pyplot as plt
import seaborn as sns

def fix_compound(df:pd.DataFrame, cmpd_col, target_cmpd, cond_col, cond_val):
    df.loc[df[cmpd_col]==target_cmpd, cond_col] = cond_val
    return df

def write_report(outdir:str, model_name:str, cell_line:str, accuracy, f1_pos, f1_neg, loss, roc_auc, ):
    print('hello')
    lines = [
        f'# AutoGluon ML Report for {cell_line}\n',
        f' - Leader: {model_name}\n',
        f' - Accuracy: {accuracy}\n',
        f' - f1-positive: {f1_pos}\n',
        f' - f1-negative: {f1_neg}\n',
        f' - Final Model Loss: {loss}\n', 
        f' - ROC AUC: {roc_auc}\n',
    ]
    report_dest = os.path.join(outdir, f'{cell_line}_{model_name}_AutoGluon_report.md')
    with open(report_dest, "ab+") as f:
        for line in lines:
            f.write(line.encode())
    


def train_model(df:pd.DataFrame, data_cols:str, meta_cols:str, model_dir:str, cell_line:str):
    with open(data_cols, 'rb') as f:
        data_cols = pickle.load(f)
    with open(meta_cols, 'rb') as f:
        meta_cols = pickle.load(f)
    
    df.fillna(value=0, inplace=True)

    df = fix_compound(df, 'Metadata_CMPD', 'Clavulanate', 'Metadata_COND', 'NC')
    df['Metadata_Label'] = 0
    df.loc[df['Metadata_COND']=='PC', 'Metadata_Label'] = 1
    
    test = df.loc[df['Metadata_Replicant']==1].copy()
    train = df.loc[df['Metadata_Replicant']!=1].copy()
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(train[data_cols].values, train['Metadata_Label'].values)
    
    train_df = pd.DataFrame(data=X_res, columns=data_cols)
    train_df['Metadata_Label'] = y_res
    
    predictor = TabularPredictor(
        label='Metadata_Label',
        eval_metric='roc_auc',
        problem_type='binary',
        path=model_dir
    ).fit(train_data=train_df[data_cols+['Metadata_Label']], presets='best_quality', time_limit=4000, verbosity=2)
    
    leaderboard_df = predictor.leaderboard(test, silent=True)

    preds = predictor.predict(test)

    accuracy = accuracy_score(y_true=test['Metadata_Label'].values, y_pred=preds)
    f1_pos = f1_score(y_true=test['Metadata_Label'].values, y_pred=preds, pos_label=1)
    f1_neg = f1_score(y_true=test['Metadata_Label'].values, y_pred=preds, pos_label=0)

    loss = log_loss(y_true=test['Metadata_Label'].values, y_pred=preds)
    roc_auc = roc_auc_score(y_true=test['Metadata_Label'].values, y_score=preds)
    
    cm = confusion_matrix(y_true=test['Metadata_Label'].values, y_pred=preds)

    model_name = leaderboard_df.iloc[0]['model']

    plt.clf()

    ax = sns.heatmap(data=cm, annot=True)
    ax.set(xlabel='Predict Label', ylabel='True Label', title=f"{cell_line}_SMOTE_Confusion_Matrix")
    fig = ax.get_figure()
    plt.savefig(os.path.join(model_dir, f"{cell_line}_{model_name}_confusion_matrix.png"), dpi=fig.dpi)
    
    write_report(model_dir, model_name=model_name, cell_line=cell_line,
                 accuracy=accuracy, f1_pos=f1_pos, f1_neg=f1_neg, loss=loss, roc_auc=roc_auc)
    
    feat_cols = [(c, 'feature') for c in data_cols]
    metas = [(c, 'metadata') for c in meta_cols]
    col_df = pd.DataFrame().from_records(feat_cols+metas, columns=['Column_Name', 'Column_Type'])
    col_df.to_csv(os.path.join(model_dir, 'column_types.csv'), index=False)
    df['Pred_Label'] = predictor.predict(df[data_cols], model=model_name)
    df[['Pred_Proba_Neg', 'Pred_Proba_Pos']] = predictor.predict_proba(df[data_cols], model=model_name)
    df.to_csv(os.path.join(model_dir, f"{cell_line}_preds.csv"), index=False)

    
if __name__ == '__main__':
    wide_path = 'patient_lines/64B/64B_wide.csv'
    data_col_path = 'patient_lines/data_cols'
    meta_col_path = 'patient_lines/meta_cols'
    model_dest = 'patient_lines/64B/models'
    cell_line = '64B'

    df = pd.read_csv(wide_path)
    train_model(df, data_col_path, meta_col_path, model_dest, cell_line)

    