import os
import re

import numpy as np
import pandas as pd

from pycytominer import aggregate, normalize
from scipy.stats import wasserstein_distance

#TODO: add number of replicates to DMOS norm

def column_rename(df:pd.DataFrame, typeline:str):
    meta_cols = df.columns[df.columns.str.contains(pat='ObjectNumber|Metadata|Location|ImageNumber|Source|Parent|Number_Object|Count', flags=re.IGNORECASE)].tolist() 
    data_cols = df.drop(columns=meta_cols).columns.tolist()

    col_dict = {}
    new_data_cols = []
    for x in data_cols:
        col_dict[x] = f"{typeline}_{x}"
        new_data_cols.append(f"{typeline}_{x}")
    dt = df.rename(columns=col_dict)
    return dt, new_data_cols

def merge_files(cell, nuc,cyto, merge_cols=['ImageNumber', 'ObjectNumber'], target_meta=None):
    print("reading files...")
    cell_file = cell
    cell = pd.read_csv(cell)
    nuc = pd.read_csv(nuc)
    cyto = pd.read_csv(cyto)
    
    if target_meta is None:
        target_meta = cell.columns[cell.columns.str.contains(pat='ImageNumber|ObjectNumber|Compound|Concentration|Control|PlateID|WellID', flags=re.IGNORECASE)].tolist()
        
    
    meta_df = cell[target_meta].copy()
    cell, cell_data = column_rename(cell, typeline='Cell')
    cell = cell[merge_cols+cell_data]
    
    nuc, nuc_data = column_rename(nuc, typeline='Nucleus')
    nuc = nuc[merge_cols+nuc_data]

    cyto, cyto_data = column_rename(cyto, typeline='Cytoplasm')
    cyto = cyto[merge_cols+cyto_data]

    print('Final Merge...')

    final = pd.merge(cell, nuc, on=merge_cols, how='left')
    del cell
    del nuc
    print('first merge')
    final = final.merge(cyto, on=merge_cols, how='left')
    del cyto
    final = final.merge(meta_df, on=merge_cols, how='left')
    final = final[target_meta+cell_data+nuc_data+cyto_data]
    return final, target_meta, cell_data+nuc_data+cyto_data

def agg_norm(df:pd.DataFrame, data_cols,  meta_cols, strata=['Metadata_PlateID', 'Metadata_WellID']):
    if 'ObjectNumber' in meta_cols:
        meta_cols.remove('ObjectNumber')

    df_agg = aggregate(
        population_df=df,
        strata=strata,
        features=data_cols,
        operation='median',
        output_file=None
    )

    df_meta = df[meta_cols].copy()
    # df_meta.drop(columns='ObjectNumber')
    df_meta.drop_duplicates(inplace=True)

    dt = pd.merge(df_agg, df_meta, on=strata, how='left')

    df_norm = normalize(
        profiles=dt,
        features=data_cols,
        meta_features=meta_cols,
        method='mad_robustize',
        output_file=None
    )
    return df_norm

    
def dmso_norm(df:pd.DataFrame, meta_cols, data_cols, compound_col='Metadata_Compound'):
    """
    Takes 3 highest median DMSO rows from multiple plates in a single dataframe df
    Args:
        df (pd.DataFrame): Aggregated and Normalized cell, nucleus, and cytoplasm data
        meta_cols (list[str]): list of metadata columns
        data_cols (list[str]): list of feature columns
        compound_col (str, optional): Name of compound columns. Defaults to 'Metadata_Compound'.

    Returns:
        pd.DataFrame : dataframe with original data and top 3 DMSO rows 
    """
    dmso = df.loc[df[compound_col]=='DMSO'].copy()

    frame = dmso[data_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(0)
    
    n = len(frame)
    dist = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            d = wasserstein_distance(frame.iloc[i], frame.iloc[j])
            dist[i,j] = d
            dist[j,i] = d
    
    scores = np.median(dist, axis=1)
    order = np.argsort(scores)
    idx = order[:3]

    out = dmso.iloc[idx].copy()
    
    out.insert(0, 'row_index', dmso.index[idx])
    out.insert(0, 'rank', range(1,len(idx)+1))
    out.insert(2, 'median_distance', scores[idx])

    out = out[meta_cols+data_cols]

    dt = df.loc[df[compound_col]!='DMSO']
    dt = pd.concat([dt, out])
    return dt

def long_to_wide(df:pd.DataFrame, data_cols, meta_cols, compound_col='Metadata_Compound', conc_col='Metadata_Concentration', num_conc=5, num_reps=3):
    """
    Applies Long-to-Wide transformation to re-embed concentration level onto measurement columns

    Args:
        df (pd.DataFrame): measurement dataframe
        data_cols (_type_): measurement column list
        meta_cols (_type_): metadata column list
        compound_col (str, optional): Compound name column. Defaults to 'Metadata_Compound'.

    Returns:
        tuple : wide, wide_data_cols, meta_sub_cols - long-to-wide dataframe, new data_cols, new meta cols (includes replicant)
    """
    dt = df.loc[df[compound_col]!='DMSO'].copy()
    dmso = df.loc[df[compound_col]=='DMSO'].copy()
    
    dfs = []
    for c in dt[compound_col].unique():
        print(c)
        temp = dt.loc[dt[compound_col] == c].copy()
        concs = temp[conc_col].unique().tolist()
        concs.sort()
        concs_steps = {concs[i]:i+1 for i in range(num_conc)}
        temp['Metadata_Conc_Step'] = temp[conc_col].apply(lambda x: concs_steps[x])
        dfs.append(temp)
    
    df = pd.concat(dfs)
    
    concs_steps = [i+1 for i in range(num_conc)]

    duplicates_dmso = []
    for conc in concs_steps:
        temp = dmso.copy()
        temp['Metadata_Conc_Step'] = conc
        duplicates_dmso.append(temp)
    dmos_df = pd.concat(duplicates_dmso, ignore_index=True)
    
    combined_df = pd.concat([df, dmos_df])
    non_feature_cols = combined_df.drop(columns=data_cols).columns.tolist()
    
    # Adding replicant column for later
    combined_df = combined_df.sort_values([compound_col, 'Metadata_Conc_Step'])
    combined_df['Metadata_Replicant'] = (combined_df.groupby(compound_col).cumcount() % num_reps + 1)

    # pivot table
    pivot = (
        combined_df
        .set_index([compound_col, 'Metadata_Replicant', 'Metadata_Conc_Step'])[data_cols]
        .unstack('Metadata_Conc_Step')
    )

    # flatten the table
    data = {}
    for c in data_cols:
        for conc in concs_steps:
            col = (c, conc)
            new_col = f"{c}_dose_{conc}"
            if col in pivot.columns:
                data[new_col] = pivot[col].astype('float64')
            else:
                data[new_col] = pd.Series([float['nan']] * len(pivot), index=pivot.index, dtype='float64')
    
    flat = pd.DataFrame(data=data, index=pivot.index)

    # Show wide to long
    # print(flat.shape, combined_df.shape)

    flat = flat.reset_index()
    # flat = flat.rename(columns={compound_col:combined_df, 'Metadata_Replicant':'Metadata_Replicant'})

    meta_sub_cols = combined_df.drop(columns=data_cols).columns.tolist()
    
    meta_df = (
        combined_df[meta_sub_cols]
        .drop_duplicates(subset=[compound_col, 'Metadata_Replicant'])
        .set_index([compound_col, 'Metadata_Replicant'])
    )
    wide = (
        flat.set_index([compound_col, 'Metadata_Replicant'])
        .join(meta_df, how='left')
        .reset_index()
    )

    wide_data_cols = wide.drop(columns=meta_sub_cols).columns.tolist()
    
    return wide, wide_data_cols, meta_sub_cols

def compound_fix(df:pd.DataFrame, compound_col:str, compound_name:str, contorl_col:str, control_value:str):
    """
    Fixes single compound labeled incorrectly

    Args:
        df (pd.DataFrame): Measurement Dataframe
        compound_col (str): Name of Compound Column
        compound_name (str): Name of Compound
        contorl_col (str): Name of Contorl Column
        control_value (str): Correct contorl label

    Returns:
        pd.DataFrame : dataframe with updated compound contorl status
    """
    df.loc[df[compound_col]==compound_name, contorl_col] = control_value
    return df

def add_label(df:pd.DataFrame, contorl_col:str='Metadata_Control', pos_ctrl='PC'):
    """
    Adds binary class label for ML training

    Args:
        df (pd.DataFrame): processed measurement data
        contorl_col (str): control column
        pos_ctrl (str, optional): Positive class identifier. Defaults to 'PC'.

    Returns:
        pd.DataFrame : dataframe with class label
    """
    df['Metadata_Label'] = 0
    df.loc[df[contorl_col]==pos_ctrl, 'Metadata_Label'] = 1
    return df