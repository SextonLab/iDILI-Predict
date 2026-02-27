import os
import re

import numpy as np
import pandas as pd

from glob import glob

import pickle
import click

from tqdm import tqdm

from preprocess import column_rename

@click.command()
@click.argument('source')
@click.argument('dest')
@click.argument('meta_path')
def concat_files(source, dest, meta_path):
    with open(meta_path, 'rb') as f:
        target_meta = pickle.load(f)
    cell = glob(os.path.join(source,'*','MyExpt_Cell.csv'))
    nuc = glob(os.path.join(source, "*", "MyExpt_Nucleus.csv"))
    cyto = glob(os.path.join(source, '*', "MyExpt_Cytoplasm.csv"))

    print("Cell...")
    cell_df = []
    for f in tqdm(cell):
        cell_df.append(pd.read_csv(f))
    cell_df = pd.concat(cell_df)
    
    cell_df, cell_cols = column_rename(cell_df, 'Cell')
    cell_df[target_meta+cell_cols].to_csv(os.path.join(dest, 'Cell_df.csv'), index=False)

    print("Nucleus...")
    nuc_df = []
    for f in tqdm(nuc):
        nuc_df.append(pd.read_csv(f))
    nuc_df = pd.concat(nuc_df)

    nuc_df, nuc_cols = column_rename(nuc_df, 'Nucleus')
    nuc_df[target_meta+nuc_cols].to_csv(os.path.join(dest, 'Nucleus_df.csv'), index=False)

    
    print("Cytoplasm...")
    cyto_df = []
    for f in tqdm(cyto):
        cyto_df.append(pd.read_csv(f))
    cyto_df = pd.concat(cyto_df)

    cyto_df, cyto_cols = column_rename(cyto_df, 'Cytoplasm')
    cyto_df[target_meta+cyto_cols].to_csv(os.path.join(dest, 'Cytoplasm_df.csv'), index=False)

    print('Merging Dataframes...')
    out = pd.merge(cell_df, nuc_df, on=target_meta, how='left').merge(cyto_df, on=target_meta, how='left')
    out.to_csv(os.path.join(dest, 'joined_df.csv'), index=False)
    out_cols = cell_cols+nuc_cols+cyto_cols

    with open(os.path.join(dest, 'data_cols'), 'wb') as f:
        pickle.dump(out_cols, f)
    
if __name__ =='__main__':
    concat_files()   