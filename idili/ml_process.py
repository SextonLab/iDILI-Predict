import os
import re

import numpy as np
import pandas as pd

from glob import glob
import pickle

from preprocess import column_rename, merge_files, agg_norm, dmso_norm, long_to_wide, compound_fix, add_label 
import click
"""
@click.command()
@click.argument('indir')
@click.argument('outdir')
@click.argument('cellline')
def process_data(indir, outdir, cellline):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print("Reading files...")
    cell_file =  glob(os.path.join(indir, "*cell*.csv"))[0]
    print(cell_file)
    # assert len(cell_file) > 0, 'No cell file found'
    # cell = pd.read_csv(cell_file)

    nuc_file = glob(os.path.join(indir, "*nucleus*.csv"))[0]
    assert len(nuc_file) > 0, 'No nucleus file found'
    # nuc = pd.read_csv(nuc_file)

    cyto_file = glob(os.path.join(indir, '*cytoplasm*.csv'))[0]
    assert len(cyto_file) >  0, 'No cytoplasm file found'
    # cyto = pd.read_csv(cyto_file)
    
    print("Merging data...")
    df, meta_cols, data_cols = merge_files(cell=cell_file, nuc=nuc_file, cyto=cyto_file)
    df.to_csv(os.path.join(outdir, f"{cellline}_joined.csv"), index=False)

    print('Aggregation and Normalization...')

    if ('Metadata_PlateID' in meta_cols) and  ('Metadata_WellID' in meta_cols):
        df_norm = agg_norm(df, data_cols=data_cols, ) # meta_cols=meta_cols
    else:
        # todo find the plateID and wellID columns
        print("I'm working on it")
        df_norm = agg_norm(df, data_cols=data_cols, meta_cols=meta_cols)
    print("Deleting DF")
    del df
    df_norm.to_csv(os.path.join(outdir, f"{cellline}_norm.csv"), index=False)
    
    print("DMSO step")
    df_dmso = dmso_norm(df_norm, meta_cols=meta_cols, data_cols=data_cols)
    df_dmso.to_csv(os.path.join(outdir, f"{cellline}_dmso_median.csv"), index=False)

    print("Wide-to-Long Transformation...")

    wide, wide_data, wide_meta = long_to_wide(df_dmso, data_cols, meta_cols)

    wide.to_csv(os.path.join(outdir, f"{cellline}_wide.csv"), index=False)
 """   
@click.command()
@click.argument('source')
@click.argument('dest')
@click.argument('data_cols')
@click.argument('meta_cols')
@click.argument('cell_line')
def process_data(source, dest, data_cols:str, meta_cols:str, cell_line:str):
    print("Reading data: ", source)
    df = pd.read_csv(source) 
    
    with open(data_cols, 'rb') as f:
        data_cols = pickle.load(f)
    with open(meta_cols, 'rb') as f:
        meta_cols = pickle.load(f)
    if 'ObjectNumber' in meta_cols:
        meta_cols.remove('ObjectNumber')
    
    print("Plate Aggregation and Normalization with pycytominer...")
    df_norm = agg_norm(df, data_cols=data_cols, meta_cols=meta_cols, strata=['Metadata_Plate', 'Metadata_WellID'])
    df_norm.to_csv(os.path.join(dest, f'{cell_line}_agg_norm.csv'), index=False)
    
    print('DMSO Controls...')
    df_dmso = dmso_norm(df_norm, meta_cols=meta_cols, data_cols=data_cols, compound_col='Metadata_CMPD')
    df_dmso.to_csv(os.path.join(dest, f"{cell_line}_DMSO.csv"), index=False)

    print('Long to Wide Transform...')
    df_wide, wide_data_cols, wide_meta_cols = long_to_wide(df_dmso, data_cols=data_cols, meta_cols=meta_cols, compound_col='Metadata_CMPD', conc_col='Metadata_CONC')
    df_wide.to_csv(os.path.join(dest, f"{cell_line}_wide.csv"), index=False)

    with open(os.path.join(dest, 'wide_data'), 'wb') as f:
        pickle.dump(wide_data_cols, f)
    
    with open(os.path.join(dest, 'wide_meta'), 'wb') as f:
        pickle.dump(wide_meta_cols, f)

     
if __name__ == '__main__':
    process_data()