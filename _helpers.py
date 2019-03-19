import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

__all__ = ['analyze_results', 'read_data', 'combine_df',
           'error_df', 'repair_dfs',
           'eval_metrics',
           'raw_density_from_cor',
           'raw_with_errs',
           'model_metadata', 'attrW_df']

def _stats(df, total_errs):
    fil_err = df.init_value != df.correct_val
    fil_repair = df.init_value != df.pred_val
    fil_correct = df.correct_val == df.pred_val

    fil_correct_repair = fil_repair & fil_correct

    print("# of cells (in this subset): ", df.shape[0])
    print("# of errors (in this subset): ", fil_err.sum())
    print()
    print("# correct preds: ", fil_correct.sum())
    print("# incorrect preds: ", (~fil_correct).sum())
    print()
    print("# of repairs: ", fil_repair.sum())
    print()
    print("# of repairs (on correct cells): ", (~fil_err & fil_repair).sum())
    print("# of repairs (on incorrect cells): ", (fil_err & fil_repair).sum())
    print()
    print("# of correct repairs: ", fil_correct_repair.sum())
    print()
    print('precision: ', fil_correct_repair.sum() / fil_repair.sum())
    print('overall recall (/ total errors): ', fil_correct_repair.sum() / total_errs)
    print('recall (/ errors in this subset): ', fil_correct_repair.sum() / fil_err.sum())
    
def analyze_results(fpath, df_ref, thresh, val_col='inf_val'):
    """
    Returns (df_all (df_ref joined predictions), df_res (probabilities for all values))
    """
    if fpath.endswith('.csv'):
        df_res = pd.read_csv(fpath).rename({val_col: 'pred_val'}, axis='columns')
    else:
        df_res = pd.read_pickle(fpath).rename({val_col: 'pred_val'}, axis='columns')    
        
    import pdb; pdb.set_trace()
    df_pred = df_res.sort_values('proba', ascending=False). \
        drop_duplicates(['_tid_', 'attribute']).sort_values(['_vid_'])
    
    plt.hist(df_pred.proba.values)
    plt.title('Distribution of probabilities')
    plt.show()
        
    df_all = df_ref.merge(df_pred[['_tid_', 'attribute', 'proba', 'pred_val']], on=['_tid_', 'attribute'])
    
    total_errs = (df_all.init_value != df_all.correct_val).sum()
    
    print('Total cells:', df_all.shape[0])
    print('Total errors: ', total_errs)
    print('Total DK cells: ', df_all.is_dk.sum())
    
    print()
    
    print('All predicted')
    print('-------------')
    _stats(df_all, total_errs)
    print()
    
    print('DK cells (repair precision/recall)')
    print('-------------')
    _stats(df_all[df_all.is_dk], total_errs)
    print()
    
    if thresh:
        print('Beyond thresh=%f' % thresh)
        print('-------------')
        _stats(df_all[df_all.proba >= thresh], total_errs)
        print()

        print('Beyond thresh=%f AND DK (we introduce all predicted ones back into weak labels)' % thresh)
        print('-------------')
        _stats(df_all[(df_all.proba >= thresh) & df_all.is_dk], total_errs)
        print()
    
    df_res.proba = df_res.proba.apply(lambda f: round(f, 4))
    return df_all, df_res

def eval_metrics(df_all):
    """
    df_all is the first dataframe from analyze_results (has init_value, correct_val, and pred_val columns).
    """
    fil_err = df_all.init_value != df_all.correct_val
    fil_cor = df_all.pred_val == df_all.correct_val
    fil_repair = df_all.init_value != df_all.pred_val
    
    return {'precision': (fil_repair & fil_cor).sum() / fil_repair.sum(),
            'recall': (fil_repair & fil_cor).sum() / fil_err.sum()
           }

def read_data(f_dom, f_dk, f_clean, f_raw, normalize=False):
    """
    Returns (df_dom, df_dk, df_clean, df_raw)
    """
    df_dom = pd.read_pickle(f_dom)
    df_dk = pd.read_pickle(f_dk)

    df_clean = pd.read_csv(f_clean, encoding='utf-8').rename({'tid': '_tid_'}, axis='columns')
    # df_clean.fillna('_nan_', inplace=True)
    # Do not include predictions for NULLs
    df_clean.dropna(inplace=True)

    """
    Raw data
    """
    df_raw = pd.read_csv(f_raw, dtype=str, encoding='utf-8').reset_index()\
        .rename({'index': '_tid_'}, axis='columns')
    for attr in df_raw.columns.values:
        if df_raw[attr].isnull().all():
            print('dropping "%s" column (all nulls)' % attr)
            df_raw.drop(attr, axis='columns', inplace=True)
            continue
    
    if normalize:
        df_clean['correct_val'] = df_clean['correct_val'].str.strip().str.lower()
        for attr in df_raw.columns.values:
            if attr == '_tid_':
                continue

            df_raw[attr] = df_raw[attr].str.strip().str.lower()
            
    df_raw = df_raw.fillna('_nan_')
    
    return df_dom, df_dk, df_clean, df_raw

def combine_df(df_dom, df_dk, df_clean):
    df_ref = df_dom.merge(df_dk, how='left',
                      on=['_tid_', 'attribute'], suffixes=['dom', 'dk']) \
                .merge(df_clean, on=['_tid_', 'attribute'])
    df_ref['is_dk'] = ~df_ref['_cid_dk'].isnull()
    df_ref.drop(['_cid_dk', '_cid_dom', '_vid_', 'init_index'], axis='columns', inplace=True)
    
    return df_ref

def error_df(df_ref):
    """
    df_ref is from combine_df.
    """
    fil_err = df_ref.init_value != df_ref.correct_val
    return df_ref[fil_err]

def repair_dfs(df_all, only_dk):
    """
    df_all is the first DF from analyze_results.
    
    if only_dk is True, only return repairs on DK cells.
    
    Returns DFs for (correct repairs,
        incorrect repairs on error cells,
        incorrect repairs on correct cells)
    """
    fil_dk = df_all.is_dk == only_dk
    
    # Correct prediction
    fil_cor_pred = df_all.correct_val == df_all.pred_val
    # Error cells
    fil_err = df_all.init_value != df_all.correct_val
    
    df_cor_rep = df_all[fil_err & fil_cor_pred & fil_dk] 
    df_incor_rep_err_cell = df_all[fil_err & ~fil_cor_pred & fil_dk] 
    df_incor_rep_cor_cell = df_all[~fil_err & ~fil_cor_pred & fil_dk] 
    
    return df_cor_rep, df_incor_rep_err_cell, df_incor_rep_cor_cell

def raw_density_from_cor(df_raw, df_targ):
    """
    Computes density of 'correct_val' in df_targ from raw data.
    """
    def row_density(row):
        attr, val = row['attribute'], row['correct_val']
        return {'attr': attr, 'val': val,
                'raw_count': (df_raw[attr] == val).sum(),
                'targ_count': ((df_targ.attribute == attr) & (df_targ.correct_val == val)).sum()}
    df_density = pd.DataFrame([row_density(row)
                         for _, row in df_targ.drop_duplicates(['attribute', 'correct_val']).iterrows()])
    return df_density.sort_values('targ_count', ascending=False)

def raw_with_errs(df_raw, df_err, delim=','):
    """
    df_raw from read_data and df_err from error_df.
    
    - `err_attrs` are the attributes in the tuple that are erroneous
    - `cor_vals` are the corresponding correct values
    - `raw_counts` are the corresponding counts of the CORRECT VALUE in the entire dataset
        a low `raw_count` means that it would be hard to repair it
    """
    df_err_density = raw_density_from_cor(df_raw, df_err).sort_values('targ_count', ascending=False)
    
    
    df_raw_err = df_raw[df_raw['_tid_'].isin(df_err['_tid_'])].copy()

    df_raw_err.insert(1, 'err_attrs',
                      df_raw_err['_tid_'].apply(
                          lambda tid: delim.join(df_err.loc[df_err['_tid_'] == tid, 'attribute'].values)))
    df_raw_err.insert(2, 'cor_vals',
                      df_raw_err['_tid_'].apply(
                          lambda tid: delim.join(df_err.loc[df_err['_tid_'] == tid, 'correct_val'].astype(str).values)))
       
    def add_raw_counts(row):
        return delim.join(str(df_err_density.loc[(df_err_density.attr == attr) & (df_err_density.val == val), 'raw_count'].iloc[0])
                           for attr, val in zip(row['err_attrs'].split(delim), row['cor_vals'].split(delim)))
    df_raw_err.insert(3, 'raw_counts',
                      df_raw_err[['err_attrs', 'cor_vals']].apply(add_raw_counts, axis='columns'))
    
    return df_raw_err


"""
Model stuff
"""

def model_metadata(prefix, init_val_idxs_suf='_init_val_idxs', train_val_idxs_suf='_train_val_idxs', 
                   attr_idxs_suf='_train_attr_idxs', sdict_suf='_sdict'):
    """
    Given the fpath and prefix for the model's output, returns (init val index map, train val index map, attr index map, torch state dict)
    """

    init_val_idxs = pickle.load(open('%s%s.pkl' % (prefix, init_val_idxs_suf), 'rb'))
    train_val_idxs = pickle.load(open('%s%s.pkl' % (prefix, train_val_idxs_suf), 'rb'))
    attr_idxs = pickle.load(open('%s%s.pkl' % (prefix, attr_idxs_suf), 'rb'))
    sdict = torch.load('%s%s.pkl' % (prefix, sdict_suf))
    
    return init_val_idxs, train_val_idxs, attr_idxs, sdict

# Returns softmax weight used in weighted average of init values for a given attr
def attrW_df(attr_idxs, sdict):
    attrW = sdict['attr_W'].numpy()

    def coattr_weights(attr):
        # convert to softmax probs
        probs = np.exp(attrW[attr_idxs[attr]]) / np.sum(np.exp(attrW[attr_idxs[attr]]))
        
        # sort by idx
        attr_names = [t[0] for t in filter(lambda t: t[0] != attr, sorted(list(attr_idxs.items()), key=lambda t: t[1]))]

        ret_df =  pd.DataFrame(list(zip(attr_names, probs)), columns=['givenattr/coattr', 'weight'])
        ret_df['weight'] = ret_df['weight'].apply(lambda w: round(w,3))
        ret_df.insert(0, 'attr', attr)
        return ret_df
    df_attrW = pd.concat([coattr_weights(attr) for attr in attr_idxs.keys()], axis=0)
    return df_attrW.pivot(index='attr', columns='givenattr/coattr')
