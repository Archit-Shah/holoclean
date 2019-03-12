import logging
import math
import pickle

from gensim.models import FastText
import numpy as np
import pandas as pd
import time
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, Softmax, ReLU
import torch.nn.functional as F
from tqdm import tqdm

from dataset import AuxTables
from evaluate import EvalEngine
from ..estimator import Estimator


class LookupDataset(Dataset):
    # Memoizes vectors (e.g. init vector, domain vector, negative indexes)
    # for every sample indexed by idx.
    class MemoizeVec:
        # If k_dims is none, we have a variable size indexing structure
        def __init__(self, n_samples, k_dims):
            self._variable = k_dims is None
            if self._variable:
                self._vec = [None for _ in range(n_samples)]
            else:
                self._vec = torch.zeros(n_samples, k_dims, dtype=torch.int64)
            self._isset = torch.zeros(n_samples, dtype=torch.int64)

        def __contains__(self, idx):
            return self._isset[idx] != 0

        # idx can be an integer, slice or a tuple or either
        def __getitem__(self, idx):
            if idx not in self:
                raise IndexError("tried to access un-set value at index %d" % idx)
            return self._vec[idx]

        # idx can be an integer, slice or a tuple or either
        def __setitem__(self, idx, vec):
            self._vec[idx] = vec
            if isinstance(idx, tuple):
                self._isset[idx[0]] = 1
            else:
                self._isset[idx] = 1

    def __init__(self, env, dataset, domain_df, neg_sample):
        self.env = env
        self.ds = dataset

        self._neg_sample = neg_sample
        self._attrs = sorted(self.ds.get_attributes())
        self._raw_data = self.ds.get_raw_data().copy()

        self.n_attrs = len(self._attrs)
        self._attr_idxs = {attr: idx for idx, attr in enumerate(self._attrs)}

        # Assign index for every unique value-attr
        self._val_idxs = {attr: {} for attr in self._attrs}

        # Reserve the 0th index as placeholder for padding in domain_idx and
        # for NULL values.
        cur_idx = 1
        for row in self._raw_data.to_records():
            for attr in self._attrs:
                if row[attr] in self._val_idxs[attr]:
                    continue

                # Use special index 0 for NULL values
                if row[attr] == '_nan_':
                    self._val_idxs[attr][row[attr]] = 0
                    continue

                self._val_idxs[attr][row[attr]] = cur_idx
                cur_idx += 1

        # Unique values (their indexes) by attr
        self._val_idxs_by_attr = {attr: np.array([self._val_idxs[attr][val]
            for val in self._raw_data[attr].unique()]) for attr in self._attrs}

        # Number of unique attr-values: size of vocab
        self.vocab_size = cur_idx

        self._raw_data_dict = self._raw_data.set_index('_tid_').to_dict('index')

        self._vid_to_idx = {vid: idx for idx, vid in enumerate(domain_df['_vid_'].values)}
        self._train_records = domain_df[['_tid_', 'attribute', 'init_value', 'init_index', 'domain', 'domain_size', 'is_clean']].to_records()
        self._max_domain = int(domain_df['domain_size'].max())

        # memoized stuff
        self._domain_idxs = self.MemoizeVec(len(self), self._max_domain)
        self._init_idxs = self.MemoizeVec(len(self), self.n_attrs - 1)
        self._neg_idxs = self.MemoizeVec(len(self), None)

    def __len__(self):
        return len(self._train_records)

    def _get_neg_idxs(self, idx, memoize=True):
        if not memoize or idx not in self._neg_idxs:
            cur = self._train_records[idx]

            # Value indices that are not in the domain
            neg_idxs = torch.LongTensor(np.setdiff1d(self._val_idxs_by_attr[cur['attribute']],
                    self._get_domain_idxs(idx),
                    assume_unique=True))
            if not memoize:
                return neg_idxs
            self._neg_idxs[idx] = neg_idxs
        return self._neg_idxs[idx]

    def _get_domain_idxs(self, idx, memoize=True):
        if not memoize or idx not in self._domain_idxs:
            cur = self._train_records[idx]

            # Domain values and their indexes (softmax indexes)
            domain_idxs = torch.LongTensor([self._val_idxs[cur['attribute']][val]
                    for val in cur['domain'].split('|||')])
            if not memoize:
                return domain_idxs
            self._domain_idxs[idx,0:len(domain_idxs)] = domain_idxs

        return self._domain_idxs[idx]

    def _get_init_idxs(self, idx, memoize=True):
        if not memoize or idx not in self._init_idxs:
            cur = self._train_records[idx]

            # Init values and their indexes (features)
            init_idxs = torch.LongTensor([self._val_idxs[attr][self._raw_data_dict[cur['_tid_']][attr]]
                    for attr in self._attrs if attr != cur['attribute']])
            if not memoize:
                return init_idxs
            self._init_idxs[idx] = init_idxs

        return self._init_idxs[idx]

    def __getitem__(self, vid):
        """
        Returns (feature vectors, domain_vectors, target class (init index))

        :param:`vid` is the desired VID.
        """
        idx = self._vid_to_idx[vid]

        init_idxs = self._get_init_idxs(idx)
        domain_idxs = self._get_domain_idxs(idx)

        cur = self._train_records[idx]
        dom_size = cur['domain_size']
        # Add negative samples to a most likely correct (clean) cell
        if self._neg_sample and cur['is_clean']:
            # It is faster not to memoize these.
            neg_idxs = self._get_neg_idxs(idx)
            neg_sample = torch.LongTensor(np.random.choice(neg_idxs,
                    size=min(len(neg_idxs), self._max_domain - dom_size),
                    replace=False))
            domain_idxs[dom_size:dom_size+len(neg_sample)] = neg_sample
            dom_size += len(neg_sample)

        # Mask out non-relevant values from padding (see below)
        domain_mask = torch.zeros(self._max_domain)
        domain_mask[dom_size:] = -1 * 1e9

        # Position of init in domain values (target)
        target = [cur['init_index']]

        attr_idx  = [self._attr_idxs[cur['attribute']]]

        return init_idxs, \
            domain_idxs, \
            torch.LongTensor(attr_idx), \
            domain_mask, \
            torch.LongTensor(target)

    def domain_values(self, idx):
        return self._train_records[idx]['domain'].split('|||')


class TupleEmbedding(Estimator, torch.nn.Module):
    WEIGHT_DECAY = 0

    def __init__(self, env, dataset, domain_df, embed_size=10, neg_sample=True,
            validate_fpath=None,
            validate_tid_col=None, validate_attr_col=None, validate_val_col=None):
        """
        :param dataset: (Dataset) original dataset
        :param neg_sample: (bool) add negative examples for clean cells during training
        :param validate_fpath: (string) filepath to validation CSV
        :param validate_tid_col: (string) column containing TID
        :param validate_attr_col: (string) column containing attribute
        :param validate_val_col: (string) column containing correct value
        """
        torch.nn.Module.__init__(self)
        Estimator.__init__(self, env, dataset)

        self._domain_df = domain_df.sort_values('_vid_')
        # Add DK information to domain dataframe
        df_dk = self.ds.aux_table[AuxTables.dk_cells].df
        self._domain_df = self._domain_df.merge(df_dk,
                on=['_tid_', 'attribute'], how='left', suffixes=('', '_dk'))
        self._domain_df['is_clean'] = self._domain_df['_cid__dk'].isnull()
        # Dataset
        self._dataset = LookupDataset(env, dataset, self._domain_df, neg_sample)

        # word2vec-like model.

        self._vocab_size = self._dataset.vocab_size
        self._n_attrs = self._dataset.n_attrs
        self._embed_size = embed_size

        self.in_W = torch.nn.Parameter(torch.zeros(self._vocab_size, self._embed_size))
        self.out_W = torch.nn.Parameter(torch.zeros(self._vocab_size, self._embed_size))
        self.out_B = torch.nn.Parameter(torch.zeros(self._vocab_size, 1))

        # logits fed into softmax used in weighted sum to combine
        # dot products of in_W and out_W per attribute.
        # Equivalent to choosing which input vectors to "focus" on.

        # Each row corresponds to the logits per each attr and there
        # are attr - 1 weights since we only have attrs - 1 init vectors.
        self.attr_W = torch.nn.Parameter(torch.zeros(self._n_attrs,
            self._n_attrs - 1))

        # Initialize all but the first 0th vector embedding (reserved).
        torch.nn.init.xavier_uniform_(self.in_W[1:])
        torch.nn.init.xavier_uniform_(self.out_W[1:])
        torch.nn.init.xavier_uniform_(self.out_B[1:])
        torch.nn.init.xavier_uniform_(self.attr_W)

        self._loss = CrossEntropyLoss()
        self._optimizer = Adam(self.parameters(), lr=self.env['learning_rate'], weight_decay=self.WEIGHT_DECAY)


        # Validation stuff

        if validate_fpath is not None
            and validate_tid_col is not None
            and validate_attr_col is not None
            and validate_val_col is not None:
            eengine = EvalEngine(self.env, self.ds)
            eengine.load_data(self.ds.raw_data.name + '_tuple_embedding_validate', validate_fpath,
                tid_col=validate_tid_col,
                attr_col=validate_attr_col,
                val_col=validate_val_col)
            self._validate_df = self._domain_df.merge(eengine.clean_data.df,
                    left_on=['_tid_', 'attribute'], right_on=['_tid_', '_attribute_'])
            self._validate_recs = self._validate_df[['_vid_', 'init_value', '_value_', 'is_clean']].to_records()
            self._validate_total_errs = (self._validate_df['init_value'] != self._validate_df['_value_']).sum()
            self._validate_detected_errs = ((self._validate_df['init_value'] != self._validate_df['_value_']) & ~self._validate_df['is_clean']).sum()

    def forward(self, init_idxs, domain_idxs, attr_idx, domain_mask, target_init_idx):
        # (batch, attrs - 1, embed size)
        init_vecs = self.in_W.index_select(0, init_idxs.view(-1)).view(*init_idxs.shape, self._embed_size)

        # Scale vectors to unit norm along the embedding dimension.
        # (batch, attrs - 1, embed size)
        init_vecs = F.normalize(init_vecs, p=2, dim=2)

        # (batch, 1, attrs - 1)
        attr_logits = self.attr_W.index_select(0, attr_idx.view(-1)).view(*attr_idx.shape, self._n_attrs - 1)

        # (batch, 1, attrs - 1)
        attr_weights = Softmax(dim=2)(attr_logits)

        # (batch, 1, embed size)
        combined_init = attr_weights.matmul(init_vecs)
        # (batch, embed size, 1)
        combined_init = combined_init.view(combined_init.shape[0], combined_init.shape[2], 1)

        # (batch, max domain, embed size)
        domain_vecs = self.out_W.index_select(0, domain_idxs.view(-1)).view(*domain_idxs.shape, self._embed_size)

        # (batch, max domain, 1)
        logits = domain_vecs.matmul(combined_init)

        # (batch, max domain, 1)
        domain_biases = self.out_B.index_select(0, domain_idxs.view(-1)).view(*domain_idxs.shape, 1)
        logits.add_(domain_biases)

        # (batch, max domain)
        logits = logits.view(logits.shape[:2])

        # Init bias
        for batch, idx in enumerate(target_init_idx.view(-1)):
            logits[batch, idx] += 1.

        # Add mask to void out-of-domain indexes
        logits.add_(domain_mask)

        # return (batch, max domain)
        return logits


    def train(self, num_epochs, batch_size, weight_entropy_lambda=0.,
            validate_results_prefix=None, validate_epoch=10, validate_prob=0.9):
        """
        :param num_epochs: (int) number of epochs to train for
        :param batch_size: (int) size of batches
        :param weight_entropy_lambda: (float) penalization strength for
            weights assigned to other attributes for a given attribute.
            A higher penalization strength means the model will depend
            on more attributes instead of putting all weight on a few
            attributes. Recommended values between 0 to 0.5.

        Validation parameters (only applicable if validation set was given
        during initialization):
        :param validate_results_prefix: (string) If not None, dumps
            validation results with this filepath prefix.
        :param validate_epoch: (int) run validation every validate_epoch-th epoch.
        :param validate_prob: (float) threshold for precision/recall statistics on
            high confidence predictions.
        """
        batch_losses = []

        # Main training loop.
        for epoch_idx in range(1, num_epochs+1):
            logging.debug("%s: epoch %d", type(self).__name__, epoch_idx)
            batch_cnt = 0
            for init_idxs, domain_idxs, attr_idx, domain_mask, target in tqdm(DataLoader(self._dataset, batch_size=batch_size, shuffle=True)):
                target = target.view(-1)

                batch_pred = self.forward(init_idxs, domain_idxs, attr_idx, domain_mask, target)

                # Do not train on null inits (target == -1)
                batch_pred = batch_pred[target >= 0]
                target = target[target >= 0]

                batch_loss = self._loss(batch_pred, target)

                # Add the negative entropy of the attr_W: that is we maximize
                # entropy of the logits of attr_W to encourage non-sparsity
                # of logits.
                attr_weights = Softmax(dim=1)(self.attr_W).view(-1)
                neg_attr_W_entropy = attr_weights.dot(attr_weights.log()) / self.attr_W.shape[0]
                batch_loss.add_(weight_entropy_lambda * neg_attr_W_entropy)

                batch_losses.append(float(batch_loss))
                self.zero_grad()
                batch_loss.backward()

                # Do not update weights for 0th reserved vectors.
                self.in_W._grad[0].zero_()
                self.out_W._grad[0].zero_()
                self.out_B._grad[0].zero_()

                self._optimizer.step()
                batch_cnt += 1
            logging.debug('%s: average batch loss: %f',
                    type(self).__name__,
                    sum(batch_losses[-1 * batch_cnt:]) / batch_cnt)

            # Validation stuff
            if self._validate_recs is not None and epoch_idx % val_epoch == 0:
                val_res = self.validate(val_prob)
                logging.debug("Precision: %.2f, Recall: %.2f, Repair Recall: %.2f",
                        val_res['precision'], val_res['recall'], val_res['repair_recall'])
                logging.debug("(DK) Precision: %.2f, Recall: %.2f", val_res['dk_precision'], val_res['dk_recall'])
                logging.debug("(prob >= %.2f & DK) Precision: %.2f, Recall: %.2f", val_prob,
                        val_res['hc_precision'], val_res['hc_recall'])

                if validate_results_prefix is not None:
                    epoch_prefix = '%s_at_%d_epoch' % (validate_results_prefix, epoch_idx)
                    self.dump_model(epoch_prefix)
                    self.dump_predictions(epoch_prefix)

        return batch_losses

    def dump_model(self, prefix):
        """
        Dump this model's parameters and other metadata (e.g. attr-val to corresponding
        index in embedding matrix) with the given :param:`prefix`.
        """
        torch.save(self.state_dict(), '%s_sdict.pkl' % prefix)
        pickle.dump(self._dataset._val_idxs, open('%s_val_idxs.pkl' % prefix, 'wb'))
        pickle.dump(self._dataset._attr_idxs, open('%s_attr_idxs.pkl' % prefix, 'wb'))

    def dump_predictions(self, fpath):
        """
        Dump inference results to :param:`fpath`.
        """
        preds = self.predict_pp_batch()

        results = []
        for vid, (pred, row) in enumerate(zip(preds, self._domain_df.to_records())):
            assert vid == row['_vid_']
            for val, proba in pred:
                results.append({'_tid_': row['_tid_'],
                    '_vid_': vid,
                    'attribute': row['attribute'],
                    'inf_val': val,
                    'proba': proba})

        results = pd.DataFrame(results)
        results.to_pickle('{}.pkl'.format(fpath))

    def validate(self, prob):
        # repairs on clean + DK cells
        cor_repair = 0
        incor_repair = 0

        # repairs only on DK cells
        cor_repair_dk = 0
        incor_repair_dk = 0

        # repairs only on DK cells AND >= prob
        cor_repair_hc = 0
        incor_repair_hc = 0

        logging.debug('Running validation set...')

        validation_preds = self.predict_pp_batch(self._validate_df)

        for preds, row in tqdm(list(zip(validation_preds, self._validate_recs))):
            inf_val, inf_prob = max(preds, key=lambda t: t[1])

            if row['init_value'] != inf_val:
                # Correct val == inf val
                if row['_value_'] == inf_val:
                    cor_repair += 1
                    if not row['is_clean']:
                        cor_repair_dk += 1
                        if inf_prob >= prob:
                            cor_repair_hc += 1
                # Correct val != inf val
                else:
                    incor_repair += 1
                    if not row['is_clean']:
                        incor_repair_dk += 1
                        if inf_prob >= prob:
                            incor_repair_hc += 1

        return {'precision': cor_repair / max(cor_repair + incor_repair, 1),
            'recall': cor_repair / self._validate_total_errs,
            'dk_precision': cor_repair_dk / max(cor_repair_dk + incor_repair_dk, 1),
            'dk_recall': cor_repair_dk / self._validate_total_errs,
            'repair_recall': cor_repair_dk / self._validate_detected_errs,
            'hc_precision': cor_repair_hc / max(cor_repair_hc + incor_repair_hc, 1),
            'hc_recall': cor_repair_hc / self._validate_total_errs,
            }


    def predict_pp(self, row, attr=None, values=None):
        init_idxs, domain_idxs, attr_idx, domain_mask, target = self._dataset[row['_vid_']]
        # Prepend batch size 1 dimension
        init_idxs = init_idxs.unsqueeze(0)
        domain_idxs = domain_idxs.unsqueeze(0)
        attr_idx = attr_idx.unsqueeze(0)
        domain_mask = domain_mask.unsqueeze(0)
        target = target.unsqueeze(0)

        pred_Y = self.forward(init_idxs, domain_idxs, attr_idx, domain_mask, target)

        return zip(self._dataset.domain_values(row['_vid_']), map(float, Softmax(dim=0)(pred_Y.view(-1))))

    def predict_pp_batch(self, df=None):
        """
        Performs batch prediction.

        df must have column '_vid_'.
        """
        if df is None:
            df = self._domain_df

        ds_tuples = [self._dataset[idx] for idx in df['_vid_'].values]
        init_idxs, domain_idxs, attr_idx, domain_mask, target = [
                torch.cat([vec.unsqueeze(0) for vec in ith_vecs])
                for ith_vecs in list(zip(*ds_tuples))]

        pred_Y = self.forward(init_idxs, domain_idxs, attr_idx, domain_mask, target)

        for logits, idx in zip(pred_Y, df['_vid_'].values):
            yield zip(self._dataset.domain_values(idx), map(float, Softmax(dim=0)(logits)))
