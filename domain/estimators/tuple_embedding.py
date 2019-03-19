import logging
import pickle
import csv

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import CrossEntropyLoss, Softmax, ReLU
import torch.nn.functional as F
from tqdm import tqdm

from dataset import AuxTables
from evaluate import EvalEngine
from ..estimator import Estimator
from utils import NULL_REPR


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

    def __init__(self, env, dataset, domain_df, embed_model, neg_sample, max_train_domain, memoize):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) dataframe containing VIDs and their
            domain values we want to train on. VIDs not included in this
            dataframe will not be trained on e.g. if you only want to
            sub-select VIDs of certain attributes.
        :param neg_sample: (bool) add negative examples for clean cells during training
        """
        # reference to model to access embeddings
        """
        FOR FULL DOT PROD
        """
        self._embed_model = embed_model
        self._max_domain = max_train_domain
        self.inference_mode = False
        """
        END FULL DOT PROD
        """





        self.env = env
        self.ds = dataset
        self.memoize = memoize

        self._neg_sample = neg_sample
        """
        Init attrs/vals herein refers to attributes and values embedded to use
        as context during training.

        Train attrs/vals herein refers to the attributes/columns and possible values
        embedded to use as the targets during training.
        """
        # Attributes to derive context from.
        self._init_attrs = sorted(self.ds.get_attributes())
        # Attributes to train on (i.e. target columns).
        self.train_attrs = sorted(domain_df['attribute'].unique())
        self._raw_data = self.ds.get_raw_data().copy()

        self.n_init_attrs = len(self._init_attrs)
        self.n_train_attrs = len(self.train_attrs)
        self._train_attr_idxs = {attr: idx for idx, attr in enumerate(self.train_attrs)}

        # Assign index for every unique value-attr (init values, input)
        self._init_val_idxs = {attr: {} for attr in self._init_attrs}
        # Assign index for every unique value-attr (train/possible values, target)
        self._train_val_idxs = {attr: {} for attr in self.train_attrs}

        self._train_idx_to_val = {0: '_nan_'}

        # Reserve the 0th index as placeholder for padding in domain_idx and
        # for NULL values.
        cur_init_idx = 1
        for row in self._raw_data.to_records():
            for attr in self._init_attrs:
                if row[attr] in self._init_val_idxs[attr]:
                    continue

                # Use special index 0 for NULL values
                if row[attr] == '_nan_':
                    self._init_val_idxs[attr][row[attr]] = 0
                    continue

                # Assign index for init values
                self._init_val_idxs[attr][row[attr]] = cur_init_idx
                cur_init_idx += 1

        cur_train_idx = 1
        for row in domain_df.to_records():
            val = row['init_value']
            attr = row['attribute']
            # Assign index for train/possible values
            if val in self._train_val_idxs[attr]:
                continue

            # Use special index 0 for NULL values
            if val == '_nan_':
                self._train_val_idxs[attr][val] = 0
                continue

            # Assign index for train/domain values
            self._train_val_idxs[attr][val] = cur_train_idx
            self._train_idx_to_val[cur_train_idx] = val
            cur_train_idx += 1

        # Unique train values (their indexes) by attr
        self._train_val_idxs_by_attr = {attr: torch.LongTensor([v for v in self._train_val_idxs[attr].values() if v != 0])
                for attr in self.train_attrs}

        # Number of unique INIT attr-values
        self.n_init_vals = cur_init_idx
        self.n_train_vals = cur_train_idx

        self._raw_data_dict = self._raw_data.set_index('_tid_').to_dict('index')

        self._vid_to_idx = {vid: idx for idx, vid in enumerate(domain_df['_vid_'].values)}


        """
        FOR FULL DOT PROD
        """
        # self._train_records = domain_df[['_tid_', 'attribute', 'init_value', 'is_clean']].to_records()
        """
        FOR FULL DOT PROD
        """


        """
        FOR DOMAIN DF
        """
        self._train_records = domain_df[['_tid_', 'attribute', 'init_value',
                                         'init_index', 'domain', 'domain_size', 'is_clean']].to_records()
        self._max_domain = int(domain_df['domain_size'].max())
        """
        FOR DOMAIN DF
        """

        # memoized stuff
        if memoize:
            self._domain_idxs = self.MemoizeVec(len(self), self._max_domain)
            self._init_idxs = self.MemoizeVec(len(self), self.n_init_attrs - 1)
            self._neg_idxs = self.MemoizeVec(len(self), None)

    def __len__(self):
        return len(self._train_records)

    def _get_neg_idxs(self, idx):
        if not self.memoize or idx not in self._neg_idxs:
            cur = self._train_records[idx]

            # Value indices that are not in the domain
            neg_idxs = torch.LongTensor(np.setdiff1d(self._train_val_idxs_by_attr[cur['attribute']],
                    self._get_domain_idxs(idx),
                    assume_unique=True))
            if not self.memoize:
                return neg_idxs
            self._neg_idxs[idx] = neg_idxs
        return self._neg_idxs[idx]

    def _get_domain_idxs(self, idx):
        if not self.memoize or idx not in self._domain_idxs:
            cur = self._train_records[idx]

            """
            FOR DOMAIN DF
            """
            # Domain values and their indexes (softmax indexes)
            domain_idxs = torch.zeros(self._max_domain).type(torch.LongTensor)
            domain_idxs[:cur['domain_size']] = torch.LongTensor([self._train_val_idxs[cur['attribute']][val]
                    for val in cur['domain'].split('|||')])

            """
            FOR FULL DOT PROD
            """
            # domain_idxs = torch.zeros(self._max_domain).type(torch.LongTensor)
            # if cur['init_value'] != '_nan_':
            #     domain_idxs[0] = self._train_val_idxs[cur['attribute']][cur['init_value']]

            if not self.memoize:
                return domain_idxs
            self._domain_idxs[idx,0:len(domain_idxs)] = domain_idxs

        return self._domain_idxs[idx]

    def _get_init_idxs(self, idx):
        if not self.memoize or idx not in self._init_idxs:
            cur = self._train_records[idx]

            # Init values and their indexes (features)
            if self.inference_mode:
                # During inference mode, we may encounter out-of-vocab
                # context vectors. Simply assign it the 0 vector.
                init_idxs = torch.LongTensor([self._init_val_idxs[attr].get(self._raw_data_dict[cur['_tid_']][attr], 0)
                                              for attr in self._init_attrs if attr != cur['attribute']])
            else:
                init_idxs = torch.LongTensor([self._init_val_idxs[attr][self._raw_data_dict[cur['_tid_']][attr]]
                        for attr in self._init_attrs if attr != cur['attribute']])
            if not self.memoize:
                return init_idxs
            self._init_idxs[idx] = init_idxs

        return self._init_idxs[idx]

    def set_mode(self, inference_mode):
        """
        inference_mode = True means to start inference (i.e. use KNN
        for domain instead of random vectors).
        """
        self.inference_mode = inference_mode

    def __getitem__(self, vid):
        """
        Returns (feature vectors, domain_vectors, target class (init index))

        :param:`vid` is the desired VID or VIDs.
        """

        """
        FOR BATCH LOOKUP
        """
        # if hasattr(vid, '__len__'):
        #     idxs = np.array([self._vid_to_idx[one_vid] for one_vid in vid])
        #     cur_records = [self._train_records[idx] for idx in idxs]

        #     # (batch_size, init_attrs - 1)
        #     init_idxs = torch.stack([self._get_init_idxs(idx) for idx in idxs])
        #     # (batch_size, max domain)
        #     domain_idxs = torch.stack([self._get_domain_idxs(idx) for idx in idxs])
        #     # (batch_size, 1)
        #     # attr_idxs  = torch.LongTensor([self._train_attr_idxs[cur['attribute']]
        #                                    # for cur in cur_records]).unsqueeze(-1)

        #     # (batch_size, embed_size)
        #     context_vecs = self._embed_model._get_combined_init_vec(
        #         init_idxs,
        #         torch.zeros(len(idxs), 1).type(torch.LongTensor)
        #         # attr_idxs
        #     ).squeeze(-1)

        #     # TODO: This only really works on one attribute at a time since
        #     # out_W contains embeddings for multiple attributes
        #     # We could separate out_W by attribute
        #     # (batch_size, # of train values)
        #     sorted_idxs = (context_vecs.matmul(self._embed_model.out_W.transpose(0, 1)) + self._embed_model.out_B.view(1, -1)).argsort(1, descending=True)

        #     # (batch_size, # of train values - 1)
        #     sorted_idxs = sorted_idxs[sorted_idxs.ne(domain_idxs[:, 0:1])].view(len(idxs), -1)

        #     neg_samples = sorted_idxs[:, :self._max_domain - 1]
        #     domain_idxs[:, 1:] = neg_samples

        #     domain_mask = torch.zeros_like(domain_idxs).type(torch.FloatTensor)
        #     domain_mask[(domain_idxs == 0)] = -1* 1e-9

        #     # target = torch.zeros(len(vid), 1).type(torch.LongTensor)

        #     return torch.LongTensor(vid), \
        #         init_idxs, \
        #         domain_idxs, \
        #         None, \
        #         domain_mask, \
        #         None
        """
        FOR BATCH LOOKUP
        """


        idx = self._vid_to_idx[vid]
        cur = self._train_records[idx]

        init_idxs = self._get_init_idxs(idx)
        domain_idxs = self._get_domain_idxs(idx)
        attr_idx  = torch.LongTensor([self._train_attr_idxs[cur['attribute']]])

        """
        FOR DOMAIN DF
        """
        dom_size = cur['domain_size']
        # Add negative samples to a most likely correct (clean) cell
        if self._neg_sample and dom_size < self._max_domain and cur['is_clean']:
            # It is faster not to memoize these.
            neg_idxs = self._get_neg_idxs(idx)
            neg_sample = torch.LongTensor(np.random.choice(neg_idxs,
                    size=min(len(neg_idxs), self._max_domain - dom_size),
                    replace=False))

            domain_idxs[dom_size:dom_size+len(neg_sample)] = neg_sample
            dom_size += len(neg_sample)

        # Position of init in domain values (target)
        target = cur['init_index']
        """
        END DOMAIN DF
        """




        """
        FOR FULL DOT PROD`
        """
        # dom_size = (domain_idxs != 0).sum()

        # # 1-D (embed_size) vector
        # context_vec = self._embed_model._get_combined_init_vec(
        #         init_idxs.unsqueeze(0),
        #         attr_idx.unsqueeze(0),
        #     ).view(-1)


        # # Only check neighbours of our current attribute
        # slice_idxs = self._train_val_idxs_by_attr[cur['attribute']]
        # # Omit our init value embedding
        # if dom_size > 0:
        #     slice_idxs = slice_idxs[slice_idxs != domain_idxs[0]]

        # # Get top KNN neighbours
        # sorted_idxs = (self._embed_model.out_W[slice_idxs].matmul(context_vec) +
        #         self._embed_model.out_B[slice_idxs].view(-1)).argsort(descending=True)

        # neg_samples = slice_idxs[sorted_idxs[:self._max_domain - dom_size]]
        # domain_idxs[dom_size:dom_size+len(neg_samples)] = neg_samples
        # dom_size += len(neg_samples)

        # # We assign the first index in domain_idxs as our target init index
        # target = 0

        # # Do not train on NULL init values/cells
        # if cur['init_value'] == '_nan_':
        #     target = -1

        # assert (cur['init_value'] == '_nan_' and target == -1) or \
        #        (target == 0 and cur['init_value'] == self._train_idx_to_val[int(domain_idxs[0])])
        """
        END FULL DOT PROD
        """






        # Mask out non-relevant values from padding (see below)
        domain_mask = torch.zeros(self._max_domain)
        domain_mask[dom_size:] = -1 * 1e9


        return vid, \
            init_idxs, \
            domain_idxs, \
            attr_idx, \
            domain_mask, \
            torch.LongTensor([target])

    def domain_values(self, vid):
        idx = self._vid_to_idx[vid]
        return self._train_records[idx]['domain'].split('|||')

class VidSampler(Sampler):
    def __init__(self, domain_df, shuffle=True, train_only_clean=True):
        # No NULLs and non-zero domain
        domain_df = domain_df[domain_df['init_value'] != NULL_REPR]

        # Train on only clean cells
        if train_only_clean:
            self._vids = domain_df.loc[domain_df['is_clean'], '_vid_']
        else:
            self._vids = domain_df['_vid_'].values

        self._vids = self._vids

        if shuffle:
            self._vids = np.random.permutation(self._vids)

    def __iter__(self):
        return iter(self._vids.tolist())

    def __len__(self):
        return len(self._vids)


class TupleEmbedding(Estimator, torch.nn.Module):
    WEIGHT_DECAY = 0

    def __init__(self, env, dataset, domain_df,
            train_attrs=None,
            max_train_domain=20,
            memoize=True,
            embed_size=10, neg_sample=True,
            validate_fpath=None,
            validate_tid_col=None, validate_attr_col=None, validate_val_col=None):
        """
        :param dataset: (Dataset) original dataset
        :param domain_df: (DataFrame) dataframe containing domain values
        :param train_attrs: (list[str]) attributes/columns to train on. If None,
            trains on every attribute.
        :param neg_sample: (bool) add negative examples for clean cells during training
        :param validate_fpath: (string) filepath to validation CSV
        :param validate_tid_col: (string) column containing TID
        :param validate_attr_col: (string) column containing attribute
        :param validate_val_col: (string) column containing correct value
        """
        torch.nn.Module.__init__(self)
        Estimator.__init__(self, env, dataset, domain_df)

        """
        FOR DOMAIN DF
        """
        filter_empty_domain = self.domain_df['domain_size'] == 0
        if filter_empty_domain.sum():
            logging.warning('%s: removing %d cells with empty domains',
                type(self).__name__,
                filter_empty_domain.sum())
            self.domain_df = self.domain_df[~filter_empty_domain]
        """
        FOR DOMAIN DF
        """

        # Add DK information to domain dataframe
        df_dk = self.ds.aux_table[AuxTables.dk_cells].df
        self.domain_df = self.domain_df.merge(df_dk,
                on=['_tid_', 'attribute'], how='left', suffixes=('', '_dk'))
        self.domain_df['is_clean'] = self.domain_df['_cid__dk'].isnull()

        if train_attrs is not None:
            self.domain_df = self.domain_df[self.domain_df['attribute'].isin(train_attrs)]
        self.domain_recs = self.domain_df.to_records()

        # Dataset
        self._dataset = LookupDataset(env, dataset, self.domain_df, self, neg_sample, max_train_domain, memoize)

        self._train_attrs = self._dataset.train_attrs
        self._train_idx_to_val = self._dataset._train_idx_to_val

        # word2vec-like model.

        self._n_init_vals = self._dataset.n_init_vals
        self._n_train_vals = self._dataset.n_train_vals
        self._n_init_attrs = self._dataset.n_init_attrs
        self._n_train_attrs = self._dataset.n_train_attrs
        self._embed_size = embed_size

        self.in_W = torch.nn.Parameter(torch.zeros(self._n_init_vals, self._embed_size))
        self.out_W = torch.nn.Parameter(torch.zeros(self._n_train_vals, self._embed_size))
        self.out_B = torch.nn.Parameter(torch.zeros(self._n_train_vals, 1))

        # logits fed into softmax used in weighted sum to combine
        # dot products of in_W and out_W per attribute.
        # Equivalent to choosing which input vectors to "focus" on.
        # Each row corresponds to the logits per each attr/column we want
        # to predict for and there
        # are init_attr - 1 weights since we only have init_attr - 1 init
        # vectors.
        self.attr_W = torch.nn.Parameter(torch.zeros(self._n_train_attrs,
            self._n_init_attrs - 1))

        # Initialize all but the first 0th vector embedding (reserved).
        torch.nn.init.xavier_uniform_(self.in_W[1:])
        torch.nn.init.xavier_uniform_(self.out_W[1:])
        torch.nn.init.xavier_uniform_(self.out_B[1:])
        torch.nn.init.xavier_uniform_(self.attr_W)

        self._loss = CrossEntropyLoss()
        self._optimizer = Adam(self.parameters(), lr=self.env['learning_rate'], weight_decay=self.WEIGHT_DECAY)


        # Validation stuff
        self._do_validation = False
        if validate_fpath is not None \
            and validate_tid_col is not None \
            and validate_attr_col is not None \
            and validate_val_col is not None:
            eengine = EvalEngine(self.env, self.ds)
            eengine.load_data(self.ds.raw_data.name + '_tuple_embedding_validate', validate_fpath,
                tid_col=validate_tid_col,
                attr_col=validate_attr_col,
                val_col=validate_val_col)
            self._validate_df = self.domain_df.merge(eengine.clean_data.df,
                    left_on=['_tid_', 'attribute'], right_on=['_tid_', '_attribute_'])

            self._validate_recs = self._validate_df[['_vid_', 'init_value', '_value_', 'is_clean']].to_records()
            self._validate_total_errs = (self._validate_df['init_value'] != self._validate_df['_value_']).sum()
            self._validate_detected_errs = ((self._validate_df['init_value'] != self._validate_df['_value_']) & ~self._validate_df['is_clean']).sum()
            self._do_validation = True

    def _get_combined_init_vec(self, init_idxs, attr_idx):
        """
        Constructs the "context vector" by combining the init embedding vectors.
        """
        # (batch, init_attrs - 1, embed size)
        init_vecs = self.in_W.index_select(0, init_idxs.view(-1)).view(*init_idxs.shape, self._embed_size)

        # Scale vectors to unit norm along the embedding dimension.
        # (batch, init_attrs - 1, embed size)
        init_vecs = F.normalize(init_vecs, p=2, dim=2)

        # (batch, 1, init_attrs - 1)
        attr_logits = self.attr_W.index_select(0, attr_idx.view(-1)).view(*attr_idx.shape, self._n_init_attrs - 1)

        # (batch, 1, init_attrs - 1)
        attr_weights = Softmax(dim=2)(attr_logits)

        # (batch, 1, embed size)
        combined_init = attr_weights.matmul(init_vecs)
        # (batch, embed size, 1)
        combined_init = combined_init.view(combined_init.shape[0], combined_init.shape[2], 1)

        # (batch, embed size, 1)
        return combined_init

    def forward(self, init_idxs, domain_idxs, attr_idx, domain_mask, target_init_idx):
        # (batch, embed size, 1)
        combined_init = self._get_combined_init_vec(init_idxs, attr_idx)

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


    def train(self, shuffle=True, train_only_clean=True):
        """
        :param num_epochs: (int) number of epochs to train for
        :param batch_size: (int) size of batches
        :param shuffle: (bool) shuffle the dataset while training
        :param weight_entropy_lambda: (float) penalization strength for
            weights assigned to other attributes for a given attribute.
            A higher penalization strength means the model will depend
            on more attributes instead of putting all weight on a few
            attributes. Recommended values between 0 to 0.5.
        """

        num_epochs = self.env['embed_estimator_num_epochs']
        weight_entropy_lambda = self.env['embed_estimator_lambda']
        batch_size = self.env['embed_estimator_batch_size']

        dump_batch = self.env['embed_estimator_dump_batch']
        dump_prefix = self.env['embed_estimator_dump_prefix']

        validate_epoch = self.env['embed_estimator_validate_epoch']
        validate_prefix = self.env['embed_estimator_validate_prefix']


        # Returns VIDs to train on.
        sampler = VidSampler(self.domain_df, shuffle=shuffle, train_only_clean=train_only_clean)

        logging.debug("%s: training (lambda = %f) on %d cells (%d cells in total) in %d columns: %s",
                      type(self).__name__,
                      weight_entropy_lambda,
                      len(sampler),
                      self.domain_df.shape[0],
                      len(self._train_attrs),
                      self._train_attrs)

        batch_losses = []
        # Main training loop.
        for epoch_idx in range(1, num_epochs+1):
            logging.debug("%s: epoch %d", type(self).__name__, epoch_idx)
            batch_cnt = 0
            for _, init_idxs, domain_idxs, attr_idx, domain_mask, target in tqdm(DataLoader(self._dataset,
                batch_size=batch_size, sampler=sampler)):
                target = target.view(-1)

                batch_pred = self.forward(init_idxs, domain_idxs, attr_idx, domain_mask, target)

                # Do not train on null inits (target == -1)
                # We already filter these in VidSampler
                # batch_pred = batch_pred[target >= 0]
                # target = target[target >= 0]
                # if len(target) == 0:
                #     continue

                batch_loss = self._loss(batch_pred, target)

                # Add the negative entropy of the attr_W to the cost: that is
                # we maximize entropy of the logits of attr_W to encourage
                # non-sparsity of logits.
                if weight_entropy_lambda != 0.:
                    import pdb; pdb.set_trace()
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

                if batch_cnt % dump_batch == 0:
                    self.dump_model('%s_batch_%d_epoch_%d' % (dump_prefix, batch_cnt, epoch_idx))
            self.dump_model('%s_epoch_%d' % (dump_prefix, epoch_idx))

            logging.debug('%s: average batch loss: %f',
                    type(self).__name__,
                    sum(batch_losses[-1 * batch_cnt:]) / batch_cnt)

            # Validation stuff
            if self._do_validation and epoch_idx % validate_epoch == 0:
                val_res = self.validate()
                logging.debug("Precision: %.2f, Recall: %.2f, Repair Recall: %.2f",
                        val_res['precision'], val_res['recall'], val_res['repair_recall'])
                logging.debug("(DK) Precision: %.2f, Recall: %.2f", val_res['dk_precision'], val_res['dk_recall'])

                if validate_prefix is not None:
                    epoch_prefix = '%s_at_%d_epoch' % (validate_prefix, epoch_idx)
                    self.dump_model(epoch_prefix)
                    self.dump_predictions(epoch_prefix)

        return batch_losses

    def dump_model(self, prefix):
        """
        Dump this model's parameters and other metadata (e.g. attr-val to corresponding
        index in embedding matrix) with the given :param:`prefix`.
        """
        torch.save(self.state_dict(), '%s_sdict.pkl' % prefix)
        pickle.dump(self._dataset._init_val_idxs, open('%s_init_val_idxs.pkl' % prefix, 'wb'))
        pickle.dump(self._dataset._train_val_idxs, open('%s_train_val_idxs.pkl' % prefix, 'wb'))
        pickle.dump(self._dataset._train_attr_idxs, open('%s_train_attr_idxs.pkl' % prefix, 'wb'))

    def dump_predictions(self, fpath):
        """
        Dump inference results to :param:`fpath`.
        """
        self._dataset.set_mode(inference_mode=True)
        preds = self.predict_pp_batch()
        self._dataset.set_mode(inference_mode=False)

        logging.debug('constructing and dumping predictions...')
        with open(fpath + '.csv', 'w') as csv_f:
            writer = csv.writer(csv_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Write header.
            writer.writerow(['_tid_', '_vid_', 'attribute', 'inf_val', 'proba'])

            written_rows = 0
            for ((vid, pred), row) in zip(preds, self.domain_recs):
                written_rows += 1
                assert vid == row['_vid_']
                max_val, max_proba = max(pred, key=lambda t: t[1])
                writer.writerow([
                    row['_tid_'],
                    vid,
                    row['attribute'],
                    max_val,
                    max_proba
                ])
                if written_rows % 1000 == 0:
                    csv_f.flush()

        # results = []
        # for ((vid, pred), row) in zip(preds, self.domain_df.to_records()):
        #     assert vid == row['_vid_']
        #     max_val, max_proba = max(pred, key=lambda t: t[1])
        #     results.append({'_tid_': row['_tid_'],
        #         '_vid_': vid,
        #         'attribute': row['attribute'],
        #         'inf_val': max_val,
        #         'proba': max_proba})

        #     # for val, proba in pred:
        #     #     results.append({'_tid_': row['_tid_'],
        #     #         '_vid_': vid,
        #     #         'attribute': row['attribute'],
        #     #         'inf_val': val,
        #     #         'proba': proba})
        #results = pd.DataFrame(results)
        #logging.debug('dumping prediction df...')
        #results.to_pickle('{}.pkl'.format(fpath))

    def validate(self):
        # repairs on clean + DK cells
        cor_repair = 0
        incor_repair = 0

        # repairs only on DK cells
        cor_repair_dk = 0
        incor_repair_dk = 0

        logging.debug('Running validation set...')

        validation_preds = self.predict_pp_batch(self._validate_df)

        for (vid, preds), row in tqdm(list(zip(validation_preds, self._validate_recs))):
            assert vid == row['_vid_']

            inf_val, inf_prob = max(preds, key=lambda t: t[1])

            if row['init_value'] != inf_val:
                # Correct val == inf val
                if row['_value_'] == inf_val:
                    cor_repair += 1
                    if not row['is_clean']:
                        cor_repair_dk += 1
                # Correct val != inf val
                else:
                    incor_repair += 1
                    if not row['is_clean']:
                        incor_repair_dk += 1

        return {'precision': cor_repair / max(cor_repair + incor_repair, 1),
            'recall': cor_repair / self._validate_total_errs,
            'dk_precision': cor_repair_dk / max(cor_repair_dk + incor_repair_dk, 1),
            'dk_recall': cor_repair_dk / self._validate_total_errs,
            'repair_recall': cor_repair_dk / self._validate_detected_errs,
            }


    def predict_pp(self, row, attr=None, values=None):
        """
        row defines __getitem__(vid).

        One should only pass in VIDs that have been trained on (see :param:`train_attrs`).
        """
        vid, init_idxs, domain_idxs, attr_idx, domain_mask, target = self._dataset[row['_vid_']]
        # Prepend batch size 1 dimension
        init_idxs = init_idxs.unsqueeze(0)
        domain_idxs = domain_idxs.unsqueeze(0)
        attr_idx = attr_idx.unsqueeze(0)
        domain_mask = domain_mask.unsqueeze(0)
        target = target.unsqueeze(0)

        pred_Y = self.forward(init_idxs, domain_idxs, attr_idx, domain_mask, target)

        # return vid, zip(self._dataset.domain_values(row['_vid_']), map(float, Softmax(dim=0)(pred_Y.view(-1))))



        domain_values = [self._train_idx_to_val[idx] for idx in domain_idxs.numpy()]
        return vid, zip(domain_values, map(float, Softmax(dim=0)(pred_Y.view(-1))))

    def predict_pp_batch(self, df=None):
        """
        Performs batch prediction.

        df must have column '_vid_'.
        One should only pass in VIDs that have been trained on (see
        :param:`train_attrs`).
        """
        if df is None:
            df = self.domain_df

        logging.debug('batch predicting...')

        logging.debug('getting indices...')

        """
        FOR FULL DOT PROD AND BATCH LOOKUP
        """
        # ds_tuples = []
        # for vids in tqdm(np.array_split(df['_vid_'].values, 1000)):
        #     ds_tuples.append(self._dataset[vids][1:])

        # init_idxs, domain_idxs, attr_idx, domain_mask, target = [torch.cat(tensors, dim=0) if tensors[0] is not None else None for tensors in list(zip(*ds_tuples))]
        # attr_idx = torch.zeros(init_idxs.shape[0], 1).type(torch.LongTensor)
        # target = torch.zeros(init_idxs.shape[0], 1).type(torch.LongTensor)
        """
        FOR FULL DOT PROD AND BATCH LOOKUP
        """

        """
        FOR DOMAIN DF
        """
        ds_tuples = [self._dataset[vid][1:] for vid in df['_vid_'].values]
        init_idxs, domain_idxs, attr_idx, domain_mask, target = [
                torch.cat([vec.unsqueeze(0) for vec in ith_vecs])
                for ith_vecs in list(zip(*ds_tuples))]
        """
        FOR DOMAIN DF
        """


        logging.debug('done getting indices.')

        logging.debug('starting batch prediction...')

        pred_Y = self.forward(init_idxs, domain_idxs, attr_idx, domain_mask, target)
        logging.debug('done batch prediction')

        # for logits, vid in zip(pred_Y, df['_vid_'].values):
        #     yield vid, zip(self._dataset.domain_values(vid), map(float, Softmax(dim=0)(logits)))


        for logits, vid, cur_dom_idxs in zip(pred_Y, df['_vid_'].values, domain_idxs):
            domain_values = [self._train_idx_to_val[idx] for idx in cur_dom_idxs.numpy()]
            yield vid, zip(domain_values, map(float, Softmax(dim=0)(logits)))
