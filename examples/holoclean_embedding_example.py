import sys
sys.path.append('../')
import holoclean
from detect import *


"""
Modify these per experiment.
"""

### Model hyperparameters
# Regularization term for weights used in combining context
LAMBDA = 0
# Embedding size per word
EMBED_SIZE = 10
# Epochs for embedding model
EPOCHS = 50
# Batch size for embedding model
BATCH = 32
# Columns/attributes to train emedding estimator on (None = all columns)
TRAIN_ATTRS = None
# Train only on clean cells (cells not marked by error detectors as DK/don't know)
TRAIN_ONLY_CLEAN = False
# # of nearest neighbours to use in non-domain df mode
KNN = None

### Other hyperparameters
# Memoizes training data. Warning: for large datasets, this may blow up memory.
MEMOIZE = True

### Data settings
# Name of dataset
DS_NAME = 'hospital'
# Filepath to raw data CSV
RAW_FPATH = '../testdata/hospital.csv'
# Filename to denial constraints
DC_FPATH = '../testdata/hospital_constraints.txt'

### Model results settings
# Dump domain of dataset with this prefix
DOMAIN_PREFIX = 'experiments/hospital2/meta/hospital_domain'
DUMP_PREFIX = 'experiments/hospital2/results/embed_epochs_{}epoch_{}batch_{}lambda_{}_attrs'.format(
        EPOCHS,
        BATCH,
        LAMBDA,
        "ALL" if TRAIN_ATTRS is None else ','.join(TRAIN_ATTRS))
# Dump model params every nth batch
DUMP_BATCH = int(1e6)

### Validation set settings (set to None if no validation required)
# Filepath to validation set (must have 'tid', 'attribute', 'correct_val' columns)
VALIDATE_FPATH = '../testdata/hospital_clean.csv'
# Filepath prefix to dump validation results to
VALIDATE_PREFIX = '{}_validation'.format(DUMP_PREFIX)
# Run validation every nth epoch
VALIDATE_EPOCH = 5

"""
Run Holoclean.
"""

# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=10,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True,

    domain_df_prefix=DOMAIN_PREFIX,

    embed_estimator_lambda=LAMBDA,
    embed_estimator_embed_size=EMBED_SIZE,
    embed_estimator_num_epochs=EPOCHS,
    embed_estimator_batch_size=BATCH,
    embed_estimator_train_attrs=TRAIN_ATTRS,
    embed_estimator_train_only_clean=TRAIN_ONLY_CLEAN,
    embed_estimator_knn=KNN,

    embed_estimator_memoize=MEMOIZE,

    embed_estimator_dump_batch=DUMP_BATCH,
    embed_estimator_dump_prefix=DUMP_PREFIX,

    embed_estimator_validate_fpath=VALIDATE_FPATH,
    embed_estimator_validate_prefix=VALIDATE_PREFIX,
    embed_estimator_validate_epoch=VALIDATE_EPOCH,
).session

# 2. Load training data and denial constraints.
hc.load_data(DS_NAME, RAW_FPATH)
hc.load_dcs(DC_FPATH)
hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [ViolationDetector(), NullDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.generate_domain(store_to_db=False)

# 5. Run weak label embedding model.
hc.weak_label()
