import sys
sys.path.append('../')
import holoclean
from detect import NullDetector
import json


# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.3,
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
    print_fw=True
).session

# 2. Load training data and denial constraints.
hc.load_data('tobefilled', 'tobefilled')
hc.ds.knn_prefix = 'knn_50k_valid_1lambda'
hc.ds.train_attrs = json.loads('tobefilled')

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
