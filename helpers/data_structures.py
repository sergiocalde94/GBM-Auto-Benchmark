from collections import namedtuple


Prediction = namedtuple(
    'Prediction',
    'y_score_train y_score y_pred_train y_pred'
)

Metrics = namedtuple(
    'Metrics',
    'auc_train auc_test acc_train acc_test pr_auc_train pr_auc_test'
)
