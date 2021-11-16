import os
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import MMOE

DATA_PATH = '../../datas/census-income/census-income.sample'
MODEL_PATH = '/tmp/mmoe/'
CKPT_PATH = os.path.join(MODEL_PATH, 'checkpoint.h5')
SAVER_PATH = os.path.join(MODEL_PATH, 'model_saver.h5')


def gen_feature():
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    data = pd.read_csv(DATA_PATH, header=None, names=column_names)

    data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})  # 收入分成两类，低于50000，高于50000
    data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)  # 婚姻状态分成是否结过婚两类
    data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)

    # 分割sparse features和dense features
    columns = data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if
                      col not in sparse_features and col not in ['label_income', 'label_marital']]

    # 填充nan
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    # dense tensor归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_features] \
                             + [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    # 分割训练和测试集
    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    return linear_feature_columns, dnn_feature_columns, train, test


def def_model_and_train(linear_feature_columns, dnn_feature_columns, train, test):
    # 构造模型输入集合
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 定义模型
    model = MMOE(dnn_feature_columns,
                 tower_dnn_hidden_units=[64],
                 task_types=['binary', 'binary'],  # 两个二值分类目标
                 task_names=['label_income', 'label_marital'])
    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )

    # train and evaluate
    callbacks = [
        # Model saver callback.
        tf.keras.callbacks.ModelCheckpoint(filepath=CKPT_PATH),
        # Tensorboard callback.
        tf.keras.callbacks.TensorBoard(log_dir=MODEL_PATH)
    ]

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if os.path.exists(CKPT_PATH):
        model.load_weights(filepath=CKPT_PATH)

    history = model.fit(train_model_input, [train['label_income'].values, train['label_marital'].values],
                        batch_size=256, epochs=100, verbose=2, validation_split=0.2, callbacks=callbacks)
    pred_ans = model.predict(test_model_input, batch_size=256)

    print("test income AUC", round(roc_auc_score(test['label_income'], pred_ans[0]), 4))
    print("test marital AUC", round(roc_auc_score(test['label_marital'], pred_ans[1]), 4))


if __name__ == '__main__':
    linear_feature_columns, dnn_feature_columns, train, test = gen_feature()
    def_model_and_train(linear_feature_columns, dnn_feature_columns, train, test)
