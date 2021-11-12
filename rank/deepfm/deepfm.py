import os
import tensorflow as tf
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

MODEL_PATH = "/tmp/deepfm/"
CKPT_PATH = os.path.join(MODEL_PATH, "checkpoint.h5")
SAVER_PATH = os.path.join(MODEL_PATH, "model_saver.h5")


def gen_feature():
    data = pd.read_csv('../../datas/criteo_sample/criteo_sample.csv')
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )  # 填充nan值
    data[dense_features] = data[dense_features].fillna(0, )  # 填充nan值

    # 1.do simple Transformation for dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # print(data[dense_features])

    # 2.set hashing space for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=1000, embedding_dim=4, use_hash=True, dtype='string')
                              # since the input is string
                              for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
    # deepfm的FM和DNN两部分共享输入特征
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    # 分割训练和测试集
    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    return linear_feature_columns, dnn_feature_columns, train, test


def def_model_and_train(linear_feature_columns, dnn_feature_columns, train, test):
    # 生成模型输入的dict
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary')
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
    model.summary()
    # 定义model fit过程的回调
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
    history = model.fit(train_model_input, train['label'].values, batch_size=256, epochs=500, verbose=2,
                        validation_split=0.2, callbacks=callbacks)
    model.save(filepath=SAVER_PATH)

    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test['label'].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test['label'].values, pred_ans), 4))


if __name__ == '__main__':
    linear_feature_columns, dnn_feature_columns, train, test = gen_feature()
    def_model_and_train(linear_feature_columns, dnn_feature_columns, train, test)
