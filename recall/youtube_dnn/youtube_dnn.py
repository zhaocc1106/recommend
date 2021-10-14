import pandas as pd
import os
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss

MODEL_PATH = "/tmp/youtube_dnn/"
CKPT_PATH = os.path.join(MODEL_PATH, "checkpoint.h5")
SAVER_PATH = os.path.join(MODEL_PATH, "model_saver")
DATA_PATH = "./"
SEQ_LEN = 50  # Max movie history sequence len.
NEG_SAMPLE = 1  # The rate of negative sample.


def gen_feature():
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(DATA_PATH + 'ml-1m/users.dat', sep='::', header=None, names=unames)
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(DATA_PATH + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(DATA_PATH + 'ml-1m/movies.dat', sep='::', header=None, names=mnames)

    data = pd.merge(pd.merge(ratings, movies), user)  # .iloc[:10000]

    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]

    # Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, NEG_SAMPLE)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)
    return train_model_input, train_label, test_model_input, test_label, feature_max_idx


def def_model_and_train(train_model_input, train_label, feature_max_idx):
    # count #unique features for each sparse field and generate feature config for sequence feature
    embedding_dim = 32

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # Define model.
    import tensorflow as tf
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=100,
                       user_dnn_hidden_units=(128, 64, embedding_dim))
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)  # "binary_crossentropy")
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
    if not os.path.exists(SAVER_PATH):
        os.makedirs(SAVER_PATH)

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=512, epochs=20, verbose=1, validation_split=0.0, callbacks=callbacks)


if __name__ == '__main__':
    train_model_input, train_label, test_model_input, test_label, feature_max_idx = gen_feature()
    def_model_and_train(train_model_input, train_label, feature_max_idx)
