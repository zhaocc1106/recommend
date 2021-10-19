import pandas as pd
import os
import numpy as np
import faiss
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tqdm import tqdm

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
from deepmatch.utils import recall_N

MODEL_PATH = "/tmp/youtube_dnn/"
CKPT_PATH = os.path.join(MODEL_PATH, "checkpoint.h5")
SAVER_PATH = os.path.join(MODEL_PATH, "model_saver")
DATA_PATH = "./"
SEQ_LEN = 50  # Max movie history sequence len.
NEG_SAMPLE = 1  # The rate of negative sample.
EMBEDDING_DIM = 32  # The embedding dimension len.


def gen_feature():
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(DATA_PATH + 'ml-1m/users.dat', sep='::', header=None, names=unames)
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(DATA_PATH + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(DATA_PATH + 'ml-1m/movies.dat', sep='::', header=None, names=mnames)

    data = pd.merge(pd.merge(ratings, movies), user)  # .iloc[:10000]

    # Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", ]
    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1  # 规范每个特征的label到1到n_class之间
        feature_max_idx[feature] = data[feature].max() + 1  # 记录每个特征最大标签+1，用于keras.layers.Embedding传入的input_dim参数

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    user_profile.set_index("user_id", inplace=True)
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    train_set, test_set = gen_data_set(data, NEG_SAMPLE)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)
    return train_model_input, train_label, test_model_input, test_label, feature_max_idx, item_profile, test_set


def def_model_and_train(train_model_input, train_label, feature_max_idx):
    # count #unique features for each sparse field and generate feature config for sequence feature

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], EMBEDDING_DIM,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], EMBEDDING_DIM)]

    # Define model.
    import tensorflow as tf
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=100,
                       user_dnn_hidden_units=(128, 64, EMBEDDING_DIM))
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

    print(model.user_embedding.shape)
    print(model.item_embedding.shape)

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=512, epochs=20, verbose=1, validation_split=0.0, callbacks=callbacks)

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    return user_embedding_model, item_embedding_model


def evaluate(user_embedding_model, item_embedding_model, user_input, item_profile, test_set):
    test_true_label = {line[0]: [line[2]] for line in test_set}
    print('test_true_label: {}'.format(test_true_label))

    item_input = {"movie_id": item_profile['movie_id'].values, }
    user_embeds = user_embedding_model.predict(user_input, batch_size=2 ** 12)
    item_embeds = item_embedding_model.predict(item_input, batch_size=2 ** 12)
    print('user_embeds.shape: {}'.format(user_embeds.shape))
    print('item_embeds.shape: {}'.format(item_embeds.shape))

    # Create faiss index database.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # 建立内积距离作为相似度计算的向量索引数据库
    print('index.is_trained: {}'.format(index.is_trained))
    index.add(item_embeds)
    print('index.ntotal: {}'.format(index.ntotal))
    distance, idx = index.search(np.ascontiguousarray(user_embeds), 50)
    print('distance.shape: {}'.format(distance.shape))
    print('idx.shape: {}'.format(idx.shape))
    s = []
    hit = 0
    for i, uid in tqdm(enumerate(user_input['user_id'])):
        pred = [item_profile['movie_id'].values[x] for x in idx[i]]
        filter_item = None
        recall_score = recall_N(test_true_label[uid], pred, N=50)
        s.append(recall_score)
        if test_true_label[uid] in pred:
            hit += 1

    print("")
    print("recall", np.mean(s))
    print("hit rate", hit / len(user_input['user_id']))


if __name__ == '__main__':
    train_model_input, train_label, test_model_input, test_label, feature_max_idx, item_profile, test_set \
        = gen_feature()
    user_embedding_model, item_embedding_model = def_model_and_train(train_model_input, train_label, feature_max_idx)
    evaluate(user_embedding_model, item_embedding_model, test_model_input, item_profile, test_set)
