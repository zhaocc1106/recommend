import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import faiss
from tqdm import tqdm
from datas.movielens_data.preprocess import gen_data_set_sdm, gen_model_input_sdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

from deepmatch.models import SDM
from deepmatch.utils import sampledsoftmaxloss
from deepmatch.utils import recall_N
from deepctr.feature_column import SparseFeat, VarLenSparseFeat

MODEL_PATH = "/tmp/sdm/"
CKPT_PATH = os.path.join(MODEL_PATH, "checkpoint.h5")
SAVER_PATH = os.path.join(MODEL_PATH, "model_saver.h5")
DATA_PATH = '../../datas/movielens_data/'
SEQ_LEN_SHORT = 5  # 短期兴趣长度
SEQ_LEN_PREFER = 50  # 长期兴趣长度
EMBEDDING_DIM = 32  # embedding特征的长度


def gen_feature():
    # 准备原始数据
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(DATA_PATH + 'ml-1m/users.dat', sep='::', header=None, names=unames)
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(DATA_PATH + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(DATA_PATH + 'ml-1m/movies.dat', sep='::', header=None, names=mnames)
    data = pd.merge(pd.merge(ratings, movies), user)  # .iloc[:10000]

    # 对稀疏特征做label enconding
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres"]
    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip", "genres"]].drop_duplicates('user_id')
    user_profile.set_index("user_id", inplace=True)
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    # 生成短期兴趣序列和长期兴趣序列数据
    train_set, test_set = gen_data_set_sdm(data, seq_short_len=SEQ_LEN_SHORT, seq_prefer_len=SEQ_LEN_PREFER)

    # 生成模型输入和目标集合
    train_model_input, train_label = gen_model_input_sdm(train_set, user_profile, SEQ_LEN_SHORT, SEQ_LEN_PREFER)
    test_model_input, test_label = gen_model_input_sdm(test_set, user_profile, SEQ_LEN_SHORT, SEQ_LEN_PREFER)
    return train_model_input, train_label, test_model_input, test_label, feature_max_idx, item_profile, test_set


def def_model_and_train(train_model_input, train_label, feature_max_idx):
    # 定义用户features，指定特征类型
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('short_movie_id', feature_max_idx['movie_id'], EMBEDDING_DIM,
                                                        embedding_name="movie_id"), SEQ_LEN_SHORT, 'mean',
                                             'short_sess_length'),
                            VarLenSparseFeat(SparseFeat('prefer_movie_id', feature_max_idx['movie_id'], EMBEDDING_DIM,
                                                        embedding_name="movie_id"), SEQ_LEN_PREFER, 'mean',
                                             'prefer_sess_length'),
                            VarLenSparseFeat(SparseFeat('short_genres', feature_max_idx['genres'], EMBEDDING_DIM,
                                                        embedding_name="genres"), SEQ_LEN_SHORT, 'mean',
                                             'short_sess_length'),
                            VarLenSparseFeat(SparseFeat('prefer_genres', feature_max_idx['genres'], EMBEDDING_DIM,
                                                        embedding_name="genres"), SEQ_LEN_PREFER, 'mean',
                                             'prefer_sess_length'),
                            ]

    # 定义视频features，指定特征类型
    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], EMBEDDING_DIM)]

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    # units must be equal to item embedding dim!
    model = SDM(user_feature_columns, item_feature_columns, history_feature_list=['movie_id', 'genres'],
                units=EMBEDDING_DIM, num_sampled=100, )
    optimizer = optimizers.Adam(lr=0.001, clipnorm=5.0)
    model.compile(optimizer=optimizer, loss=sampledsoftmaxloss)  # 模型内部指定了采样损失，这里loss输出模型输出值

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

    history = model.fit(train_model_input, train_label,  # train_label没用到
                        batch_size=512, epochs=10, verbose=1, validation_split=0.0, callbacks=callbacks)
    model.save(filepath=SAVER_PATH)

    print('user_embed.shape: {}'.format(model.user_embedding.shape))
    print('item_embed.shape: {}'.format(model.item_embedding.shape))
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)  # 抽取出输出用户embed特征的模型
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)  # 抽取出输出视频embed特征的模型

    return user_embedding_model, item_embedding_model


def evaluate(user_embedding_model, item_embedding_model, user_input, item_profile, test_set):
    # {用户id: 用于测试的观看过的视频id列表，上边生成测试数据时只把视频历史最后一个视频当做这里的测试集的视频列表}
    test_true_label = {line[0]: [line[3]] for line in test_set}

    item_input = {"movie_id": item_profile['movie_id'].values, }
    user_embeds = user_embedding_model.predict(user_input, batch_size=2 ** 12)  # 生成所有用户的embed特征
    item_embeds = item_embedding_model.predict(item_input, batch_size=2 ** 12)  # 生成所有视频的embed特征
    print('user_embeds.shape: {}'.format(user_embeds.shape))
    print('item_embeds.shape: {}'.format(item_embeds.shape))

    # Create faiss index database.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # 建立内积距离作为相似度计算的向量索引数据库
    print('index.is_trained: {}'.format(index.is_trained))
    index.add(item_embeds)
    print('index.ntotal: {}'.format(index.ntotal))
    distance, idx = index.search(np.ascontiguousarray(user_embeds), 50)  # 查询与用户特征最近邻的视频特征
    print('distance.shape: {}'.format(distance.shape))  # 视频与用户的embed特征距离
    print('idx.shape: {}'.format(idx.shape))  # 视频id
    s = []
    hit = 0
    for i, uid in tqdm(enumerate(user_input['user_id'])):
        pred = [item_profile['movie_id'].values[x] for x in idx[i]]
        filter_item = None
        recall_score = recall_N(test_true_label[uid], pred, N=50)  # 真实的视频label存在于预估中的前N个label的比率，即召回率
        s.append(recall_score)
        if test_true_label[uid] in pred:  # 命中了用户的可能感兴趣的视频，对应测试集中用户看过的视频
            hit += 1

    print("")
    print("recall", np.mean(s))
    print("hit rate", hit / len(user_input['user_id']))


if __name__ == "__main__":
    train_model_input, train_label, test_model_input, test_label, feature_max_idx, item_profile, test_set \
        = gen_feature()
    user_embedding_model, item_embedding_model = def_model_and_train(train_model_input, train_label, feature_max_idx)
    evaluate(user_embedding_model, item_embedding_model, test_model_input, item_profile, test_set)
