import random
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):  # 遍历每个用户
        pos_list = hist['movie_id'].tolist()  # 观看历史
        rating_list = hist['rating'].tolist()  # 评分历史

        if negsample > 0:
            # 如果需要负采样，则从没看过的视频中随机采样，采样的比率由negsample决定
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):  # 遍历每个视频构造训练和测试样本
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                # 当前用户的正样本，观看历史列表中每个视频可以作为前面视频观看历史的训练学习目标
                train_set.append((reviewerID,
                                  hist[::-1],  # 该训练样本的输入是当前视频之前的视频列表
                                  pos_list[i],  # 该训练样本的待分类的视频id，分类成用户观看过label为1，没看过分类为0
                                  1,  # 1代表正样本，即用户观看过的
                                  len(hist[::-1]),  # 该样本的输入历史长度
                                  rating_list[i]))  # 评级没用到
                for negi in range(negsample):
                    # 当前用户的负样本
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0, len(hist[::-1])))
            else:
                # 当前用户的测试样本，把历史列表最后一个视频和之前的观看历史作为该用户的测试数据
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    # print(len(train_set[0]), len(test_set[0]))
    return train_set, test_set


def gen_data_set_sdm(data, seq_short_len=5, seq_prefer_len=50):
    data.sort_values("timestamp", inplace=True)
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        genres_list = hist['genres'].tolist()
        rating_list = hist['rating'].tolist()
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            if i <= seq_short_len and i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], [0] * seq_prefer_len, pos_list[i], 1, len(hist[::-1]), 0,
                                  rating_list[i], genres_hist[::-1], [0] * seq_prefer_len))
            elif i != len(pos_list) - 1:
                train_set.append(
                    (reviewerID, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1, seq_short_len,
                     len(hist[::-1]) - seq_short_len, rating_list[i], genres_hist[::-1][:seq_short_len],
                     genres_hist[::-1][seq_short_len:]))
            elif i <= seq_short_len and i == len(pos_list) - 1:
                test_set.append((reviewerID, hist[::-1], [0] * seq_prefer_len, pos_list[i], 1, len(hist[::-1]), 0,
                                 rating_list[i], genres_hist[::-1], [0] * seq_prefer_len))
            else:
                test_set.append(
                    (reviewerID, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1, seq_short_len,
                     len(hist[::-1]) - seq_short_len, rating_list[i], genres_hist[::-1][:seq_short_len],
                     genres_hist[::-1][seq_short_len:]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)

    # 构造模型的输入特征，每个输入特征的第一纬代表batch size，都是一样的，后续维度则是该特征的实际shape
    train_model_input = {
        "user_id": train_uid,  # 所有样本的用户id
        "movie_id": train_iid,  # 所有样本的待分类的视频id，分类成用户观看过label为1，没看过分类为0
        "hist_movie_id": train_seq_pad,  # 所有样本的用户观看历史列表
        "hist_len": train_hist_len  # 所有样本的用户观看历史列表长度
    }
    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    # 模型的输出，即待分类的视频的分类标签，即训练目标label
    train_label = np.array([line[3] for line in train_set])

    return train_model_input, train_label


def gen_model_input_sdm(train_set, user_profile, seq_short_len, seq_prefer_len):
    train_uid = np.array([line[0] for line in train_set])
    short_train_seq = [line[1] for line in train_set]
    prefer_train_seq = [line[2] for line in train_set]
    train_iid = np.array([line[3] for line in train_set])
    train_label = np.array([line[4] for line in train_set])
    train_short_len = np.array([line[5] for line in train_set])
    train_prefer_len = np.array([line[6] for line in train_set])
    short_train_seq_genres = np.array([line[8] for line in train_set])
    prefer_train_seq_genres = np.array([line[9] for line in train_set])

    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_len, padding='post', truncating='post',
                                         value=0)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_len, padding='post', truncating='post',
                                          value=0)
    train_short_genres_pad = pad_sequences(short_train_seq_genres, maxlen=seq_short_len, padding='post',
                                           truncating='post',
                                           value=0)
    train_prefer_genres_pad = pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_len, padding='post',
                                            truncating='post',
                                            value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "short_movie_id": train_short_item_pad,
                         "prefer_movie_id": train_prefer_item_pad, "prefer_sess_length": train_prefer_len,
                         "short_sess_length":
                             train_short_len, 'short_genres': train_short_genres_pad,
                         'prefer_genres': train_prefer_genres_pad}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label
