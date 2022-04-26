# recommend
基于[deepctr](https://github.com/shenweichen/DeepCTR)和[deepmatch](https://github.com/shenweichen/DeepMatch)实现一些推荐相关的算法，包括召回和排序等。<br>
拷贝deepctr和deepmatch源码集成到该仓库，对源码细节加入更多的详细注释。
* 召回：常见的召回模型输出item user向量，通过faiss等vector similarity index db查找user近邻items集合作为召回集合。
* 排序：常见的排序模型的输出可以对召回的items进行排序，例如ctr点击率概率，回归分值等。

## 目录
[recall/fm](https://github.com/zhaocc1106/recommend/blob/master/recall/fm/fm.py): 实践fm网络作为召回网络。<br>
[recall/youtube_dnn](https://github.com/zhaocc1106/ctr/tree/master/recall/youtube_dnn): 实践youtube dnn网络作为召回网络。<br>
[recall/dssm](https://github.com/zhaocc1106/ctr/tree/master/recall/dssm): 实践dssm网络作为召回网络。<br>
[recall/sdm](https://github.com/zhaocc1106/ctr/tree/master/recall/sdm): 实践sdm网络作为召回网络。<br>
[rank/deepfm](https://github.com/zhaocc1106/ctr/tree/master/rank/deepfm): 实践deepfm模型。<br>
[rank/mmoe](https://github.com/zhaocc1106/ctr/tree/master/rank/mmoe): 实践mmoe多任务模型网络。
