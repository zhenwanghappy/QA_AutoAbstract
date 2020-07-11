# question and answer summary and reasoning
(一)数据集：
训练集（82943条记录）建立模型，基于汽车品牌、车系、问题内容与问答对话的文本，输出建议报告文本。
测试集（20000条记录）使用训练好的模型，输出建议报告的结果文件。

（二）过程：
1. 使用jieba分词，过滤低频词，保留前30000个词构建vocab(id2word)，没有采用高频词的亚采样
2. 使用word2vec训练词向量，使用负采样，训练了5个epoch
3. 取出word2vec中的词向量，与vocab构建embbedding矩阵
4. 构建seq2seq(注意力机制)模型作为base_line，encoder端使用双向gru，只在decoder端使用mask
5. 构建pgn+coverage模型，并在encoder端和decoder使用mask