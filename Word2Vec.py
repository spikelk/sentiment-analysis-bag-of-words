import pandas as pd
import numpy as np
import re
import nltk.data
import logging
import time
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# 读取数据
train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" \
      % (train["review"].size, test["review"].size, unlabeled_train["review"].size ))

# 减少数据量以加快运行速度
sample_size = 25000  # 仅使用前1000条数据进行测试
train = train[:sample_size]
test = test[:sample_size]
unlabeled_train = unlabeled_train[:sample_size]
print("Using %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews for testing\n" \
      % (train["review"].size, test["review"].size, unlabeled_train["review"].size ))

# 数据预处理，返回词列表
def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review, features="lxml").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

# 简单的句子分割方法，使用句号、问号、感叹号等作为分隔符
def review_to_sentences( review, remove_stopwords=False ):
    # 使用正则表达式分割句子
    raw_sentences = re.split(r'[.!?]+', review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences

# 存放所有评论的句子列表
sentences = []
# 解析有标签训练数据集的评论数据
for review in train["review"]:
    sentences += review_to_sentences(review)
# 解析无标签训练数据集的评论数据
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review)
# 查看数据
print(len(sentences)) # 795538个句子
print(sentences[0])

# 导入内置日志记录模块并对其进行配置，以便Word2Vec创建良好的输出消息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 设置参数
num_features = 100   # 按词频取频率最高的100个词构成词汇表
min_word_count = 5   # 词频小于5的词则弃用（适应小数据集）
num_workers = 4      # 线程数
context = 5          # 上下文窗口大小（适应小数据集）
downsampling = 1e-3  # 设置频繁单词降低采样

# Word2Vec不需要标签
# Word2Vec模型由词汇表中每个单词的特征向量组成，存储在numpy一个名为“syn0” 的数组中
# 每个词存在一个one-hot向量，向量的维度是300。如果该词在词汇表中出现过，则向量中词汇表中对应的位置为1，其他位置全为0。如果在词汇表中不出现，则向量为全0
# Word2Vec可以将One-Hot Encoder转化为低维度的连续值，也就是稠密向量，并且其中意思相近的词将被映射到向量空间中相近的位置。
# 训练、保存模型
model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count = min_word_count, window = context, sample = downsampling)
model_name = "100features_5minwords_5context"  # 更新模型名称以反映新参数
model.save(model_name)

# 探索模型结果。不匹配与最大相似
print(model.wv.doesnt_match("man woman child kitchen".split()))
print(model.wv.doesnt_match("france england germany berlin".split()))
print(model.wv.doesnt_match("paris berlin london austria".split()))
print(model.wv.most_similar("man"))
print(model.wv.most_similar("queen"))
print(model.wv.most_similar("awful"))

# 加载模型
model = word2vec.Word2Vec.load("100features_5minwords_5context")

# numpy数组大小为（16490，300），代表词频≥40的单词数目及每个单词对应的特征数
type(model.wv.vectors)
model.wv.vectors.shape

# 访问单个单词向量，返回一个1x300的numpy数组
model.wv.vectors[0]
model.wv["want"]

# -----------------------------  方法一：向量平均  ---------------------------------

# 将一个句子中所有符合词频≥40的单词，所对应的词向量累加起来，再除以进行了词向量转换的所有单词的个数
# 方法最终将一个句子转成特征向量的形式
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0
    # 获取词汇
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model.wv[word])
    # 避免除以零
    if nwords > 0:
        featureVec = np.divide(featureVec,nwords)
    return featureVec

# 将每段评论转化为基于词向量的特征向量
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs

# 处理训练数据集，将每段文本数据切成词后转成特征向量
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

# 处理测试数据集，将每段文本数据切成词后转成特征向量
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

# 模型训练、预测、生成结果
forest = RandomForestClassifier( n_estimators = 100 )
forest = forest.fit( trainDataVecs, train["sentiment"] )
result = forest.predict( testDataVecs )
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

# -----------------------------  方法二：聚类  ---------------------------------

start = time.time()
# Word2Vec创建语义相关单词的集群。也可以通过使用聚类算法K-Means，利用集群中单词的相似性进行分类。
# 测试表明每个群集平均只有5个单词左右的小群集比具有多个词的大群集产生更好的结果
word_vectors = model.wv.vectors
num_clusters = word_vectors.shape[0] // 5
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )
# 计算运行时间
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

# 将单词与类别压缩成字典
word_centroid_map = dict(zip( model.wv.index_to_key, idx ))

# 打印出0到9簇的单词
for cluster in range(0,10):
    print("\nCluster %d" % cluster)
    words = []
    for i in range(0,len(word_centroid_map.values())):
        if( list(word_centroid_map.values())[i] == cluster ):
            words.append(list(word_centroid_map.keys())[i])
    print(words)

# 创建质心袋。质心袋是一个列表，列表长度为分类个数，元素值为每个分类的单词数
def create_bag_of_centroids( wordlist, word_centroid_map ):
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

# 训练集创建质心袋
train_centroids = np.zeros( (train["review"].size, num_clusters), dtype="float32" )
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# 测试集创建质心袋
test_centroids = np.zeros(( test["review"].size, num_clusters), dtype="float32" )
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# 模型训练、预测、生成结果
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )