import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# 读取数据
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3 )
train.shape
test.shape
train.columns.values
print(train["review"][0])

# 去除标签
example1 = BeautifulSoup(train["review"][0], features="lxml")
print(train["review"][0])
print(example1.get_text())

# 将非字母字符替换为空格
letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text() )
print(letters_only)

# 转换为小写并切分
lower_case = letters_only.lower()
words = lower_case.split()

# 停用词
print(stopwords.words("english"))

# 去除停用词
words = [w for w in words if not w in stopwords.words("english")]
print(words)

# 将上述预处理步骤封装成方法，返回词列表
def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

# 测试第一条评论
clean_review = review_to_words( train["review"][0] )
print(clean_review)

# 处理所有评论数据
num_reviews = train["review"].size
clean_train_reviews = []
for i in range(num_reviews):
    if((i+1)%1000 == 0):
        print("Review %d of %d\n" %(i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

num_reviews = len(test["review"])
clean_test_reviews = []
for i in range(num_reviews):
    if((i+1) % 1000 == 0):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_test_reviews.append(review_to_words(test["review"][i]))

# CountVectorizer会将文本中的词语转换为词频矩阵。即模型从所有文档中学习词汇表，然后在每个文档中统计每个词出现的次数
# 创建词袋
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# 为便于理解，以下为假设结果
# array([[1, 0, 2, ..., 0, 0, 0],
#        [0, 5, 0, ..., 7, 0, 0],
#        ...,
#        [0, 0, 3, ..., 4, 0, 0],
#        [4, 6, 0, ..., 0, 8, 0]], dtype=int64)

# 查看词袋形状、特征名
print(train_data_features.shape)  # (25000,5000)
vocab = vectorizer.get_feature_names_out()
print(vocab)

# 统计词在所有评论中出现次数
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)

# 随机森林模型训练、预测、生成结果文件
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, train["sentiment"])
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )