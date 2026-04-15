import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 将文本数据处理成词列表
def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review, features="lxml").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

# 读取数据
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3 )
y = train["sentiment"]

# 处理所有文本数据（使用小部分数据进行测试）
sample_size = 25000  # 仅使用前1000条数据进行测试
traindata = []
for i in range(min(sample_size, len(train["review"]))):
    traindata.append(" ".join(review_to_wordlist(train["review"][i], True)))
testdata = []
for i in range(min(sample_size, len(test["review"]))):
    testdata.append(" ".join(review_to_wordlist(test["review"][i], True)))

# TfidfVectorizer将原始文档的集合转化为tf-idf特性的矩阵，计算特征的计数权重。
# TfidfVectorizer类将CountVectorizer和TfidfTransformer类封装在一起。
# 模型训练、转换
tfv = TfidfVectorizer(min_df=5,
                      max_features=5000,  # 限制特征数量
                      strip_accents='unicode',
                      analyzer='word',
                      token_pattern=r'\w{1,}',
                      ngram_range=(1, 2),  # 减少ngram范围
                      use_idf=True,
                      smooth_idf=True,
                      sublinear_tf=True,
                      stop_words = 'english')
X_train = tfv.fit_transform(traindata)
X_test = tfv.transform(testdata)
X_train.toarray()
# 为便于理解，以下为假设结果
# array([[0.85, 0., 0.65, ..., 0., 0., 0.],
#        [0., 0.74, 0., ..., 0.24, 0., 0.],
#        ...,
#        [0., 0., 0.53, ..., 0., 0.22, 0.],
#        [0.89, 0., 0., ..., 0., 0., 0.36]])

# 逻辑回归模型训练、预测、生成结果文件
model = LogisticRegression(penalty='l2',
                           dual=False,
                           tol=0.0001,
                           C=1,
                           fit_intercept=True,
                           intercept_scaling=1.0,
                           class_weight=None,
                           random_state=None)
# 只使用样本数据进行训练
y_sample = y[:sample_size]
model.fit(X_train, y_sample)
print('5 Fold CV Score: %f'%np.mean(cross_val_score(model, X_train, y_sample, cv=5, scoring='roc_auc')))
result = model.predict(X_test)
# 只使用与result对应的测试数据的id
output = pd.DataFrame( data={"id":test["id"][:sample_size], "sentiment":result} )
output.to_csv('Tfidf.csv', index=False, quoting=3)