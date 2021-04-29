import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
style.use("ggplot")


df = pd.read_excel('titanic.xls')


df


df.drop(columns=["name", "cabin", "boat", "home.dest", "body", "ticket"], inplace=True)


df = df.dropna()


embarked = set(df["embarked"])
embarked = {data:index for index, data in enumerate(embarked)}


for index, data in zip(df.index, df["embarked"]):
    df.at[index, "embarked"] = embarked[data]


gender = {"male": 1, "female":0}


for index, data in zip(df.index, df["sex"]):
    df.at[index, "sex"] = gender[data]


df


x = df.drop(columns=["survived"])
x = np.array(x)
x = preprocessing.scale(x)


y = np.array(df["survived"])


clf = KMeans()
clf.fit(x)


correct = 0

for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(x))


pred = clf.predict(x)
print(pred)



