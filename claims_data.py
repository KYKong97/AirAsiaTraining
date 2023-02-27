# -*- coding: utf-8 -*-
"""Claims Data

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1luEEG7SlUcri-jokqJsojeHSQTl1WiKA
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""# EDA"""

df = pd.read_excel('Sample Data.xlsx')
print(df.shape)
df.head()

df.infer_objects()

for column in df.columns:
  type_col = pd.api.types.infer_dtype(df[column])
  if 'mixed' in type_col:
    print(column)
    df[column] = df[column].astype(str)

import sweetviz as sv

my_report = sv.analyze(df.infer_objects())
my_report.show_html( filepath='SWEETVIZ_REPORT.html')

df.info()

"""## Missing Value or outlier"""

print(df.drop_duplicates().shape)
print(df.shape)

df = df.drop_duplicates()

"""# Drop Unnecessary column"""

drop_column = ['Merimen CaseID','Claim No','RepCoName','RepCoName','RecPartQty','RecPartNo','PartNo','SourceFile']
df = df.drop(drop_column,axis=1)

df.describe()

"""# Drop betterment as all zero"""

df = df.drop('RecPartsBetterment',axis=1)

"""# Categorical Data"""

df.describe(include = 'object')

"""## Drop RecCondition as too many unique"""

df = df.drop('RecCondition',axis=1)

df = df.drop('RecPartsInsFPInd',axis=1)

"""# Clustering"""

df.head()

df_continous = df[['RecPartsRepAmount','RecPartsAdjAmount','RecPartsInsAmount']]
df_continous.fillna(0,inplace=True)
df_continous['RecPartsRepAmount'] = df_continous['RecPartsRepAmount'].astype(float)
df_continous['RecPartsAdjAmount'] = df_continous['RecPartsAdjAmount'].astype(float)
df_continous['RecPartsInsAmount'] = df_continous['RecPartsInsAmount'].astype(float)

from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=2, random_state=0).fit(df_continous)

unique, counts = np.unique(kmeans_model.labels_, return_counts=True)
print(unique)
print(counts)
print(kmeans_model.cluster_centers_)

df_continous['y_pred'] = kmeans_model.labels_

fig=plt.figure()
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.scatter(
    df_continous[df_continous['y_pred']==0]['RecPartsRepAmount'],
    df_continous[df_continous['y_pred']==0]['RecPartsAdjAmount'],
    df_continous[df_continous['y_pred']==0]['RecPartsInsAmount'],
    color='red',s=1,alpha=0.1)

ax.scatter(
    df_continous[df_continous['y_pred']==1]['RecPartsRepAmount'],
    df_continous[df_continous['y_pred']==1]['RecPartsAdjAmount'],
    df_continous[df_continous['y_pred']==1]['RecPartsInsAmount'],
    color='green',s=1,alpha=0.1)
ax.scatter(
    kmeans_model.cluster_centers_[0][0],
    kmeans_model.cluster_centers_[0][1],
    kmeans_model.cluster_centers_[0][2],
    c="black",s=150,label="Centers",alpha=1,marker='x'
)
ax.scatter(
    kmeans_model.cluster_centers_[1][0],
    kmeans_model.cluster_centers_[1][1],
    kmeans_model.cluster_centers_[1][2],
    c="black",s=150,label="Centers",alpha=1,marker='x'
)
ax.set_xlabel('RecPartsRepAmount')
ax.set_ylabel('RecPartsAdjAmount')
ax.set_zlabel('RecPartsInsAmount')
plt.show()

df['KMeansLabel'] = kmeans_model.labels_

print(
    df[
    df['KMeansLabel']==0
]['RecPartsInsAmount'].describe()
)
df[
    df['KMeansLabel']==0
]['RecPartsInsAmount'].hist()

print(
    df[
    df['KMeansLabel']==1
]['RecPartsInsAmount'].describe()
)

df[
    df['KMeansLabel']==1
]['RecPartsInsAmount'].hist()

"""# Get y """

y = df['RecPartsInsAmount'].tolist()

"""# Remove outlier"""

c = 'RecPartsInsAmount'
q1 = df[c].quantile(0.25)
q3 = df[c].quantile(0.75)
iqr = q3-q1
lower_limit = q1-(1.5*iqr)
upper_limit = q3+(1.5*iqr)

df_outlier = df[
    (df[c]<lower_limit)|
    (df[c]>upper_limit)
  
]
df_outlier.shape

df_copy = df.copy()
# for c in df.columns:
#   if df[c].dtype!='O':
#     if c=='RecPartsInsAmount':
#       continue
#     q1 = df[c].quantile(0.25)
#     q3 = df[c].quantile(0.75)
#     iqr = q3-q1
#     lower_limit = q1-(1.5*iqr)
#     upper_limit = q3+(1.5*iqr)
#     print("{} Upper:{} Lower:{}".format(c,upper_limit,lower_limit))
#     df_copy = df_copy[
#         (df_copy[c]>lower_limit)&
#         (df_copy[c]<upper_limit)&
#         (~df_copy[c].isnull())
#     ]

df_copy = df_copy[
    ~df_copy['RecPartsInsAmount'].isnull()
]

"""# EDA

# Exploration of Continuous Variable
"""

import seaborn as sns

sns.distplot(df_copy['RecPartsInsAmount'])
sns.distplot(df_copy['RecPartsAdjAmount'])
sns.distplot(df_copy['RecPartsRepAmount'])
plt.show()

df_copy.to_csv('download.csv',index=False)

import plotly.graph_objects as go

import numpy as np

x0 = df_copy['RecPartsInsAmount']
# Add 1 to shift the mean of the Gaussian distribution
x1 = df_copy['RecPartsAdjAmount']

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0,nbinsx=20))
fig.add_trace(go.Histogram(x=x1,nbinsx=20))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.5)
fig.show()

for c in df_copy.columns:
  if df_copy[c].dtype!='O':
    if c=='RecPartsInsAmount':
      continue
    
    plt.hist(df_copy[c])
    plt.title(c)
    plt.show()

plt.hist(df_copy['RecPartsInsAmount'])

"""# Correlation"""

df_copy.corr()

df.head()

df.groupby(
    "RecPartParticulars"
).size().reset_index(
    name="count"
).plot.bar(x='RecPartParticulars', y='count', rot=90)

df.groupby(
    "PriceSource"
).size().reset_index(
    name="count"
).plot.bar(x='PriceSource', y='count', rot=90)

df.groupby(
    "SalvageInd"
).size().reset_index(
    name="count"
).plot.bar(x='SalvageInd', y='count', rot=90)

# RecPartsDisc
df['RecPartsDisc'].hist()

df.plot.scatter(x='RecPartsRepAmount',y='RecPartsAdjAmount')

df.plot.scatter(x='RecPartsAdjAmount',y='RecPartsInsAmount')

"""## Nan In Adjuster"""

df[
    df['RecPartsAdjAmount'].isnull()
]['RecPartsInsAmount'].hist()

df['isAdjAmtNull'] = np.where(
    df['RecPartsAdjAmount'].isnull(),
    1,0
)
plt.hist(df[df['isAdjAmtNull']==0]['RecPartsInsAmount'],label='Adj Not Null')
plt.hist(df[df['isAdjAmtNull']==1]['RecPartsInsAmount'],label='Adj Null')
plt.legend()

df.groupby(
    "isAdjAmtNull"
).size().reset_index()

df.groupby(
    "ClaimType"
).agg(
    mean=('RecPartsInsAmount','mean'),
    median=('RecPartsInsAmount','median'),
).reset_index()



"""# Train Test Split"""

from sklearn.model_selection import train_test_split
df_copy = df_copy[
    ~df_copy['RecPartsAdjAmount'].isnull()
]
X_train, X_test, y_train, y_test = train_test_split(df_copy[['RecPartsAdjAmount','KMeansLabel']], df_copy['RecPartsInsAmount'], test_size=0.33, random_state=42)

"""# Simple Linear Regression"""

import numpy as np
# X_train = np.expand_dims(X_train,1)
# X_test = np.expand_dims(X_test,1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_pred = reg.predict(X_test)

r2_score(y_pred,y_test)

y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)

y_pred = reg.predict(X_train)
mean_squared_error(y_train, y_pred,squared=False)

"""# Multiple variables"""

df_copy.head()

df_copy = pd.get_dummies(df_copy,drop_first=True)
df_copy.head()

X_train, X_test, y_train, y_test = train_test_split(df_copy.drop("RecPartsInsAmount",axis=1), df_copy['RecPartsInsAmount'], test_size=0.33, random_state=42)
# X_train = np.expand_dims(X_train,1)
# X_test = np.expand_dims(X_test,1)

X_train.head()

reg = LinearRegression().fit(X_train, y_train)

from sklearn.metrics import r2_score
y_pred = reg.predict(X_test)
r2_score(y_pred,y_test)

coefficient = reg.coef_

mapping = {}
for col, coeff in zip(X_train.columns,coefficient.tolist()):
  mapping[col] = coeff
mapping
