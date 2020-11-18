import pandas as pd
import statsmodels.api as sm
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


df= pd.read_csv("term-deposit-marketing-2020.csv")
df.replace(('yes', 'no'), (1, 0), inplace=True)
df.describe().T
df2 = df.sort_values("balance", ascending=False).iloc[200:]
df2.describe().T
df3 = df2.sort_values("duration", ascending=False).iloc[200:]
df3.describe().T
sb.countplot(df3["y"])

df_job = df3["job"]
df_job = pd.get_dummies(df_job)
display(df_job.head())

df_marital = df3["marital"]
df_marital = pd.get_dummies(df_marital)
display(df_marital.head())

df_education = df3["education"]
df_education = pd.get_dummies(df_education)
display(df_education.head())

df_contact = df3["contact"]
df_contact = pd.get_dummies(df_contact)
display(df_contact.head())

df_month = df3["month"]
df_month = pd.get_dummies(df_month)
display(df_month.head())

df3.drop(["job", "marital", "education", "contact", "month"], axis=1, inplace=True)
df3 = pd.concat([df3, df_contact, df_education, df_job, df_marital, df_month], axis=1)

y = df3["y"]
x = df3.drop(["y"], axis =1)

y.head()
x.head()

loj_model = LogisticRegression(solver = "liblinear").fit(x,y)
loj_model.intercept_
loj_model.coef_
##bu değerlere bakılarak default, balance, day, duration ve bazı ayların pozitif kat
##sayısı yüksektir. housing değişkeninin negatif etkisi çok yüksektir ve meslek
##gruplarından housemaidin en yüksek negatif etkiye sahip olduğu görülmektedir.
##bu bilgilere göre uygun müşterilere ulaşılabilir. 


y_pred = loj_model.predict(x)
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))

loj_model.predict_proba(x)[0:10]



x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.30,
                                                    random_state=42)

loj_model = LogisticRegression(solver = "liblinear").fit(x_train,y_train)

y_pred = loj_model.predict(x_test)
print(accuracy_score(y_test, y_pred))
cross_val_score(loj_model, x_test, y_test, cv=5)








