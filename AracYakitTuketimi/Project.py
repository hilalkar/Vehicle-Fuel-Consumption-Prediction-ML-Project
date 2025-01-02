import pandas as pd
import seaborn as sns  #veri görselleştirme için
import numpy as np
import matplotlib.pyplot as plt
from bokeh.sampledata.autompg import autompg
from scipy import stats   #skewness değeri bulmak için
from scipy.stats import norm,skew# istatistikden skewnss değeri için
from seaborn import histplot
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#XGboost
import xgboost as xgb
from streamlit import columns

#veri setini yükleme
column_name=["MPG","Cylinders","Displacement", "Horsepower", "Weight", "Accelaration", "Model year", "Origin"]
data= pd.read_csv("auto-mpg.data",names = column_name, na_values="?", comment="\t",sep=" ", skipinitialspace="True")
data=data.rename(columns= {"MPG" : "target","Accelaration":"Acceleration"})

#veri inceleme
print(data.head())
print("Data shape:" ,data.shape)
data.info()
describe=data.describe()

#eksik değerler
print(data.isna().sum())
data["Horsepower"]=data["Horsepower"].fillna(data["Horsepower"].mean())
print(data.isna().sum())
sns.histplot(data["Horsepower"], kde=True, bins=30)
plt.show()

#Korelasyon Analizi
corr_matrix=data.corr()
sns.clustermap(corr_matrix, annot=True, fmt=".2f")
plt.title("correlation btw features")
plt.show()

threshold=0.75
filtre=np.abs(corr_matrix["target"])>threshold
corr_features=corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot=True,fmt=".2f")
plt.title("correlation btw features")
plt.show()

#Kategorik değişkenler için görseller
plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())
plt.show()

plt.figure()
sns.countplot(data['Origin'])
print(data['Origin'].value_counts())
plt.show()

#box
for c in data.columns:
    plt.figure()
    sns.boxplot(y=c,data=data)
plt.show()

#aykırı değerler filtreleme
thr= 2
#horsepower için filtreleme
Horsepower_desc = describe["Horsepower"]
q3_hp = Horsepower_desc.iloc[6]
q1_hp = Horsepower_desc.iloc[4]
IQR_hp =q3_hp - q1_hp
top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr*IQR_hp
filter_hp_bottom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_bottom & filter_hp_top
data = data[filter_hp]

#Acceleration için filtreleme
if "Acceleration" in describe:
  acceleration_desc = describe["Acceleration"]
  q3_acc = acceleration_desc.iloc[6]
  q1_acc = acceleration_desc.iloc[4]
  IQR_acc =q3_acc - q1_acc
  top_limit_acc = q3_acc + thr*IQR_acc
  bottom_limit_acc = q1_acc - thr*IQR_acc
  filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
  filter_acc_top= data["Acceleration"] < top_limit_acc
  filter_acc = filter_acc_bottom & filter_acc_top
  data = data[filter_acc]
else:
    print("Acceleration column not found in describe")

#Hedef değişkenin log dönüşümü
sns.histplot(data.target,kde=True, stat="density",bins=30)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, data.target.mean(), data.target.std())
plt.plot(x, p, label="Normal Dağılım", color="red")
plt.legend()
plt.show()

(mu, sigma)= norm.fit(data["target"])
print("mu : {}, sigma = {} ".format(mu,sigma))

#QQ plot
plt.figure()
stats.probplot(data["target"], plot=plt)
plt.show()

# Log dönüşümü ve tekrar QQ plot
data["target"]= np.log1p(data["target"])
plt.figure()
sns.distplot(data.target,kde=True ,fit=norm )
(mu, sigma)= norm.fit(data["target"])
print("mu : {}, sigma = {} ".format(mu,sigma))
plt.figure()
stats.probplot(data["target"], plot=plt)
plt.show()

#Özellikler için skewness
skewed_feats=data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness= pd.DataFrame(skewed_feats, columns=["skewed"])

#One-Hot encoding
data["Cylinders"]=data["Cylinders"].astype(str)
data["Origin"]= data["Origin"].astype(str)
data=pd.get_dummies(data)

#Model verilerini ayırma
x=data.drop("target", axis=1)  #girdi özellikleri
y=data["target"]                     #hedef değişken

print(x.shape)
print(y.shape)
test_size=0.2     #test seti oranı (%20 test seti)
X_train, X_test, Y_train, Y_test=train_test_split(x,y,test_size=test_size, random_state=42)
print("Eğitim ve test seti verileri başarıyla ayrıldı")

#Eğitim setini ölçeklendirme
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Lineer Regresyon
lr=LinearRegression()
lr.fit(X_train,Y_train)
print("LR coef: " ,lr.coef_)
y_predict_dummy =lr.predict(X_test)
mse= mean_squared_error(Y_test, y_predict_dummy)
print("Linear Regression MSE :", mse)

#Ridge Regression
ridge=Ridge(alpha=0.1,random_state =42, max_iter=10000, solver="auto")
ridge.fit(X_train, Y_train)

#Ridge için uygun olan alpha parametreleri
alphas =np.logspace(-4,-0.5,30)
tuned_parameters = [{'alpha' : alphas}]
n_folds=5
clf= GridSearchCV(ridge, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error", refit=True)
clf.fit(X_train,Y_train)

scores=clf.cv_results_["mean_test_score"]
scores_std=clf.cv_results_["std_test_score"]
print("Ridge coef:", ridge.coef_)

y_predict_dummy=ridge.predict(X_test)
mse=mean_squared_error(Y_test, y_predict_dummy)
print("Ridge MSE:" ,mse)

#Ridge sonuclarını görselleştirmek
plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")
plt.show()

#Lasso regression
lasso=Lasso(alpha=0.1, random_state =42, max_iter=10000)
alphas=np.logspace(-4,-0.5,30)

tuned_parameters = [{"alpha" : alphas}]
n_folds = 5

clf = GridSearchCV(lasso,tuned_parameters, cv=n_folds,scoring='neg_mean_squared_error',refit=True)
clf.fit(X_train,Y_train)
scores= clf.cv_results_['mean_test_score']
scores_std=clf.cv_results_['std_test_score']
print("Lasso Coef: ",clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("lasso Best Estimator: ",lasso)

Y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,Y_predicted_dummy)
print("Lasso MSE: ",mse)

# Averaging Models
class AveragingModels(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X,y)

        return self

    def predict(self, X):
        predictions =np.column_stack([model.predict(X) for model in self.models_])
        return  np.mean(predictions,axis=1)

averaged_models = AveragingModels(models=(ridge,lasso))
averaged_models.fit(X_train, Y_train)

y_predicted_dummy =averaged_models.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print(("Averaged Models MSE: ",mse))
