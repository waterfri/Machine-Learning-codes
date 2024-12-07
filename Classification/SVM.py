from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metric import accuracy_score

# load datasets
iris = datasets.load_iris()
X = iris.data[:100] # 详见鸢尾花数据集构成
y = iris.target[:100]

# split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create SVM model
svm = SVC(kernel='linear', C=1.0)
# C值是一个惩罚参数
# C值过大，倾向于严格分类，容易过拟合
# C值过小，允许一定程度的分类错误，有更大的间隔，降低过拟合风险
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))