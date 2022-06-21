# Import required libraries
import pandas as pd

# Import necessary modules
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
dat = pd.read_csv('diabetes.csv') 
print(dat.shape)
print(dat.describe().transpose())
x1 = dat.drop('Outcome', axis=1).values 
y1 = dat['Outcome'].values
# Evaluate using a train and a test set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Holdout Validation Approach Accuracy: %.2f%%" % (result*100.0))

kfold = model_selection.KFold(n_splits=10, random_state=100, shuffle=True)
model_kfold = LogisticRegression(solver='lbfgs', max_iter=1000)
results_kfold = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold)
print("K-fold Cross-Validation Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 

skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)
model_skfold = LogisticRegression(solver='lbfgs', max_iter=1000)
results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold)
print("Stratified K-fold Cross-Validation Accuracy: %.2f%%" % (results_skfold.mean()*100.0))

# Leave One Out Cross-Validation
loocv = model_selection.LeaveOneOut()
model_loocv = LogisticRegression(solver='lbfgs', max_iter=1000)
results_loocv = model_selection.cross_val_score(model_loocv, x1, y1, cv=loocv)
print("Leave One Out Cross-Validation Accuracy: %.2f%%" % (results_loocv.mean()*100.0))

# Repeated Random Test-Train Splits
kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.30, random_state=100)
model_shufflecv = LogisticRegression(solver='lbfgs', max_iter=1000)
results_4 = model_selection.cross_val_score(model_shufflecv, x1, y1, cv=kfold2)
print("Repeated Random Test-Train Splits Accuracy: %.2f%% (%.2f%%)" % (results_4.mean()*100.0, results_4.std()*100.0))