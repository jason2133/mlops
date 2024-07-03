import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score,ConfusionMatrixDisplay,confusion_matrix,precision_score,recall_score,roc_curve,roc_auc_score,balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from xgboost import plot_importance
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import os, itertools, random, argparse
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgbm', choices=['lr', 'sgd', 'mlp', 'knn', 'rf', 'dt', 'gaunb', 'adaboost', 'gradientboost', 'xgboost', 'lightgbm', 'catboost'], help='model to choose')
    args = parser.parse_args()
    print('--- Parameters ---')
    print(args)
    print('-' * 30)

    trained_model = joblib.load(f'../saving_model/{args.model}_save.pkl')
    X_test = pd.read_csv('../dataset/X_test.csv')
    y_test = pd.read_csv('../dataset/y_test.csv')

    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    print(' ')
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == '__main__':
    main()
