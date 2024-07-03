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
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import os, itertools, random, argparse
import joblib
import optuna
import warnings
warnings.filterwarnings(action='ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgbm', choices=['lr', 'sgd', 'mlp', 'knn', 'rf', 'dt', 'gaunb', 'adaboost', 'gradientboost', 'xgboost', 'lightgbm', 'catboost'], required=True, help='model to choose')
    parser.add_argument('--tuning', default=True, choices=[True, False], required=True, help='hyperparameter tuning w/ Optuna')

    args = parser.parse_args()
    print('--- Parameters ---')
    print(args)
    print('-' * 30)

    X_train = pd.read_csv('../dataset/X_train.csv')
    y_train = pd.read_csv('../dataset/y_train.csv')

    # Training Model
    if args.model == 'lr':
        param = {
            'C': trial.suggest_loguniform('C', 1e-5, 1e2)
        }
        train_model = LogisticRegression(C=1e2,
                          multi_class='ovr',
                          random_state=17,
                          max_iter=200
                          )
        train_model.fit(X_train, y_train)
    
    elif args.model == 'sgd':
        train_model = SGDClassifier()
        train_model.fit(X_train, y_train)

    elif args.model == 'mlp':
        train_model = MLPClassifier(random_state=17, max_iter=1000)
        train_model.fit(X_train, y_train)

    elif args.model == 'knn':
        train_model = KNeighborsClassifier(n_neighbors=5, p=2)
        train_model.fit(X_train, y_train)

    elif args.model == 'rf':
        train_model = RandomForestClassifier(n_estimators=5, random_state=17)
        train_model.fit(X_train, y_train)
    
    elif args.model == 'dt':
        train_model = DecisionTreeClassifier(random_state=17)
        train_model.fit(X_train, y_train)
    
    elif args.model == 'gaunb':
        train_model = GaussianNB()
        train_model.fit(X_train, y_train)

    elif args.model == 'adaboost':
        train_model = AdaBoostClassifier(n_estimators=50, random_state=17)
        train_model.fit(X_train, y_train)
    
    elif args.model == 'gradientboost':
        train_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=17)
        train_model.fit(X_train, y_train)
    
    elif args.model == 'xgboost':
        le = LabelEncoder()
        y_train_xgb = le.fit_transform(y_train)
        train_model = XGBClassifier()
        train_model.fit(X_train, y_train_xgb)
    
    elif args.model == 'lightgbm':
        train_model = LGBMClassifier()
        train_model.fit(X_train, y_train)

    elif args.model == 'catboost':
        train_model = CatBoostClassifier()
        train_model.fit(X_train, y_train)
    
    print('--- Training the model is ended. ---')

    # Saving Model
    joblib.dump(train_model, f'../saving_model/{args.model}_save.pkl')
    print('--- Trained model was saved. ---')

    return args.model

if __name__ == '__main__':
    main()
