from dataloader import *
from train_model import *
def start():
    data = dataset_randomforest(r"./dataset")
    data2 = dataset_randomforest_linear(r"./dataset")
    print(data.data_analysis())
    print(data2.data_analysis())
    
    Classifier = Classifier_model()
    Classifier.XGBClassifier_train()
    Classifier.GradientBoostingClassifier_train()
    Classifier.LGBMClassifier_train()
    Classifier.CatBoostClassifier_train()
    

    Regression = Regression_model()
    Regression.LogisticRegression_train()
    Regression.RandomForestRegressor_train()
    Regression.GradientBoostingRegressor_train()
    Regression.CatBoostRegressor_train()


if __name__ == '__main__':
    start()


