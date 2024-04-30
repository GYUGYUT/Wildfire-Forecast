from sklearn.ensemble import *
from sklearn.linear_model import *
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import *
from catboost import *
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from dataloader import *
from sklearn.metrics import *
import numpy as np
import seaborn as sn
"""
XGBClassifier
여러 개의 결정 트리를 임의적으로 학습하는 부스팅 앙상블
순차적 학습 방법을 사용함
"""
import warnings
warnings.filterwarnings('ignore')
x_vars = ["TIME_UNIT_WD","LOC_INFO_X",
"LOC_INFO_Y","TIME_UNIT_WS","TP",
"SPT_FRSTT_DIST","SPT_SAFE_CNTER_DIST","DSP_REQRE_TIME",
"FIRE_SUPESN_TIME","TIME_UNIT_WD","HUMIDITY","FSMDEM_TM","FSMDEM_YMD","FIRE_OCRN_TIME","FIRE_OCRN_YMD"]
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def confusion(con_y_label,con_y_pred,best_path):
    classes = ["No", "YES"]
    filepath = os.path.join(best_path,r"conpusion.png")
    cf_matrix = confusion_matrix(con_y_label, con_y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(filepath)


def report(file_path,y_pred,y_true):
    f = open(file_path, 'w')
    f.write(classification_report( y_pred,y_true))
    f.close()

class Classifier_model():

    def __init__(self) -> None:

        self.classifier_model_data = dataset_randomforest(r"./dataset").data

        self.XGBClassifier_model = XGBClassifier(random_state=0) 
        self.GradientBoostingClassifier_model = GradientBoostingClassifier(random_state=0)
        self.LGBMClassifier_model = LGBMClassifier(random_state=0)
        self.CatBoostClassifier = CatBoostClassifier(random_state=1004)
        # self.SVCClassifier = SVC(random_state=1004)
    
    def XGBClassifier_train(self):
        best_path = r"./XGBClassifier"
        createDirectory(best_path)
        xgb_param_grid={
        'n_estimators' : [5,10,15,20,30],
        'learning_rate' : [0.01,0.05,0.1,0.15],
        'max_depth' : [3,5,7,10,15],
        'gamma' : [0,1,2,3],
        'colsample_bytree' : [0.8,0.9],
            }
        # xgb_param_grid={
        # 'n_estimators' : [500,550,600,650,1000],
        # 'learning_rate' : [0.15,0.2,0.25,0.3,0.4],
        # 'max_depth' : [6,7,8,9,10],
        # 'gamma' : [0],
        # 'colsample_bytree' : [0.6,0.7,0.8],
        #     }
        xgb_grid = GridSearchCV(self.XGBClassifier_model, param_grid = xgb_param_grid, scoring="accuracy", n_jobs=-1, verbose = 2) 
        xgb_grid.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        
        print("best param : ",xgb_grid.best_params_)
        
        result_df = pd.DataFrame(xgb_grid.cv_results_)

        print(result_df[['params','mean_test_score','rank_test_score']].head(10))
        result_df.to_csv(os.path.join(best_path,'XGBClassifier_result.csv'))
        predic_model = XGBClassifier(
            n_estimators = xgb_grid.best_params_["n_estimators"],
            learning_rate = xgb_grid.best_params_["learning_rate"],
            max_depth = xgb_grid.best_params_["max_depth"],
            gamma = xgb_grid.best_params_["gamma"],
        )

        predic_model.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        predic = predic_model.predict(self.classifier_model_data[1])

        
        with open(os.path.join(best_path,'XGBClassifier_result.txt'), "w") as f:
            f.write("train accuracy : {0: .4f}\n".format(xgb_grid.best_score_))
            f.write("best param : {}\n".format(xgb_grid.best_params_))
        confusion(list(self.classifier_model_data[3]["label"]),list(predic),best_path)
        report(os.path.join(best_path,'XGBClassifier_report.txt'),list(predic),list(self.classifier_model_data[3]["label"]))

    def GradientBoostingClassifier_train(self):
        param_gbm = {"max_depth" : [5,6,7,8,9,10],
            "learning_rate" : [0.001,0.01,0.05,0.1],
            "n_estimators" : [15,20,30]
            }
        # param_gbm = {"max_depth" : [9],
        #     "learning_rate" : [0.07,0.1],
        #     "n_estimators" : [500,550,600,700,1000]
        #     }
        best_path = r"./GradientBoostingClassifier"
        createDirectory(best_path)
        Classifier = GridSearchCV(self.GradientBoostingClassifier_model, param_grid = param_gbm, scoring="accuracy", n_jobs=-1, verbose = 2) 
        Classifier.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        
        print("best param : ",Classifier.best_params_)
        
        result_df = pd.DataFrame(Classifier.cv_results_)

        print(result_df[['params','mean_test_score','rank_test_score']].head(10))
        result_df.to_csv(os.path.join(best_path,'GradientBoostingClassifier_result.csv'))
        predic_model = GradientBoostingClassifier(
            n_estimators = Classifier.best_params_["n_estimators"],
            learning_rate = Classifier.best_params_["learning_rate"],
            max_depth = Classifier.best_params_["max_depth"],
        )

        predic_model.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        predic = predic_model.predict(self.classifier_model_data[1])

        
        with open(os.path.join(best_path,'GradientBoostingClassifier_result.txt'), "w") as f:
            f.write("train accuracy : {0: .4f}\n".format(Classifier.best_score_))
            f.write("best param : {}\n".format(Classifier.best_params_))
        confusion(list(self.classifier_model_data[3]["label"]),list(predic),best_path)
        report(os.path.join(best_path,'GradientBoostingClassifier_report.txt'),list(predic),list(self.classifier_model_data[3]["label"]))
    
    def LGBMClassifier_train(self):
        param_lgb = {"learning_rate" : [0.001,0.01,0.05,0.1],
            "max_depth" : [4,5,6,7,8,9,10],
            "n_estimators" : [5,10,15],
            }
        # param_lgb = {"learning_rate" : [0.1,0,15],
        #     "max_depth" : [30,40,45,50],
        #     "num_leaves" : [100,300,500,900,1200],
        #     "n_estimators" : [250,270,300],
        #     "learning_rate" : [0.05,0.1,]
        #     }
        best_path = r"./LGBMClassifier"
        createDirectory(best_path)
        Classifier = GridSearchCV(self.LGBMClassifier_model, param_grid = param_lgb, scoring="accuracy", n_jobs=-1, verbose = 2) 
        Classifier.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        
        print("best param : ",Classifier.best_params_)
        
        result_df = pd.DataFrame(Classifier.cv_results_)

        print(result_df[['params','mean_test_score','rank_test_score']].head(10))
        result_df.to_csv(os.path.join(best_path,'LGBMClassifier_result.csv'))
        predic_model = LGBMClassifier(
            n_estimators = Classifier.best_params_["n_estimators"],
            learning_rate = Classifier.best_params_["learning_rate"],
            max_depth = Classifier.best_params_["max_depth"],
        )

        predic_model.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        predic = predic_model.predict(self.classifier_model_data[1])
        
        with open(os.path.join(best_path,'LGBMClassifier_result.txt'), "w") as f:
            f.write("train accuracy : {0: .4f}\n".format(Classifier.best_score_))
            f.write("best param : {}\n".format(Classifier.best_params_))
        confusion(list(self.classifier_model_data[3]["label"]),list(predic),best_path)
        report(os.path.join(best_path,'LGBMClassifier_report.txt'),list(predic),list(self.classifier_model_data[3]["label"]))
    
    def CatBoostClassifier_train(self):
        best_path = r"./CatBoostClassifier"
        createDirectory(best_path)
        param_cat = {"depth" : [2,3,4,5],
        "iterations" : [15,30,40],
        "learning_rate" : [0.001,0.01,0.05], 
            }
        Classifier = GridSearchCV(self.CatBoostClassifier, param_grid = param_cat, scoring="accuracy",verbose = 2) 
        Classifier.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        
        print("train accuracy : {0: .4f}".format(Classifier.best_score_))
        print("best param : ",Classifier.best_params_)
        
        result_df = pd.DataFrame(Classifier.cv_results_)

        print(result_df[['params','mean_test_score','rank_test_score']].head(10))
        result_df.to_csv(os.path.join(best_path,'CatBoostClassifier_result.csv'))
        predic_model = CatBoostClassifier(
            depth = Classifier.best_params_["depth"],
            iterations = Classifier.best_params_["iterations"],
            learning_rate = Classifier.best_params_["learning_rate"],
        )

        predic_model.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        predic = predic_model.predict(self.classifier_model_data[1])

        with open(os.path.join(best_path,'CatBoostClassifier_result.txt'), "w") as f:
            f.write("train accuracy : {0: .4f}\n".format(Classifier.best_score_))
            f.write("best param : {}\n".format(Classifier.best_params_))
        confusion(list(self.classifier_model_data[3]["label"]),list(predic),best_path)
        report(os.path.join(best_path,'CatBoostClassifier_report.txt'),list(predic),list(self.classifier_model_data[3]["label"]))
    
    # def SVCClassifier_train(self):
    #     best_path = r"./SVRClassifier"
    #     createDirectory(best_path)
    #     param_cat = {'C': [0.001,0.01,0.1], 
    #             'gamma': [10,1, 0.1, 0.01, 0.001, 0.0001],
    #             'kernel': ['rbf', 'poly', 'sigmoid'] 
    #         }
    #     Classifier = GridSearchCV(self.SVCClassifier, param_grid = param_cat, scoring="accuracy",verbose = 2, n_jobs=-1) 
    #     Classifier.fit(self.classifier_model_data[0], self.classifier_model_data[2])
        
    #     print("best param : ",Classifier.best_params_)
        
    #     result_df = pd.DataFrame(Classifier.cv_results_)

    #     print(result_df[['params','mean_test_score','rank_test_score']].head(10))
    #     result_df.to_csv(os.path.join(best_path,'CatBoostClassifier_result.csv'))
    #     predic_model = SVC(
    #         C = Classifier.best_params_['C'],
    #         gamma = Classifier.best_params_["gamma"],
    #         kernel = Classifier.best_params_["kernel"],
    #     )

    #     predic_model.fit(self.classifier_model_data[0], self.classifier_model_data[2])
    #     predic = predic_model.predict(self.classifier_model_data[1])

        
    #     with open(os.path.join(best_path,'SVCClassifier_result.txt'), "w") as f:
    #         f.write("train accuracy : {0: .4f}\n".format(Classifier.best_score_))
    #         f.write("best param : {}\n".format(Classifier.best_params_))
    #     confusion(list(self.classifier_model_data[3]["label"]),list(predic),best_path)
    #     report(os.path.join(best_path,'SVCClassifier_report.txt'),list(predic),list(self.classifier_model_data[3]["label"]))

class Regression_model():

    def __init__(self) -> None:

        self.Regression_model_data = dataset_randomforest_linear(r"./dataset").data

        self.LogisticRegression_model = LogisticRegression(random_state=0)
        self.RandomForestRegressor_model = RandomForestRegressor(random_state=0)
        self.GradientBoostingRegressor_model = GradientBoostingRegressor(random_state=0)
        self.CatBoostRegressor_model = CatBoostRegressor(random_state=1004)
        #self.SVRRegressor_model = SVR()

    def LogisticRegression_train(self):
        best_path = r"./LogisticRegression"

        createDirectory(best_path)

        param = {"penalty" : ["l2","l1"],
            "C" : [0.01,0.1,1,5,10],
            "max_iter" : [25,30,50,100,150],
        }
        Grid_model = GridSearchCV(self.LogisticRegression_model, param_grid = param, scoring="accuracy", n_jobs=-1, verbose = 2) 
        Grid_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        
        predic_model = LogisticRegression(
            penalty = Grid_model.best_params_["penalty"],
            C = Grid_model.best_params_["C"],
            max_iter = Grid_model.best_params_["max_iter"],
        )
        predic_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        predic_train = predic_model.predict(self.Regression_model_data[0])
        predic_test = predic_model.predict(self.Regression_model_data[1])


        train_score = predic_model.score(self.Regression_model_data[0], self.Regression_model_data[2])
        test_score = predic_model.score(self.Regression_model_data[1], self.Regression_model_data[3])

        mse_train = mean_squared_error(self.Regression_model_data[2],predic_train)
        mse_test = mean_squared_error(self.Regression_model_data[3],predic_test)

        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)

        
        with open(os.path.join(best_path,'LogisticRegression_result.txt'), "w") as f:

            f.write("train\n")
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntrain_score : {0:3f}\n\n'.format(mse_train, rmse_train,r2_score(self.Regression_model_data[2], predic_train),train_score))
            
            f.write("test accuracy : {0: .4f}\n".format(accuracy_score(self.Regression_model_data[3], predic_test)))
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntest_score : {0:3f}\n\n'.format(mse_test, rmse_test,r2_score(self.Regression_model_data[3], predic_test),test_score))

            f.write("best param : {}\n".format(Grid_model.best_params_))

    def RandomForestRegressor_train(self):
        best_path = r"./RandomForestRegressor"
        createDirectory(best_path)
        param =  {'n_estimators': [5,10,15,20,25],
            'max_features': list(range(1, len(x_vars), 2)),
            }

        Grid_model = GridSearchCV(self.RandomForestRegressor_model, param_grid = param, scoring="accuracy", n_jobs=-1, verbose = 2) 
        Grid_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        
        predic_model = RandomForestRegressor(
            n_estimators = Grid_model.best_params_["n_estimators"],
            max_features = Grid_model.best_params_["max_features"]
        )
        predic_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        predic_train = predic_model.predict(self.Regression_model_data[0])
        predic_test = predic_model.predict(self.Regression_model_data[1])


        train_score = predic_model.score(self.Regression_model_data[0], self.Regression_model_data[2])
        test_score = predic_model.score(self.Regression_model_data[1], self.Regression_model_data[3])

        mse_train = mean_squared_error(self.Regression_model_data[2],predic_train)
        mse_test = mean_squared_error(self.Regression_model_data[3],predic_test)

        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)

        
        with open(os.path.join(best_path,'RandomForestRegressor_result.txt'), "w") as f:

            f.write("train\n")
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntrain_score : {0:3f}\n\n'.format(mse_train, rmse_train,r2_score(self.Regression_model_data[2], predic_train),train_score))
            
            f.write("test\n")
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntest_score : {0:3f}\n\n'.format(mse_test, rmse_test,r2_score(self.Regression_model_data[3], predic_test),test_score))

            f.write("best param : {}\n".format(Grid_model.best_params_))

    def GradientBoostingRegressor_train(self):
        best_path = r"./GradientBoostingRegressor"
        createDirectory(best_path)
        param =  {'n_estimators': [5,10,15],
            'max_features': list(range(1, len(x_vars), 2)),
            'min_samples_leaf': [1, 3, 4, 5]
            }

        Grid_model = GridSearchCV(self.GradientBoostingRegressor_model, param_grid = param, scoring="accuracy", n_jobs=-1, verbose = 2) 
        Grid_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        
        predic_model = GradientBoostingRegressor(
            max_features = Grid_model.best_params_["max_features"],
            min_samples_leaf = Grid_model.best_params_["min_samples_leaf"],
            n_estimators = Grid_model.best_params_["n_estimators"],
        )
        predic_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])

        predic_train = predic_model.predict(self.Regression_model_data[0])
        predic_test = predic_model.predict(self.Regression_model_data[1])


        train_score = predic_model.score(self.Regression_model_data[0], self.Regression_model_data[2])
        test_score = predic_model.score(self.Regression_model_data[1], self.Regression_model_data[3])

        mse_train = mean_squared_error(self.Regression_model_data[2],predic_train)
        mse_test = mean_squared_error(self.Regression_model_data[3],predic_test)

        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)

        
        with open(os.path.join(best_path,'GradientBoostingRegressor_result.txt'), "w") as f:

            f.write("train\n")
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntrain_score : {0:3f}\n\n'.format(mse_train, rmse_train,r2_score(self.Regression_model_data[2], predic_train),train_score))
            
            f.write("test\n")
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntest_score : {0:3f}\n\n'.format(mse_test, rmse_test,r2_score(self.Regression_model_data[3], predic_test),test_score))

            f.write("best param : {}\n".format(Grid_model.best_params_))

    def CatBoostRegressor_train(self):
        best_path = r"./CatBoostRegressor"
        createDirectory(best_path)
        param = {
        "loss_function" : ['RMSE','MAE'],
        "depth" : [2,3,4,5],
        "iterations" : [10,15,20],
        "learning_rate" : [0.01,0.05,0.1], 
        }

        Grid_model = GridSearchCV(self.CatBoostRegressor_model, param_grid = param, scoring="accuracy",verbose = 2) 
        Grid_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        
        predic_model = CatBoostRegressor(
            loss_function = Grid_model.best_params_["loss_function"],
            depth = Grid_model.best_params_["depth"],
            iterations = Grid_model.best_params_["iterations"],
            learning_rate = Grid_model.best_params_["learning_rate"],
        )
        predic_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        predic_train = predic_model.predict(self.Regression_model_data[0])
        predic_test = predic_model.predict(self.Regression_model_data[1])


        train_score = predic_model.score(self.Regression_model_data[0], self.Regression_model_data[2])
        test_score = predic_model.score(self.Regression_model_data[1], self.Regression_model_data[3])

        mse_train = mean_squared_error(self.Regression_model_data[2],predic_train)
        mse_test = mean_squared_error(self.Regression_model_data[3],predic_test)

        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)

        
        with open(os.path.join(best_path,'CatBoostRegressor_result.txt'), "w") as f:

            f.write("train\n")
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntrain_score : {0:3f}\n\n'.format(mse_train, rmse_train,r2_score(self.Regression_model_data[2], predic_train),train_score))
            
            f.write("test\n")
            f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntest_score : {0:3f}\n\n'.format(mse_test, rmse_test,r2_score(self.Regression_model_data[3], predic_test),test_score))

            f.write("best param : {}\n".format(Grid_model.best_params_))
    
    # def SVRRegressor_train(self):
    #     best_path = r"./SVRRegressor"
    #     createDirectory(best_path)
    #     param = {'C': [0.01,0.1, 1, 10, 100, 1000], 
    #             'gamma': [10,1, 0.1, 0.01, 0.001, 0.0001],
    #             'kernel': ['rbf', 'poly', 'sigmoid'] 
    #         }
    #     Grid_model = GridSearchCV(self.SVRRegressor_model, param_grid = param, scoring="accuracy",verbose = 2,n_jobs=-1,) 
    #     Grid_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
        
    #     predic_model = SVR(
    #         C = Grid_model.best_params_["C"],
    #         gamma = Grid_model.best_params_["gamma"],
    #         kernel = Grid_model.best_params_["kernel"],
    #     )
    #     predic_model.fit(self.Regression_model_data[0], self.Regression_model_data[2])
    #     predic_train = predic_model.predict(self.Regression_model_data[0])
    #     predic_test = predic_model.predict(self.Regression_model_data[1])


    #     train_score = predic_model.score(self.Regression_model_data[0], self.Regression_model_data[2])
    #     test_score = predic_model.score(self.Regression_model_data[1], self.Regression_model_data[3])

    #     mse_train = mean_squared_error(self.Regression_model_data[2],predic_train)
    #     mse_test = mean_squared_error(self.Regression_model_data[3],predic_test)

    #     rmse_train = np.sqrt(mse_train)
    #     rmse_test = np.sqrt(mse_test)

        
    #     with open(os.path.join(best_path,'SVRRegressor_result.txt'), "w") as f:

    #         f.write("train\n")
    #         f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntrain_score : {0:3f}\n\n'.format(mse_train, rmse_train,r2_score(self.Regression_model_data[2], predic_train),train_score))
            
    #         f.write("test\n")
    #         f.write('MSE : {0:.3f}\nRMSE : {1:.3f}\nR^2(Variance score) : {0:.3f}\ntest_score : {0:3f}\n\n'.format(mse_test, rmse_test,r2_score(self.Regression_model_data[3], predic_test),test_score))

    #         f.write("best param : {}\n".format(Grid_model.best_params_))
