import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
scaler = MinMaxScaler()

class dataset_randomforest():

    def __init__(self,data_path) -> None:

        self.data_path = data_path

        self.data = self.dataloader()


    def data_analysis(self):
        df_list = []
        count = 0
        for i in os.listdir(self.data_path):
            data_path = os.path.join(self.data_path,i)
            temp = pd.read_csv(data_path)[["PRPRTY_DMGE_AMT","TIME_UNIT_WD","LOC_INFO_X","LOC_INFO_Y",
            "TIME_UNIT_WS","TP","SPT_FRSTT_DIST","SPT_SAFE_CNTER_DIST","DSP_REQRE_TIME",
            "HUMIDITY","FSMDEM_TM","FSMDEM_YMD","FIRE_OCRN_TIME","FIRE_OCRN_YMD"]]
            df_list.append(temp)
            count += 1 
        a = pd.concat(df_list, ignore_index=True)
        plt.rcParams["figure.figsize"] = (20,20)
        plt.rc('font', size=10)
        sb.heatmap(a.corr(method = 'pearson'),
                annot = True, #실제 값 화면에 나타내기
                cmap = 'Greens', #색상
                vmin = -1, vmax=1 , #컬러차트 영역 -1 ~ +1
                )
        plt.savefig("./data_heatmap.jpg")
        return a.corr(method='pearson')
        #return a.corr(method='pearson')["PRPRTY_DMGE_AMT"].sort_values(ascending=False)
    def dataloader(self):
        df_list = []
        count = 0
        for i in os.listdir(self.data_path):
            data_path = os.path.join(self.data_path,i)
            temp = pd.read_csv(data_path)[["PRPRTY_DMGE_AMT","TIME_UNIT_WD","LOC_INFO_X","LOC_INFO_Y",
            "TIME_UNIT_WS","TP","SPT_FRSTT_DIST","SPT_SAFE_CNTER_DIST","DSP_REQRE_TIME",
            "HUMIDITY","FSMDEM_TM","FSMDEM_YMD","FIRE_OCRN_TIME","FIRE_OCRN_YMD"]]
            df_list.append(temp)
            count += 1 
        print("사용 데이터 파일 수 : ",count)
        return self.null_data_fix(pd.concat(df_list, ignore_index=True))

    def null_data_fix(self,data):
        print("결측치 해결전")
        print(data.isnull().sum()) #결측치 확인
        data.to_csv(os.path.join('total_data_set.csv'))
        #data = data.fillna(data.mean())
        data = data.where(pd.notnull(data), data.mean(), axis='columns')
        print("결측치 해결후")
        print(data.isnull().sum()) #결측치 확인
        return self.target_make(data)

    def make_dic_m(self):
        dic = {}
        for data in range(1,13):
            if(len(str(data)) == 1):
                dic["0"+str(data)] = data
            else:
                dic[str(data)] = data
        return dic

    def make_dic_t(self):
        dic = {}
        for data in range(0,25):
            if(len(str(data)) == 1):
                dic["0"+str(data)] = data
            else:
                dic[str(data)] = data
        return dic

    def target_make(self,  data):

        label = [ 0 if i == 0 else 1 for i in data["PRPRTY_DMGE_AMT"]]
        print("피해 없음 데이터 : ",label.count(0))
        print("피해 발생 데이터 : ", len(label) - label.count(0))
        plt.pie([label.count(0),len(label) - label.count(0)],autopct='%.1f%%', labels=["No Damage","Damage"])
        plt.savefig("./data_rat.png")
        plt.clf()
        FSMDEM_M_dic = self.make_dic_m()
        FIRE_OCRN_M_dic = self.make_dic_m()

        FSMDEM_T_dic = self.make_dic_t()
        FIRE_OCRN_T_dic = self.make_dic_t()

        temp = []
        for i in data["FSMDEM_YMD"]:
            if(len(str(i))):
                temp.append(str(i)[4:6])
        print(FSMDEM_M_dic)
        temp_FSMDEM_M = pd.DataFrame( [FSMDEM_M_dic[i] for i in temp], columns=["FSMDEM_YMD"] )

        temp = []
        for i in data["FIRE_OCRN_YMD"]:
            if(len(str(i))):
                temp.append(str(i)[4:6])
        temp_FIRE_OCRN_M = pd.DataFrame( [FIRE_OCRN_M_dic[i] for i in temp], columns=["FIRE_OCRN_YMD"] )

        temp = []
        for i in data["FSMDEM_TM"]:
            if(len(str(i)) == 6):
                temp.append(str(i)[0:2])
            else:
                temp.append("0"+str(i)[0])
        temp_FSMDEM_T = pd.DataFrame( [FSMDEM_T_dic[i] for i in temp], columns=["FSMDEM_TM"] )

        temp = []
        for i in data["FIRE_OCRN_TIME"]:
            if(len(str(i)) == 6):
                temp.append(str(i)[0:2])
            else:
                temp.append("0"+str(i)[0])
        temp_FIRE_OCRN_T = pd.DataFrame( [FIRE_OCRN_T_dic[i] for i in temp], columns=["FIRE_OCRN_TIME"] )
    
        data.drop(columns=["FSMDEM_TM","FSMDEM_YMD","FIRE_OCRN_TIME", "FIRE_OCRN_YMD"])
        x2 = pd.concat([data,temp_FSMDEM_M,temp_FIRE_OCRN_M,temp_FSMDEM_T,temp_FIRE_OCRN_T],axis=1)

        self.data_view(x2)
        
        x2 = scaler.fit_transform(x2)
        

        y = pd.DataFrame(label, columns=["label"])
        X_train, X_test, y_train, y_test = train_test_split( x2 , y , test_size=0.2 , random_state=5 )

        return  [X_train, X_test, y_train, y_test]

    def data_view(self,data):
        print("data : " ,data[:10])
        data.to_csv(os.path.join(r".",'class_Data.csv'))

class dataset_randomforest_linear():

    def __init__(self,data_path) -> None:

        self.data_path = data_path

        self.data = self.dataloader()


    def data_analysis(self):
        df_list = []
        count = 0
        for i in os.listdir(self.data_path):
            data_path = os.path.join(self.data_path,i)
            temp = pd.read_csv(data_path)[["PRPRTY_DMGE_AMT","TIME_UNIT_WD","LOC_INFO_X","LOC_INFO_Y",
            "TIME_UNIT_WS","TP","SPT_FRSTT_DIST","SPT_SAFE_CNTER_DIST","DSP_REQRE_TIME",
            "HUMIDITY","FSMDEM_TM","FSMDEM_YMD","FIRE_OCRN_TIME","FIRE_OCRN_YMD"]]
            df_list.append(temp)
            count += 1 
        a = pd.concat(df_list, ignore_index=True)
        plt.rcParams["figure.figsize"] = (20,20)
        plt.rc('font', size=10)
        sb.heatmap(a.corr(method = 'pearson'),
                annot = True, #실제 값 화면에 나타내기
                cmap = 'Greens', #색상
                vmin = -1, vmax=1 , #컬러차트 영역 -1 ~ +1
                )
        plt.savefig("./data_heatmap.jpg")
        return a.corr(method='pearson')
        #return a.corr(method='pearson')["PRPRTY_DMGE_AMT"].sort_values(ascending=False)
    def dataloader(self):
        df_list = []
        count = 0
        for i in os.listdir(self.data_path):
            data_path = os.path.join(self.data_path,i)
            temp = pd.read_csv(data_path)[["PRPRTY_DMGE_AMT","TIME_UNIT_WD","LOC_INFO_X","LOC_INFO_Y",
            "TIME_UNIT_WS","TP","SPT_FRSTT_DIST","SPT_SAFE_CNTER_DIST","DSP_REQRE_TIME",
            "HUMIDITY","FSMDEM_TM","FSMDEM_YMD","FIRE_OCRN_TIME","FIRE_OCRN_YMD"]]
            df_list.append(temp)
            count += 1 
        print("사용 데이터 파일 수 : ",count)
        return self.null_data_fix(pd.concat(df_list, ignore_index=True))

    def null_data_fix(self,data):
        print("결측치 해결전")
        print(data.isnull().sum()) #결측치 확인
        data.to_csv(os.path.join('total_data_set.csv'))
        #data = data.fillna(data.mean())
        data = data.where(pd.notnull(data), data.mean(), axis='columns')
        print("결측치 해결후")
        print(data.isnull().sum()) #결측치 확인
        return self.target_make(data) 

    def make_dic_m(self):
        dic = {}
        for data in range(1,13):
            if(len(str(data)) == 1):
                dic["0"+str(data)] = data
            else:
                dic[str(data)] = data
        return dic

    def make_dic_t(self):
        dic = {}
        for data in range(0,25):
            if(len(str(data)) == 1):
                dic["0"+str(data)] = data
            else:
                dic[str(data)] = data
        return dic

    def target_make(self,  data):

        label = [ i for i in data["PRPRTY_DMGE_AMT"]]
        print("피해 없음 데이터 : ",label.count(0))
        print("피해 발생 데이터 : ", len(label) - label.count(0))
        FSMDEM_M_dic = self.make_dic_m()
        FIRE_OCRN_M_dic = self.make_dic_m()

        FSMDEM_T_dic = self.make_dic_t()
        FIRE_OCRN_T_dic = self.make_dic_t()

        temp = []
        for i in data["FSMDEM_YMD"]:
            if(len(str(i))):
                temp.append(str(i)[4:6])
        print(FSMDEM_M_dic)
        temp_FSMDEM_M = pd.DataFrame( [FSMDEM_M_dic[i] for i in temp], columns=["FSMDEM_YMD"] )

        temp = []
        for i in data["FIRE_OCRN_YMD"]:
            if(len(str(i))):
                temp.append(str(i)[4:6])
        temp_FIRE_OCRN_M = pd.DataFrame( [FIRE_OCRN_M_dic[i] for i in temp], columns=["FIRE_OCRN_YMD"] )

        temp = []
        for i in data["FSMDEM_TM"]:
            if(len(str(i)) == 6):
                temp.append(str(i)[0:2])
            else:
                temp.append("0"+str(i)[0])
        temp_FSMDEM_T = pd.DataFrame( [FSMDEM_T_dic[i] for i in temp], columns=["FSMDEM_TM"] )

        temp = []
        for i in data["FIRE_OCRN_TIME"]:
            if(len(str(i)) == 6):
                temp.append(str(i)[0:2])
            else:
                temp.append("0"+str(i)[0])
        temp_FIRE_OCRN_T = pd.DataFrame( [FIRE_OCRN_T_dic[i] for i in temp], columns=["FIRE_OCRN_TIME"] )
        data.drop(columns=["FSMDEM_TM","FSMDEM_YMD","FIRE_OCRN_TIME", "FIRE_OCRN_YMD"])
        x2 = pd.concat([data,temp_FSMDEM_M,temp_FIRE_OCRN_M,temp_FSMDEM_T,temp_FIRE_OCRN_T],axis=1)

        self.data_view(x2)
        
        x2 = scaler.fit_transform(x2)
        
        
        y = pd.DataFrame(label, columns=["label"])
        X_train, X_test, y_train, y_test = train_test_split( x2 , y , test_size=0.2 , random_state=5 )
        return  [X_train, X_test, y_train, y_test]

    def data_view(self,data):
        print("data : " ,data[:10])
        data.to_csv(os.path.join(r".",'linear_data.csv'))



