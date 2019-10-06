import csv
import numpy as np
import pymongo
import random

import time
from sklearn import preprocessing
from sklearn.svm import SVR
import warnings
import json
import math
warnings.filterwarnings("ignore")
# from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
# In[2]:
class Bunch(dict):
    def __init__(self,**kwargs):
        dict.__init__(self,kwargs)
    def __setattr__(self,key,value):
        self[key]=value
    def __getattr__(self,key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
            
    def __getstate__(self):
        return self.__dict__
# In[182]:
def Connectdatabase():
    conn = pymongo.MongoClient(host="localhost", port=27017)
    db = conn.MongoDB_Data
    return db
def load_data(filename):
    with open(filename) as f:
        data_file = csv.reader(f)
        data = []
        target = []
        temp = next(data_file)
        feature_names = np.array(temp)
        for i,d in enumerate(data_file):
            temp=[]
            for j in d:
                if j=='':
                    j=0
                temp.append(j)
            data.append(np.asarray(temp[:-1],dtype = np.float))
            target.append(np.asarray(d[-1],dtype = np.float))
        data = np.array(data)
        print(data)
        target = np.array(target)
        target=target.reshape(-1,1)
        print(target)
        n_samples = data.shape[0]
        n_features = data.shape[1]
    return Bunch(sample = n_samples,features = n_features,data = data,target = target,
feature_names = feature_names)
# In[183]:
def ImportData(fileName):
    sampleData = load_data(fileName)
    return sampleData
# In[3]:
def Normalize(sampleData):
    minMaxScaler = preprocessing.MinMaxScaler()
    X = minMaxScaler.fit_transform(sampleData.data)
    Y = minMaxScaler.fit_transform(sampleData.target)
    return Bunch(X = X, Y = Y)
# In[4]:
#import pydotplus
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
def RandomForest(dataX, dataY,feature_names=None):#randomforest reduce dimension
    rf = RandomForestRegressor(n_estimators=6,random_state=0)
    rf.fit(dataX, dataY)
    Estimators=rf.estimators_
    # numTrees=len(Estimators)
    # print numTrees
    # for num in range(0,numTrees):
    #     dot_data = tree.export_graphviz(Estimators[num], out_file=None,filled=True, rounded=True,
    #                                     feature_names=feature_names,
    #                                     special_characters=True)
    #     graph = pydotplus.graph_from_dot_data(dot_data)
    #     graph.write_png("iris" + str(num) + ".png")
    importance = rf.feature_importances_
    print("随机森林Gini重要度为",importance)
    indices = np.argsort(importance)
    print(indices)
    return importance,indices
def ElasticNet(dataX,dataY,pre_indice):
    '''
    弹性回归得到每个特征的权重系数
    :param dataX:
    :param dataY:
    :param indices:
    :return:
    '''
    Encv=ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
                 l1_ratio=[.01, .1, .5, .9, .99],
                 max_iter=5000,random_state=0)
    Encv.fit(dataX,dataY)
    print(Encv)
    weight_coef=Encv.coef_
    print("弹性网络回归的权重系数",list(weight_coef))
    remain_indice=[]
    for index,coef in enumerate(weight_coef):
        if(coef!=0):
            remain_indice.append(index)
    print("弹性网络保留的特征序号",remain_indice)
    # Oringal_indice=[]
    # for i in remain_indice:
    #     Oringal_indice.append(pre_indice[i])
    # weight_coef=[abs(weight) for weight in weight_coef]
    # print("弹性网络回归权重系数为",weight_coef)#权重系数越大，
    # # 表明该特征对响应变量的影响越大，所以该特征的重要度越高
    # indice=np.argsort(weight_coef)
    # print(indice)
    return remain_indice

def SvrPrediction(trainX, trainY, testX):
    rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
               param_grid={"C": np.logspace(-3, 3, 7),
               "gamma": np.logspace(-2, 2, 5)})
    rbfSVR.fit(trainX, trainY)
    predictY = rbfSVR.predict(testX)
    return predictY
def modelEvaluation(testY,predictY):
    rmse=np.sqrt(mean_squared_error(testY,predictY))
    mae=mean_absolute_error(testY,predictY)
    r2 = r2_score(testY, predictY)
    return Bunch(rmse = rmse, mae = mae, r2 = r2)
# In[5]:

def aver_splitData(K):
    '''
    将数据平均划分为k，其中随机挑选k/2份用随机森林进行特征打分，
    剩余k/2使用弹性网络回归赋予特征权重系数
    :param dataX:
    :param dataY:
    :param K:
    :return:
    '''
    random.seed(0)
    k_index=[i for i in range(K)]
    RF_rdindex=random.sample(k_index,int(K/2))
    EN_rdindex=[index for index in k_index
                if index not in RF_rdindex]
    return RF_rdindex,EN_rdindex
def RF_ranking(dataX,dataY):
    feature_importance,sort_indice=RandomForest(dataX,dataY)
    return feature_importance,sort_indice
def EN_ranking(dataX,dataY):
    feature_weight,sortindice=ElasticNet(dataX,dataY)
    return feature_weight,sortindice
def Ensemble_ranking(dataX,dataY,K):
    '''
    :param dataX:
    :param dataY:
    :param K:
    :return:
    '''
    #将数据随机平均分为k份
    gama=0.4
    rf_rdindex,en_rdindex=aver_splitData(K)
    m,n=dataX.shape
    RF_Sum_importance=[0]*n
    for rf_i in rf_rdindex:
        train_X=dataX[int(rf_i*len(dataX)/K)+1:int((rf_i+1)*len(dataX)/K)]
        train_Y=dataY[int(rf_i*len(dataX)/K)+1:int((rf_i+1)*len(dataX)/K)]
        RF_importance,RF_indice=RandomForest(train_X,train_Y)
        RF_Sum_importance+=RF_importance
    RF_aver_importance=RF_Sum_importance/(K/2)
    print("随机森林-特征的平均特征重要度为",RF_Sum_importance/(K/2))
    #对求平均后的特征重要度重新从小到大排序，返回每个特征的序号
    rf_indice=np.argsort(RF_aver_importance)
    revise_feature_importance=[]
    for index in range(len(rf_indice)):
        revise_feature_importance.append((1+math.tanh(rf_indice[index]))*RF_aver_importance[index])
    print("随机森林-修改后的平均特征重要度为",revise_feature_importance)
    revise_feature_importance=np.array(revise_feature_importance)
    EN_Sum_importance = [0] * n
    for rf_j in en_rdindex:
        train_X = dataX[int(rf_j * len(dataX) / K) + 1:int((rf_j + 1) * len(dataX)/K)]
        train_Y = dataY[int(rf_j * len(dataX) / K) + 1:int((rf_j + 1) * len(dataX)/K)]
        EN_importance, EN_indice = ElasticNet(train_X, train_Y)
        EN_importance=np.array(EN_importance)
        EN_Sum_importance += EN_importance
    # EN_Sum_importance=np.array(EN_Sum_importance)
    EN_aver_importance = EN_Sum_importance/(K/2)
    print(len(EN_aver_importance))
    print("弹性网络回归-特征的平均特征重要度为", EN_Sum_importance/(K / 2))
    feature_importance=gama*revise_feature_importance+(1-gama)*EN_aver_importance
    print("集成学习特征打分为",feature_importance)
    ensemble_rankindice=np.argsort(feature_importance)
    print("集成学习特征得分升序排序的特征序号为",ensemble_rankindice)
    return feature_importance,ensemble_rankindice
def RandomForest_FS(dataX,dataY):
    '''
    :param dataX:
    :param dataY:
    :return:
    '''
    rf_importance,rf_rankindice=RF_ranking(dataX,dataY)
    kf = KFold(n_splits=5)
    pretototal_rmse=0
    running=True
    index=0
    for train_indice,test_indice in kf.split(dataX):
        rePredictY = SvrPrediction(dataX[train_indice],
                                   dataY[train_indice], dataX[test_indice])
        reResultIndex = modelEvaluation(dataY[test_indice], rePredictY)
        pretototal_rmse+=reResultIndex.rmse
        index+=1
    print(pretototal_rmse)
    # opt_rankindice=rf_rankindice
    while running:
        print("eheeee")
        # print("1",opt_rankindice)
        opt_rankindice=list(rf_rankindice)
        for i in range(1, len(opt_rankindice)):
            coorIn = [] * (len(opt_rankindice) - i)
            for j in range(i, len(opt_rankindice)):
                coorIn.append(opt_rankindice[j])
            if(len(coorIn)>0):
                total_rmse=0
                kf=KFold(n_splits=5)
                for train_index,test_index in kf.split(dataX):
                    rePredictY = SvrPrediction(dataX[train_index][:, coorIn]
                                               , dataY[train_index], dataX[test_index][:, coorIn])
                    reResultIndex = modelEvaluation(dataY[test_index], rePredictY)
                    total_rmse+=reResultIndex.rmse
                if total_rmse<=pretototal_rmse:
                    print("1111",total_rmse)
                    # del list(opt_rankindice)[i-1]
                else:
                    # print(opt_rankindice)
                    running=False
            else:
                    running=False
    # print("22222")
    print(opt_rankindice)
def Ensemble_FS(dataX,dataY,K):
    '''
    :param dataX:
    :param dataY:
    :return:
    '''

    rf_importance,rf_rankindice=Ensemble_ranking(dataX,dataY,K)
    kf = KFold(n_splits=5)
    pretototal_rmse=0
    running=True
    index=0
    for train_indice,test_indice in kf.split(dataX):
        rePredictY = SvrPrediction(dataX[train_indice],
                                   dataY[train_indice], dataX[test_indice])
        reResultIndex = modelEvaluation(dataY[test_indice], rePredictY)
        pretototal_rmse+=reResultIndex.rmse
        index+=1
    print(pretototal_rmse)
    print(rf_rankindice)
    # opt_rankindice=rf_rankindice
    # while running:
    #     print("eheeee")
    #     # print("1",opt_rankindice)
    #     opt_rankindice=list(rf_rankindice)
    for i in range(1, len(rf_rankindice)):
        coorIn = [] * (len(rf_rankindice) - i)
        for j in range(i, len(rf_rankindice)):
            coorIn.append(rf_rankindice[j])
    if(len(coorIn)>0):
        total_rmse=0
        kf=KFold(n_splits=5)
        for train_index,test_index in kf.split(dataX):
            rePredictY = SvrPrediction(dataX[train_index][:, coorIn]
                                       , dataY[train_index], dataX[test_index][:, coorIn])
            reResultIndex = modelEvaluation(dataY[test_index], rePredictY)
            total_rmse+=reResultIndex.rmse
        if total_rmse<=pretototal_rmse:
            pass
def LayerThreeSelection(threeInputData, pre_rmse, indices,feature_names):
    dataX = threeInputData.dataX[:,indices]
    m,n=dataX.shape
    dataY = threeInputData.dataY
    if m<5000 or n>40:
        importance, coorIndex = RandomForest(dataX, dataY, feature_names)
        optIndex=coorIndex
        for i in range(1, len(coorIndex)):
            coorIn = [] * (len(coorIndex) - i)
            for j in range(i, len(coorIndex)):
                coorIn.append(coorIndex[j])
            ceRmse = 0
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(dataX):
                trainX = threeInputData.dataX[train_index][:, coorIn]
                trainY = dataY[train_index]
                testX = threeInputData.dataX[test_index][:, coorIn]
                testY = dataY[test_index]
                cePredictY = SvrPrediction(trainX, trainY, testX)
                ceResultIndex = modelEvaluation(testY, cePredictY)
                ceRmse += ceResultIndex.rmse
            print(ceRmse)
            # rePredictY = SvrPrediction(trainX[:, coorIn], trainY, testX[:, coorIn])
            # reResultIndex = modelEvaluation(testY, rePredictY)
            if ceRmse <= pre_rmse:
                pre_rmse = ceRmse
                optIndex = coorIn
            else:
                break
    else:
        optIndex=ElasticNet(dataX,dataY,indices)
    outputX = threeInputData.dataX[:,optIndex]
    return Bunch(threeDataX = outputX, threeDataY = dataY, trainIndex = threeInputData.trainIndex,
    testIndex = threeInputData.testIndex,optIndex = optIndex)

# In[6]:
def feature_selection(n_samples, n_features, data, target, feature_names,username):
    print("原始特征数：",n_features)
    no_target = False
    if target == []:
        no_target = True
    inputData = Bunch(sample = n_samples, features = n_features, data = data, target = target, feature_names = feature_names)
    normalizeData = Normalize(inputData)
    kf = KFold(n_splits=5)
    from feature_selection.rw_txt import read_txt
    pre_indices = read_txt("result2.txt")
    pre_indices=pre_indices
    print("上一层保留的特征", pre_indices)
    print(len(pre_indices))
    foldNum = 1
    totalRmse = 0
    total_mae=0
    total_r2=0
    for train_index, test_index in kf.split(normalizeData.X):
        # indices=list(result)+list(expert_remain2)
        kRmse = 0
        orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
        orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
        orgPredictY = SvrPrediction(orgTrainX[:,pre_indices], orgTrainY,orgTestX[:,pre_indices])
        orgResultIndex = modelEvaluation(orgTestY, orgPredictY)
        totalRmse += orgResultIndex.rmse
        total_mae+=orgResultIndex.mae
        total_r2+=orgResultIndex.r2
    pre_rmse=totalRmse
    print("初始的模型预测的误差",totalRmse/5,total_mae/5,total_r2/5)
    oneInputData = Bunch(dataX = normalizeData.X, dataY = normalizeData.Y, trainIndex = train_index,
                         testIndex = test_index, sample = inputData.sample)
    if no_target == False:
        threeOutputData = LayerThreeSelection(oneInputData, pre_rmse, pre_indices
                                              ,inputData.feature_names)
        threeRetainIndex = [] * len(threeOutputData.optIndex)
        for i in range(0,len(threeOutputData.optIndex)):
            threeRetainIndex.append(pre_indices[threeOutputData.optIndex[i]])
        print("The three retain features are:")
        print(len(threeRetainIndex))
        print("第三层保留的特征名",feature_names[threeRetainIndex])
        print(threeRetainIndex)
        threeRetainIndex=threeRetainIndex
        filter_indices = [indice for indice in pre_indices if indice not
                         in threeRetainIndex]
        print("删除的特征为", [indice for indice in pre_indices if indice not
                         in threeRetainIndex])
        print("删掉的特征名", feature_names[filter_indices])
        kf = KFold(n_splits=5)
        final_totalRmse = 0
        final_total_mae = 0
        final_total_r2 = 0
        for train_index, test_index in kf.split(normalizeData.X):
            # indices=list(result)+list(expert_remain2)
            kRmse = 0
            orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
            orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
            orgPredictY = SvrPrediction(orgTrainX[:, threeRetainIndex], orgTrainY, orgTestX[:, threeRetainIndex])
            orgResultIndex = modelEvaluation(orgTestY, orgPredictY)
            final_totalRmse += orgResultIndex.rmse
            final_total_mae += orgResultIndex.mae
            final_total_r2 += orgResultIndex.r2
        print("最后一层后的模型预测的误差", final_totalRmse / 5, final_total_mae / 5, final_total_r2 / 5)
        filter_indices = [i for i in pre_indices if i not in threeRetainIndex]
        print(filter_indices)
        print(inputData.feature_names[threeRetainIndex])
# In[7]:
if __name__ == '__main__':
    inputData = ImportData('data_files/one_serious4.csv')
    start=time.time()
    # Normalize_data=Normalize(inputData)
    # print(Normalize_data)
    # Ensemble_ranking(Normalize_data.X,Normalize_data.Y,4)
    # feature_indice=ElasticNet(Normalize_data.X,Normalize_data.Y)
    # importance,indice1=RandomForest(Normalize_data.X,Normalize_data.Y
    #                                 ,inputData.feature_names)
    # Ensemble_FS(Normalize_data.X,Normalize_data.Y,8)
    feature_selection(inputData.sample, inputData.features,
                      inputData.data, inputData.target,
                      inputData.feature_names,"wujunming")
    print("第三层耗时",time.time()-start)

