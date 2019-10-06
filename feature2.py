import csv
import pymongo
import numpy as np
import time
from sklearn import preprocessing
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")
# from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from scipy.stats import pearsonr
import pandas as pd
import xlrd
import json
import os
from minepy import MINE
from sklearn.model_selection import KFold
# In[32]:
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
    '''
    获取数据库连接
    :return:
    '''
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
        target = np.array(target)
        target = target.reshape(-1,1)
        n_samples = data.shape[0]
        n_features = data.shape[1]
    return Bunch(sample = n_samples,features = n_features,data = data,target = target,
feature_names = feature_names)
# In[183]:
def ImportData(fileName):
    sampleData = load_data(fileName)
    return sampleData
# In[33]:
def Normalize(sampleData):
    minMaxScaler = preprocessing.MinMaxScaler()
    X = minMaxScaler.fit_transform(sampleData.data)
    Y = minMaxScaler.fit_transform(sampleData.target)
    return Bunch(X = X, Y = Y)
def excel_to_db(filename):
    '''
    excel表中的数据写入到mongodb数据库中
    :param filename:
    :return:
    '''
    db=Connectdatabase()
    account = db.weibo
    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]
    # 读取excel第一行数据作为存入mongodb的字段名
    rowstag = table.row_values(0)
    nrows = table.nrows
    # ncols=table.ncols
    # print rows
    returnData = {}
    for i in range(1, nrows):
        # 将字段名和excel数据存储为字典形式，并转换为json格式
        returnData[i] = json.dumps(dict(zip(rowstag, table.row_values(i))))
        # 通过编解码还原数据
        returnData[i] = json.loads(returnData[i])
        # print returnData[i]
        account.insert(returnData[i])
# In[34]:
def MicEvaluate(dataX,dataY,name,pre_indices):
    '''
    计算每一个条件属性与决策属性之间的最大信息系数
    :param dataX:
    :param dataY:
    :param name:
    :return:
    '''
    dataY = dataY.reshape(1, -1)[0]
    nFeatures = len(dataX[0])
    print("输入特征数为:",nFeatures)
    coorArray = [] * nFeatures
    mine = MINE(alpha=0.6, c=15)
    for i in range(0, nFeatures):
        l = [x[i] for x in dataX]
        mine.compute_score(l, dataY)
        temp = mine.mic()
        coorArray.append(abs(temp))
    print("上一层留下的每个特征的最大互信息系数", coorArray)
    coorIndex = np.argsort(coorArray)
    coorIndex_=[]
    #返回最初的特征索引
    for i in coorIndex:
        coorIndex_.append(pre_indices[i])
    coorArray = np.array(coorArray)
    print("MIC相关系数：")

    print("特征：",dict(zip(name[coorIndex_],coorArray[coorIndex])))
    name_coorArray = dict(zip(name[coorIndex_], coorArray[coorIndex]))
    return coorIndex_, coorArray,name_coorArray
# In[35]:
def CorrelationEvaluate(dataX,dataY,name,pre_indices):
    '''
    计算每一个条件属性与决策属性之间的皮尔逊相关系数
    :param dataX:
    :param dataY:
    :param name:
    :return:
    '''
    print("原始的dataY",dataY)
    # print(dataY.any())
    dataY=dataY.reshape(1,-1)[0]
    print(dataY)
    nFeatures = len(dataX[0])
    coorArray = [] * nFeatures
    for i in range(0, nFeatures):
        l = [x[i] for x in dataX]
        coor = pearsonr(l, dataY)
        coorArray.append(abs(coor[0]))
    print("上一层留下的每个特征的皮尔逊相关系数",coorArray)
    coorIndex = np.argsort(coorArray)
    coorIndex_=[]
    for i in coorIndex:
        coorIndex_.append(pre_indices[i])
    coorArray = np.array(coorArray)
    print("皮尔逊相关系数：")
    print("特征：",dict(zip(name[coorIndex_],coorArray[coorIndex])))
    name_coorArray=dict(zip(name[coorIndex_],coorArray[coorIndex]))
    return coorIndex_, coorArray,name_coorArray
def Condfea_corcoef(inputData):
    '''
    @author:wujunming
    计算条件属性之间的相关系数
    :param inputData:
    :return:
    '''
    feature_colums = inputData.feature_names
    sample = inputData.sample
    sample_index = np.arange(sample)
    df = pd.DataFrame(inputData.data, index=sample_index, columns=feature_colums[:-1])
    c = df.corr()
    isExists = os.path.exists("E:\\superalloy\\cor_coeff")
    if not isExists:
        os.makedirs("E:\\superalloy\\cor_coeff")
        print("目录创建成功")
    else:
        print("目录已存在")
    c.to_excel("E:\\superalloy\\cor_coeff\\Correlation_coeff.xlsx")

# In[36]:
def SvrPrediction(trainX, trainY, testX):
    rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
               param_grid={"C": np.logspace(-3, 3, 7),
               "gamma": np.logspace(-2, 2, 5)})
    rbfSVR.fit(trainX, trainY)
    predictY = rbfSVR.predict(testX)
    return predictY
# In[37]:
def modelEvaluation(testY,predictY):
    rmse=np.sqrt(mean_squared_error(testY,predictY))
    mae=mean_absolute_error(testY,predictY)
    r2 = r2_score(testY, predictY)
    return Bunch(rmse = rmse, mae = mae, r2 = r2)
# In[38]:
def LayerTwoSelection(twoInputData, pre_totalrmse, name,pre_indice):
    dataX = twoInputData.dataX[:,pre_indice]
    dataY = twoInputData.dataY
    Mic_coorIndex, Mic_coorArray,Mic_name_coorArray = MicEvaluate(dataX, dataY, name,pre_indice)
    Opt_MiccoorIndex=Mic_coorIndex
    for i in range(1,len(Mic_coorIndex)):
        Mic_coorIn=[]*(len(Mic_coorIndex)-i)
        for j in range(i,len(Mic_coorIndex)):
            Mic_coorIn.append(Mic_coorIndex[j])
        print(Mic_coorIn)
        ceRmse=0
        kf=KFold(n_splits=5)
        for train_index,test_index in kf.split(dataX):
            trainX=twoInputData.dataX[train_index][:,Mic_coorIn]
            trainY=dataY[train_index]
            testX=twoInputData.dataX[test_index][:,Mic_coorIn]
            testY=dataY[test_index]
            cePredictY = SvrPrediction(trainX, trainY, testX)
            ceResultIndex = modelEvaluation(testY, cePredictY)
            ceRmse += ceResultIndex.rmse
        print(ceRmse)
        if ceRmse <= pre_totalrmse:
            pre_totalrmse=ceRmse
            Opt_MiccoorIndex=Mic_coorIn
        else:
            break
    print("1",Opt_MiccoorIndex)
    print("最大信息相关系数筛选后特征数", len(Opt_MiccoorIndex))
    Per_coorIndex, Per_coorArray,Cor_name_coorArray = CorrelationEvaluate(dataX, dataY, name, pre_indice)
    df=pd.Series(Per_coorArray,index=Cor_name_coorArray)
    df.to_excel("correlation.xlsx")
    Opt_PercoorIndex = Per_coorIndex
    for i in range(1, len(Per_coorIndex)):
        Per_coorIn = [] * (len(Per_coorIndex) - i)
        for j in range(i, len(Per_coorIndex)):
            Per_coorIn.append(Per_coorIndex[j])
        print(Per_coorIn)
        ceRmse = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(dataX):
            trainX =twoInputData.dataX[train_index][:, Per_coorIn]
            trainY = dataY[train_index]
            testX = twoInputData.dataX[test_index][:, Per_coorIn]
            testY = dataY[test_index]
            cePredictY = SvrPrediction(trainX, trainY, testX)
            ceResultIndex = modelEvaluation(testY, cePredictY)
            ceRmse += ceResultIndex.rmse
        print(ceRmse)
        if ceRmse <= pre_totalrmse:
            pre_totalrmse = ceRmse
            Opt_PercoorIndex = Per_coorIn
        else:
            break
    print("2",Opt_PercoorIndex)
    print("皮尔逊相关系数筛选后特征数",len(Opt_PercoorIndex))
    Per_and_Mic=[index for index in Opt_PercoorIndex if index in Opt_MiccoorIndex]
    print("两个集合取交集",Per_and_Mic)
    print(len(Per_and_Mic))
    Per_or_Mic=list(set(Per_coorIndex).union(set(Mic_coorIndex)))
    print("两个集合取并集", Per_or_Mic)
    print(len(Per_or_Mic))
    Per_xor_Mic = list(set(Per_coorIndex) ^ (set(Mic_coorIndex)))
    print("两个集合取异或", Per_xor_Mic)
    result_set=[Per_and_Mic,Per_or_Mic]
    print("结果集为", result_set)
    min_Rmse=0
    opt_result_index=Per_and_Mic
    for result_index in result_set:
        kf = KFold(n_splits=5)
        ceRmse=0
        for train_index, test_index in kf.split(dataX):
            trainX = twoInputData.dataX[train_index][:,result_index]
            trainY = dataY[train_index]
            testX = twoInputData.dataX[test_index][:,result_index]
            testY = dataY[test_index]
            cePredictY = SvrPrediction(trainX, trainY, testX)
            ceResultIndex = modelEvaluation(testY, cePredictY)
            ceRmse += ceResultIndex.rmse
        print(ceRmse)
        if ceRmse <= min_Rmse:
            min_Rmse=ceRmse
            opt_result_index=result_index

        else:
            continue
    print("最终获得的特征子集为",opt_result_index)
    print("删除的特征为",[indice for indice in pre_indice if indice not
     in opt_result_index])
    return Bunch(opt_result_index=opt_result_index,
                 cor_coef=Cor_name_coorArray,mic_coef=Mic_name_coorArray)
    # coefficient1 = coorArray1
    # coorIndex2, coorArray2 = CorrelationEvaluate(dataX, dataY, name)
    # coefficient = coorArray2
    # for i in range(1, len(coorIndex)):
    #     coorIn = [] * (len(coorIndex) - i)
    #     for j in range(i, len(coorIndex)):
    #         coorIn.append(coorIndex[j])
    #     cePredictY = SvrPrediction(trainX[:, coorIn], trainY, testX[:, coorIn])
    #     ceResultIndex = modelEvaluation(testY, cePredictY)
    #     if ceResultIndex.rmse <= ceRmse:
    #         ceRmse = ceResultIndex.rmse
    #         optIndex = coorIn
    #         optResultIndex = ceResultIndex
    #     else:
    #         print(coorArray[j])
    #         break
    # ficor_threshold=coorArray[j]
    # outputX = dataX[:, optIndex]
    # '''
    # coefficient:每个特征的相关系数
    # i_threshold:初始的相关系数阈值
    # f_threshold:最终的相关系数阈值
    # '''
    # return Bunch(twoDataX = outputX, twoDataY = dataY, trainIndex = twoInputData.trainIndex,
    #              testIndex = twoInputData.testIndex, resultIndex = optResultIndex, optIndex = optIndex,
    #              coefficient = coefficient,f_threshold=ficor_threshold)
# In[39]:
# def draw_heatmap(data,xlabels,ylabels):
#     cmap = cm.get_cmap('rainbow', 1000)
#     figure=plt.figure(facecolor='w')
#     ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])
#     ax.set_yticks(range(len(ylabels)))
#     ax.set_yticklabels(ylabels)
#     ax.set_xticks(range(len(xlabels)))
#     ax.set_xticklabels(xlabels)
#     vmax=data[0][0]
#     vmin=data[0][0]
#     for i in data:
#         for j in i:
#             if j>vmax:
#                 vmax=j
#             if j<vmin:
#                 vmin=j
#     map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
#     cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
#     plt.show()
#     # plt.savefig("1.png")
# def obtainpre_resultindice(filename):
#     with open(filename,"r") as fr:
#         fr_readline=fr.read()
#         indice_list=fr_readline.strip().split(",")
#         indice_list=list(map(int,indice_list))
#         return indice_list
def feature_selection(n_samples, n_features, data, target, feature_names,username):
    '''
    :param n_samples: 样本数
    :param n_features: 属性个数
    :param data: 输入X
    :param target: 输出Y
    :param feature_names: 特征名
    :param username: 使用该算法的用户名
    :return:
    '''
    # feature_importance = [1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0]
    print("---------------")
    print("原始特征数：",n_features)
    no_target = False
    if target == []:
        no_target = True
    inputData = Bunch(sample = n_samples,
                      features = n_features,
                      data = data,
                      target = target,
                      feature_names = feature_names)
    normalizeData = Normalize(inputData)
    kf = KFold(n_splits=5)
    from feature_selection.rw_txt import read_txt
    pre_resultindex=read_txt("result1.txt")
    print("上一层保留的特征",pre_resultindex)
    print(len(pre_resultindex))
    pre_total_rmse=0
    total_Mae = 0
    total_r2 = 0
    for train_index, test_index in kf.split(inputData.data):
        # print(expert_remain1)
        # kRmse = 0
        orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
        orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
        orgPredictY = SvrPrediction(orgTrainX[:,pre_resultindex], orgTrainY,orgTestX[:,pre_resultindex])
        orgResultIndex = modelEvaluation(orgTestY, orgPredictY)
        pre_total_rmse+=orgResultIndex.rmse
        total_Mae += orgResultIndex.mae
        total_r2 += orgResultIndex.r2
    print("初始的模型预测的均方根误差rmse为%lf,平均平方差mae为%lf,"
          "r2为%lf" %(pre_total_rmse/5, total_Mae / 5, total_r2 / 5))
    oneInputData = Bunch(dataX = normalizeData.X, dataY = normalizeData.Y, trainIndex = train_index,
                         testIndex = test_index, sample = inputData.sample)
    if no_target == False:
        twoOutputData = LayerTwoSelection(oneInputData, pre_total_rmse
                                          ,inputData.feature_names,pre_resultindex)
        twoRetainIndex=twoOutputData.opt_result_index
        print("第二层特征选择保留的特征final features",twoRetainIndex)
        print("第二层保留的的特征数目",len(twoRetainIndex))
        print("第二层保留的特征名", feature_names[twoRetainIndex])
        print("第二层每个特征的皮尔逊相关系数为",twoOutputData.cor_coef)
        print("第二层每个特征的最大互信息相关系数为", twoOutputData.mic_coef)
        from feature_selection.rw_txt import write_txt
        twoRetainIndex = map(str, list(twoRetainIndex))
        write_txt("result2.txt",twoRetainIndex)
# In[40]:
if __name__ == '__main__':
    inputData = ImportData("data_files/one_serious4.csv")
    Condfea_corcoef(inputData)
    start=time.time()
    feature_selection(inputData.sample,
                      inputData.features,
                      inputData.data,
                      inputData.target,
                      inputData.feature_names,
                      "wujunming")
    print("第二层耗时",time.time()-start)
