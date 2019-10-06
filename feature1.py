import csv
import pymongo
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
import json
import time
# In[35]:
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
def read_data(filename):
    file_list=filename.strip().split(".")
    if file_list[-1]=="xlsx":
        data_frame=pd.read_excel(filename)
    elif file_list[-1]=="csv":
        data_frame=pd.read_csv(filename)
    else:
        print("this is vasp/clf file")
    dataset=data_frame.as_matrix()
    # data_set=[]
    # for data in dataset:
    #     data=map(float,data)
    #     data_set.append(data)
    # print(data_set)
    data_set=np.asarray(dataset,dtype=np.float)
    n_samples=dataset.shape[0]
    print(n_samples)
    n_features=dataset.shape[1]-1
    data=np.asarray(dataset[:,:-1],dtype=np.float)
    target=np.asarray(dataset[:,-1],dtype=np.float)
    feature_names=[column for column in data_frame]
    feature_names=np.array(feature_names)
    return Bunch(sample=n_samples, features=n_features, data=data, target=target,
                 feature_names=feature_names)
#专家给出的特征重要度[1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0]
def load_data(filename):
    with open(filename) as f:
        data_file = csv.reader(f)
        data = []
        target = []
        temp = next(data_file)
        feature_names = np.array(temp)
        print(feature_names)
        for i,d in enumerate(data_file):
            temp=[]
            for j in d:
                if j=='':
                    j=0
                temp.append(j)
            data.append(np.asarray(temp[:-1],dtype = np.float))
            target.append(np.asarray(d[-1],dtype = np.float))
        # random_index=np.random.randint(0,len(data))
        data = np.array(data)
        target = np.array(target)
        target=target.reshape(-1,1)
        n_samples = data.shape[0]
        print(n_samples)
        n_features = data.shape[1]
        print(n_features)
    return Bunch(sample = n_samples,features = n_features,data = data,target = target,
feature_names = feature_names)
# In[183]:
def ImportData(fileName):
    sampleData = load_data(fileName)
    return sampleData
# In[36]:
def Normalize(sampleData):
    minMaxScaler = preprocessing.MinMaxScaler()
    X = minMaxScaler.fit_transform(sampleData.data)
    Y = minMaxScaler.fit_transform(sampleData.target)
    return Bunch(X = X, Y = Y)
# In[37]:
def ValueCounts(inputX, threshold, sample, name):
    counts=[]
    arrayX=np.array(inputX)
    for i in range(len(inputX[0])):
        counts.append(pd.value_counts(arrayX[:,i]).values[0]*100.0/sample)
    indices = np.argsort(counts)[::-1]#返回counts中数组值降序排列的索引
    filtered = list(filter(lambda x: counts[x] > threshold, indices))
    indices = list(filter(lambda x: counts[x] <= threshold, indices))
    counts = np.array(counts)
    # print "稀疏系数: "
    # print "过滤特征: ",dict(zip(name[filtered],counts[filtered]))
    # print "余下特征: ",dict(zip(name[indices],counts[indices]))
    return Bunch(indices = indices, filtered = filtered, counts = counts)
# In[38]:
def SparsityEvaluate(inputX, threshold,pre_indices):
    # print threshold
    vt = VarianceThreshold()
    vt.fit_transform(inputX)
    importance = vt.variances_
    indices = np.argsort(importance)[::-1]
    filtered = list(filter(lambda x: importance[x] <= threshold, indices))
    indices = list(filter(lambda x: importance[x] > threshold, indices))
    importance = np.array(importance)
    indices_=[]
    print("1",pre_indices,len(pre_indices))
    print("2",indices,len(indices))
    for i in indices:
        indices_.append(pre_indices[i])
    print("每个特征的方差分数",importance)
    return Bunch(indices = indices_, filtered = filtered), importance, filtered, indices
# In[39]:
def SvrPrediction(trainX, trainY, testX):
    rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
               param_grid={"C": np.logspace(-3, 3, 7),
               "gamma": np.logspace(-2, 2, 5)})
    rbfSVR.fit(trainX, trainY)
    predictY = rbfSVR.predict(testX)
    return predictY
# In[40]:
def modelEvaluation(testY,predictY):
    rmse=np.sqrt(mean_squared_error(testY,predictY))
    mae=mean_absolute_error(testY,predictY)
    r2 = r2_score(testY, predictY)
    return Bunch(rmse = rmse, mae = mae, r2 = r2)
# In[41]:
def LayerOneSelection(oneInputData,total_rmse,name,inputdata):
    v_threshold=95
    threshold=0.01
    running=True
    #没有经过第一层特征选择前的RMSE
    seRmse=total_rmse
    # optResultIndex=resultIndex
    outputX=oneInputData.dataX
    # trainX, trainY = oneInputData.dataX[oneInputData.trainIndex], oneInputData.dataY[oneInputData.trainIndex]
    # testX, testY = oneInputData.dataX[oneInputData.testIndex], oneInputData.dataY[oneInputData.testIndex]
    indices_all = [i for i in range(0,len(oneInputData.dataX[0]))]#所有特征的索引
    sIndex = ValueCounts(inputdata.data, v_threshold, oneInputData.sample, name)
    v_indices = np.array(sIndex.indices)
    sparse_coefficent=dict(zip(name[indices_all],sIndex.counts))

    df=pd.Series(sIndex.counts,index=name[0:-1])
    df.to_excel("sparse.xlsx")
    sparse_coefficent=json.dumps(sparse_coefficent)
    sparse_coefficent=json.loads(sparse_coefficent)
    print(sparse_coefficent)
    indices = v_indices #去除离散性特征后保留的特征索引
    # outputX = oneInputData.dataX[:, v_indices]
    print(indices)
    print(name[indices])
    seIndex1, importance_all, filtered1, ind1=SparsityEvaluate(oneInputData.dataX[:,indices],threshold,indices)
    df1=pd.Series(importance_all,index=name[indices])
    df1.to_excel("sparse2.xlsx")
    while running:
        seIndex, importance, filtered, ind = SparsityEvaluate(oneInputData.dataX[:,indices], threshold,indices)
        print(seIndex.indices)
        dataLen = len(seIndex.indices)
        if dataLen > 0:
            seTotal_rmse = 0
            kf = KFold(oneInputData.sample, n_folds=5)
            for train_index, test_index in kf:
                # seTrainX = oneInputData.dataX[train_index, v_indices[seIndex.indices]]
                seTrainData=oneInputData.dataX[train_index]
                seTrainX=seTrainData[:,seIndex.indices]
                seTestData = oneInputData.dataX[test_index]
                seTestX = seTestData[:,seIndex.indices]
                # seTestX = oneInputData.dataX[test_index, v_indices[seIndex.indices]]
                seTrainY = oneInputData.dataY[train_index]
                seTestY = oneInputData.dataY[test_index]
                sePredictY = SvrPrediction(seTrainX, seTrainY, seTestX)
                seResultIndex = modelEvaluation(seTestY,sePredictY)
                seTotal_rmse+=seResultIndex.rmse
            print("hello",seTotal_rmse)
            if seTotal_rmse <= seRmse:
                seRmse = seTotal_rmse
                # optResultIndex = seResultIndex
                threshold = threshold + 0.01
                outputX = oneInputData.dataX[:,seIndex.indices]
                indices = seIndex.indices
            else:
                running = False
        else:
            running = False
    criticalthreshold=threshold-0.01
    print("111111",indices)
    print(criticalthreshold)
    # print("方差:")
    filter_indices=[i for i in indices_all if i not in indices ]
    # print "特征:",dict(zip(name[v_indices],importance[v_indices]))
    return Bunch(oneDataX = outputX, oneDataY = oneInputData.dataY, indices = indices
                 ,filter_indices=filter_indices,
                 coefficient1 = sIndex.counts,
                  i_threshold=v_threshold,invar_threshold=threshold,
                 fivar_threshold=criticalthreshold,sparse_coefficent=sparse_coefficent)
# In[42]:
'''
feature_imporatance:用户或专家给出的特征重要度[1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0]
'''
def feature_selection(n_samples, n_features, data, target, feature_names,username):
    # feature_importance=[1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0]
    print("原始特征数：",n_features)
    no_target = False
    if target == []:
        no_target = True
    inputData = Bunch(sample = n_samples,
                      features = n_features,
                      data = data, target = target,
                      feature_names = feature_names)
    normalizeData = Normalize(inputData)
    #做5折交叉验证
    kfold_number=5
    kf = KFold(inputData.sample, n_folds = kfold_number)
    totalRmse = 0
    total_Mae=0
    total_r2=0
    for train_index, test_index in kf:
        kmse=0
        print(train_index, test_index)
        #划分训练集和测试集
        orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
        orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
        orgPredictY = SvrPrediction(orgTrainX, orgTrainY,orgTestX)
        orgResultIndex = modelEvaluation(orgTestY, orgPredictY)
        totalRmse+=orgResultIndex.rmse
        total_Mae +=orgResultIndex.mae
        total_r2+=orgResultIndex.r2
    print("初始的模型预测的均方根误差rmse为%lf,平均平方差mae为%lf,"
          "r2为%lf"%(totalRmse/kfold_number,total_Mae/kfold_number,total_r2/kfold_number))
    oneInputData=Bunch(dataX=normalizeData.X,dataY=normalizeData.Y,sample=inputData.sample)
    # oneInputData = Bunch(dataX = normalizeData.X, dataY = normalizeData.Y,
    #                      trainIndex = train_index,
    #                      testIndex = test_index,
    #                      sample = inputData.sample)
    if no_target == False:
        oneOutputData=LayerOneSelection(oneInputData,totalRmse,inputData.feature_names,inputData)
        # oneOutputData = LayerOneSelection(oneInputData, orgResultIndex, inputData.feature_names)
        print("The first retained features are:")
        onereainindices = oneOutputData.indices
        print("保留的特征为",list(onereainindices))
        print("保留的特征数",len(onereainindices))
        onereainindices=map(str,list(onereainindices))
        from feature_selection import rw_txt
        rw_txt.write_txt("result1.txt", onereainindices)
        print("筛选掉的特征为",oneOutputData.filter_indices)
        print("筛掉的特征名",feature_names[oneOutputData.filter_indices])
        print(len(oneOutputData.indices))
        print (inputData.feature_names[oneOutputData.indices])
        return
# In[43]:
if __name__ == '__main__':
    inputData = ImportData('data_files/one_serious4.csv')
    start=time.time()
    feature_selection(inputData.sample, inputData.features
                      , inputData.data,
                      inputData.target,
                      inputData.feature_names,"wujunming")
    print("第一层耗时",time.time()-start)