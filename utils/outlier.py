import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

def injectPoint(data, first:int=100, second:int=None):
    """Inject Global outlier(s) into data

    Args:
        data (series): the data that the outlier(s) will be injected into
        first (int): point in the data where the first outlier will be injected (defaults to 100)
        second (int): point in the data where the second outlier will be injected (optional defaults to None)

    Returns:
        data with injected outlier
    """
    u, sd, mad = np.mean(data), np.std(data), np.median(np.absolute(data - np.median(data)))
    print(f"μ: {u}   σ: {sd}    MAD: {mad}   first: {first}")
    if first == 0:
        data[first:first+2] = [u+(sd*4),u]
    elif first == len(data) - 1:
        data[first-1:first+1] = [u,u+(sd*4)]
    else: 
        data[first-1:first+2] = [u,u+(sd*4),u]
    if second != None:
        data[second-1:second+2] = [u,u-(sd*4),u]
    return data

def injectConditional(data1, data2, first:int=100, second:int=None):
    """Inject Conditional (also called Contextual) outlier(s) into data

    Args:
        data1 (series): the first data series that the outlier(s) will be injected into
        data2 (series): the second data series that the outlier(s) will be injected into
        first (int): point in the data where the first outlier will be injected (defaults to 100)
        second (int): point in the data where the second outlier will be injected (optional defaults to None)

    Returns:
        data1, data2 with injected outlier 1 SD in the opposite direction for each data series
    """
    u1, sd1 = np.std(data1), np.mean(data1)
    u2, sd2 = np.std(data2), np.mean(data2)
    # def change(data, sd):
    # 	data[100:101] = [data1[100]+(sd1*2*np.sign(r2[0,1]))]
    r2 = np.corrcoef(data1, data2)
    data1[first:first+1] = [data1[first]+(sd1*1*-np.sign(r2[0,1]))]
    data2[first:first+1] = [data2[first]+(sd2*1*np.sign(r2[0,1]))]
    if second != None:
        data1[second:second+1] = [data1[second]+(sd1*1*-np.sign(r2[0,1]))]
        data2[second:second+1] = [data2[second]+(sd2*1*np.sign(r2[0,1]))]
    return data1, data2

def injectCollective(data, first:int=100, second:int=None):
    """Inject Collective outlier(s) (10 unchanged datapoints) into data. Note because of the length of
        collective outlier

    Args:
        data (series): the data that the outlier(s) will be injected into
        first (int): point in the data where the first outlier will be injected (defaults to 100)
        second (int): point in the data where the second outlier will be injected (optional defaults to None)

    Returns:
        data with injected outlier
    """
    u, sd = np.mean(data), np.std(data)
    print(f"μ: {u}   σ: {sd}  first: {first}")
    level = u + sd*0.25
    first = 12 if first < 12 else first
    first = len(data)-12 if first > len(data)-12 else first
    data[first-6:first+6] = [u,level,level,level,level,level,level,level,level,level,level,u]
    if second != None:
        level = u - (sd*2)
        data[second-6:second+6] = [u,level,level,level,level,level,level,level,level,level,level,u]
    return data

def arma_1_1_sample(nsample=300):
    """return a stationary sample from an ARMA(1,1) process 

    Args:
        nsample (int): number of data points to return (defaults to 300)

    Returns:
        sample data
    """
    ar1 = np.array([1, -0.5])
    ma1 = np.array([1, 0.25])
    AR_object1 = ArmaProcess(ar1, ma1)
    # print(AR_object1.isstationary, AR_object1.isinvertible)
    simulated_data = AR_object1.generate_sample(nsample=nsample)
    return simulated_data

def injectOutlier(df: pd.DataFrame, field: str, outlier_type: str, anomaly: int=None) -> (pd.DataFrame, int):
    """return a dataframe with the appropriate outlier_type inject at the anomaly point
    If the type is all three, then Global, Contextual, and Collective are injected at 75, 150 and 225.

    Args:
        df (DataFrame): DataFrame with the data. Must be 240 or longer if outlier_type is 'All Three'
        field (str): the field in the df DataFrame to have outlier added to it
        outlier_type (str): the type of outlier to inject ('Global', 'Contextual', 'Collective', 'All Three')
        anomaly (int): defaults to random but can be specified (except for All Three which is 75, 150, 225)
    Returns:
        (DataFrame, anomaly)
    """
    data = None
    if anomaly == None:
        anomaly = np.random.randint(0, len(df))        
    if outlier_type == "Global":
        data = df[[field]]
        data[field] = injectPoint(df[field], first=anomaly)
        data=data.reset_index()
        data[[field,'time']] = scaler.fit_transform(data[[field,'index']])
    elif outlier_type == "Contextual":
        data1, data2 = df[field], df[field + "_TPX"]
        data1, data2 = injectConditional(data1,data2, first=anomaly)
        data = pd.DataFrame({field: data1, field+'_tpx': data2})
        data=data.reset_index()
        data[[field, field+'_tpx', 'time']] = scaler.fit_transform(data[[field, field+'_tpx','index']])
    elif outlier_type == "Collective":
        anomaly = np.random.randint(12, len(df) - 12)
        data = df[[field]]
        data[field] = injectCollective(data[field], first=anomaly)
        data["x1"] = data[field]
        data[field] = data[field].rolling(12).std(ddof=0)
        data[field] = data[field].fillna(data[field].mean())
        data[field] = data[field].apply(lambda x: max(x,0.001))
        data[field] = np.log(3/data[field])
        data=data.reset_index()
        data[[field, 'x1', 'time']] = scaler.fit_transform(data[[field, 'x1','index']])
    elif outlier_type == "All Three":
        if len(df) < 240:
            raise ValueError("DataFrame does not have enough rows for outlier_type: 'All Three'. Must be 240 rows or longer.")
        data_tpx = df[field + "_TPX"]
        data1, data_tpx = injectConditional(injectCollective(injectPoint(df[field], first=75), first=225), data_tpx, first=150)
        data_rolling = data1.rolling(12).std(ddof=0)
        data_rolling2 = data_rolling.fillna(data_rolling.mean())
        data_rolling2 = data_rolling2.apply(lambda x: max(x,0.001))
        data_rolling_norm = np.log(3/data_rolling2)
        data = pd.DataFrame({field: data1, field+'_tpx': data_tpx, 'rolling': data_rolling_norm})
        data=data.reset_index()
        data[[field, field+'_tpx', 'rolling', 'time']] = scaler.fit_transform(data[[field, field+'_tpx', 'rolling', 'index']])
    else:
        raise ValueError("injectOutlier incorrect outlier_type. Acceptable outlier_types ('Global'| 'Contextual'|'Collective'|'All Three')")
    
    return (data, anomaly)

if __name__ == '__main__':
    data = arma_1_1_sample()
    data1 = pd.DataFrame({"RETURN_NORM": data})
    data2 = injectOutlier(data1,"RETURN_NORM","Global", anomaly=50)
    print(f"data2:\r\n{data2[:5]}")