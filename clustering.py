import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

def load_data():
    dataset = pd.read_csv("climatedata.csv")
    dataset.fillna(0)
    columns = []
    for i in dataset.columns:
        columns.append(i)
    print(columns)
    dataset = dataset.dropna()
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    return  dataset;

dataset = load_data()

def clusters():

    X = dataset[['1993 [YR1993]', '1994 [YR1994]', '1995 [YR1995]', '1996 [YR1996]', '1997 [YR1997]',
                '1998 [YR1998]', '1999 [YR1999]', '2000 [YR2000]', '2001 [YR2001]', '2002 [YR2002]',
                '2003 [YR2003]', '2004 [YR2004]', '2005 [YR2005]', '2006 [YR2006]', '2007 [YR2007]',
                '2008 [YR2008]', '2009 [YR2009]', '2010 [YR2010]', '2011 [YR2011]', '2012 [YR2012]',
                '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]','2016 [YR2016]', '2017 [YR2017]',
                '2018 [YR2018]', '2019 [YR2019]', '2020 [YR2020]', '2021 [YR2021]']]
    Y = dataset['2022 [YR2022]']
    X = np.array(X)
    from sklearn.cluster import KMeans
    wcss_list= []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
        kmeans.fit(X)
        wcss_list.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss_list)
    plt.title('The Elobw Method Graph')
    plt.xlabel('Number of clusters(k)')
    plt.ylabel('wcss_list')
    plt.show()
    plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster
    plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster
def exponential(x, a, b):
    return a*np.exp(b*x)
def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))
    return results
adjR(dataset['1994 [YR1994]'],dataset['2022 [YR2022]'], 1)
adjR(dataset['1994 [YR1994]'],dataset['2022 [YR2022]'], 2)
adjR(dataset['1994 [YR1994]'], dataset['2022 [YR2022]'], 3)
adjR(dataset['1994 [YR1994]'], dataset['2022 [YR2022]'], 4)
adjR(dataset['1994 [YR1994]'], dataset['2022 [YR2022]'], 5)
model4 = np.poly1d(np.polyfit(dataset['1994 [YR1994]'],dataset['2022 [YR2022]'], 4))
polyline = np.linspace(1, 15, 50)
plt.scatter(dataset['1994 [YR1994]'],dataset['2022 [YR2022]'] )
plt.plot(polyline, model4(polyline), '--', color='red')
plt.show()
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 





