import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
import globe as globe
import mdlmngmnt as mdlmngmnt


figurenr = 0 #This figure is passed between the modules in order to keep incrementing the number of figures 
             #that are plotted


def cleanandview(filename, figurenr, doplots = True): #This function plots the before and after interpolation 
                #time series for the .csv file in filename.  The function also removes the Date column, and 
                #reads the file into a Pandas DataFrame object which is returned.
    df = pd.read_csv(filename)
    #print(df.head())
    df = df.drop('Date', axis = 1)
    #print(df.head())
    if doplots:
        plt.figure(figurenr)
        figurenr += 1
        df.plot()
    dfclean = df.interpolate()
    if doplots:
        plt.figure(figurenr)
        figurenr += 1
        dfclean.plot()
    return dfclean


def scaleandpca(xdata):  #PCA is done in this function.  Features most aligned with the identified components are 
                         #returned.
    xdata_scaled = preprocessing.scale(xdata)
    print(xdata_scaled)
    pca = PCA(n_components=globe.NrPCAComponents)
    pca.fit_transform(xdata_scaled)
    print(pca.components_)
    #print(type(pca.components_))
    principlefeatures = list()
    for i in range(globe.NrPCAComponents):
        featureindicesarray = \
            np.argpartition(pca.components_[i,:], -globe.NrFeaturesPerComponent)[-globe.NrFeaturesPerComponent:]
        featureindiceslist = list(featureindicesarray)
        #print(featureindiceslist)
        for f in featureindiceslist:
            principlefeatures.append(f)
    print(principlefeatures)
    return principlefeatures


def onehotandview(filename, figurenr, doplots = True): #The categorical time series are transfored into 
                #one-hot-encoded space in this function, and returned.
    df = pd.read_csv(filename)
    #print(df.head())
    df = df.drop('Date', axis = 1)
    #print(df.head())
    if doplots:
        plt.figure(figurenr)
        figurenr += 1
        df.plot()
    enc = OneHotEncoder(sparse=False)
    catonehot = enc.fit_transform(df.values)
    #print(catonehot)
    return catonehot


doplots = False
plotmodelresults = True
plotbestports = False
xdfclean = cleanandview('X.csv',figurenr, doplots)
#print(xdfclean.values) 
ydfclean = cleanandview('y.csv',figurenr, doplots)
catdata = onehotandview('categorical.csv',figurenr, doplots)
principlefeatures = scaleandpca(xdfclean.values)
#fitLinearRegression(catonehot, ydfclean['y_3'].values, range(catonehot.shape[1]), 'One hot categorical', plotmodelresults)
i = 0
models = list()
targetstomodel = ydfclean.columns
#targetstomodel = ['y_3']
for label in targetstomodel:
    model = mdlmngmnt.mdlmngmnt(plt, figurenr, xdfclean.values, catdata, \
            principlefeatures, ydfclean[label].values, plotmodelresults, doplots, label)
    model.fitbestlinearmodel()
    models.append(model)
for model in models:
    model.bestportprint()
if doplots or plotmodelresults or plotbestports: plt.show()