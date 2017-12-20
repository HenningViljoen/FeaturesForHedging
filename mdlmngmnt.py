import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from scipy.stats import pearsonr
from scipy.optimize import nnls
import subportfolio as subportfolio
import hedgeregressor as hedgeregressor
import globe as globe
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class mdlmngmnt(object):

    def __init__(self, plt, figurenr, xdata, catdata, principlefeatures, ydata, plotmodelresults, doplots, ydatalabel):
        self.plt = plt
        self.figurenr = figurenr
        self.xdata = xdata
        self.catdata = catdata
        self.principlefeatures = principlefeatures
        self.ydata = ydata
        self.plotmodelresults = plotmodelresults
        self.doplots = doplots
        self.listofsubportfolios = list()
        self.ydatalabel = ydatalabel
        self.bestport = None


    def fit_ridgecv(self):
        ridgecv = RidgeCV(alphas = (100,101))
        ridgecv.fit(self.xdata, self.ydata)
        #print(ridgecv.coef_)
        #print(ridgecv.alpha_)
        #print(ridgecv.score(xdata, ydata))
        yridgecv = ridgecv.predict(self.xdata)
        if self.doplots:
            self.plt.figure(self.figurenr)
            pd.DataFrame(yridgecv).plot()
            #ridgecv0.fit(xdfclean)
        return ridgecv


    def fit_lassocv(self):
        lassocv = LassoCV()
        lassocv.fit(self.xdata, self.ydata)
        print(lassocv.coef_)
        print(lassocv.alpha_)
        print(lassocv.score(self.xdata, self.ydata))
        ind = np.argpartition(lassocv.coef_, -globe.SecuritiesPerBasket)[-globe.SecuritiesPerBasket:]
        #print(ind)
        indsorted = np.sort(ind)
        print('lasso top indices:')
        print(indsorted)
        y0lassocv0 = lassocv.predict(self.xdata)
        if self.doplots:
            self.plt.figure(self.figurenr)
            pd.DataFrame(y0lassocv0).plot()


    def f_test_for_feature_selection(self):  #applies F test to each feature and returns 5 higest ranked features.
        f_test, _ = f_regression(self.xdata, self.ydata, center = False)
        f_test /= np.max(f_test)
        #print(f_test)
        ind = np.argpartition(f_test, -globe.SecuritiesPerBasket)[-globe.SecuritiesPerBasket:]
        indsorted = list(np.sort(ind))
        print('f_test top indices:')
        print(indsorted)
        print('f_test top values:')
        print(f_test[indsorted])
        return indsorted


    def mutual_info_for_feature_selection(self): #applies Mutual Information regressor
                                                 # to each feature and returns 5 higest ranked features.
        mi = mutual_info_regression(self.xdata, self.ydata)
        mi /= np.max(mi)
        #print(mi)
        ind = np.argpartition(mi, -globe.SecuritiesPerBasket)[-globe.SecuritiesPerBasket:]
        indsorted = list(np.sort(ind))
        print('mi top indices:')
        print(indsorted)
        print('mi top values:')
        print(mi[indsorted])
        return indsorted


    def pearson_r_for_feature_selection(self, nrfeatureswanted): #computes Pearson r correlation coefficient for 
                                #each feature and returns nrfeatureswanted higest ranked features.
        nrfeatures = self.xdata.shape[1]
        #print(nrfeatures)
        pearsonarray = np.zeros(nrfeatures)
        for i in range(nrfeatures):
            r = pearsonr(self.xdata[:,i], self.ydata)
            #print(r)
            pearsonarray[i] = r[0]
        ind = np.argpartition(pearsonarray, -nrfeatureswanted)[-nrfeatureswanted:]
        indsorted = list(np.sort(ind))
        print('pearson_r top indices:')
        print(indsorted)
        print('pearson_r top values:')
        print(pearsonarray[indsorted])
        return indsorted


    def randomforrests_for_feature_selection(self, nrfeatureswanted=globe.SecuritiesPerBasket):
                #Fits through corss-validation a Random Forrest regressor to the target and whole feature space,
                #and then finds the most important features for this regressor and returns the nrfeatureswanted
                #higest ranked indices in the feature space.
        nrfeatures = self.xdata.shape[1]
        #print(nrfeatures)
        pearsonarray = np.zeros(nrfeatures)
        rf = RandomForestRegressor()
        cv = TimeSeriesSplit()
        model = GridSearchCV(rf, {'min_samples_leaf': range(1,10)}, cv=cv)
        model.fit(self.xdata,self.ydata)
        rfr2 = model.score(self.xdata,self.ydata)
        print('rfr2: ' + str(rfr2))
        #print(model.best_estimator_.feature_importances_)
        ind = np.argpartition(model.best_estimator_.feature_importances_, -nrfeatureswanted)[-nrfeatureswanted:]
        indsorted = list(np.sort(ind))
        print('RandomForest top indices:')
        print(indsorted)
        print('RandomForesttop values:')
        print(model.best_estimator_.feature_importances_[indsorted])
        return indsorted


    def fitLinearRegression(self, indeces, sourceofindeces):
        lmodel = LinearRegression(fit_intercept=False)
        X = self.xdata[:, indeces]
        lmodel.fit(X, self.ydata)
        port = subportfolio.subportfolio(lmodel, X, indeces, self.ydata, 'LinearRegression', sourceofindeces, self.plotmodelresults, \
            self.plt, self.catdata)
        port.evaluatemodelaccuracy(self.figurenr, self.listofsubportfolios)


    def fitRidgeCV(self, indeces, sourceofindeces):
        cv = TimeSeriesSplit()
        lmodel = RidgeCV(cv=cv)
        X = self.xdata[:, indeces]
        lmodel.fit(X, self.ydata)
        print('alpha: ' + str(lmodel.alpha_))
        port = subportfolio.subportfolio(lmodel, X, indeces, self.ydata, 'RidgeCV', sourceofindeces, \
            self.plotmodelresults, self.plt, self.catdata)
        port.evaluatemodelaccuracy(self.figurenr, self.listofsubportfolios)


    def fitpositiveLassoCV(self, indeces, sourceofindeces):
        cv = TimeSeriesSplit()
        lmodel = LassoCV(precompute=True, fit_intercept=False, max_iter=1000, cv=cv,\
            positive=True, random_state=9999, selection='random')  #alpha=0.0001
        X = self.xdata[:, indeces]
        lmodel.fit(X, self.ydata)
        print('alpha: ' + str(lmodel.alpha_))
        port = subportfolio.subportfolio(lmodel, X, indeces, self.ydata, 'positiveLassoCV', sourceofindeces, \
            self.plotmodelresults, self.plt, \
            self.catdata)
        port.evaluatemodelaccuracy(self.figurenr, self.listofsubportfolios)


    def fitnnls(self, indeces, sourceofindeces):
        X = self.xdata[:, indeces]
        output = nnls(X, self.ydata)
        print('nnls output: ')
        print(output)
        print('')


    def fitpositiveLassoCVRFE(self):
        cv = TimeSeriesSplit(n_splits=3) #3 is the default
        lassomodel = LassoCV(n_alphas=2, alphas=np.linspace(0.01, 0.1, num=2), fit_intercept=False, precompute=True, \
                max_iter=2000, cv=cv,\
                positive=True, random_state=9999, selection='random')  #alpha=0.01 #alphas=np.linspace(0.01, 0.1, num=10), 
                                                                
        rfe = RFE(lassomodel, globe.SecuritiesPerBasket)
        fit = rfe.fit(self.xdata, self.ydata)
        print(fit)
    #print("Num Features: " + str(fit.n_features_))
    #print("Selected Features: " + str(fit.support_))
    #print("Feature Ranking: " + str(fit.ranking_))
        i = 0
        indices = list()
        for included in fit.support_:
            if included:
                indices.append(i)
            i += 1
        print('Indeces selected by RFE:')
        print(indices)
        lmodel = rfe.estimator_
        X = rfe.transform(self.xdata)
        port = subportfolio.subportfolio(lmodel, X, indices, self.ydata, 'positiveLassoCVRFE', 'RFE', self.plotmodelresults, \
            self.plt, self.catdata)
        port.evaluatemodelaccuracy(self.figurenr, self.listofsubportfolios)


    def fithedgeregressorRFE(self):
        indsorted = self.pearson_r_for_feature_selection(globe.NrFeaturesStartHedgeRegressorRFE)
        Xtrimmed = self.xdata[:, indsorted + self.principlefeatures]
        hedgereg = hedgeregressor.hedgeregressor()
        rfe = RFE(hedgereg, globe.SecuritiesPerBasket)
        fit = rfe.fit(Xtrimmed, self.ydata)
        #print(fit)
        #print("Num Features: " + str(fit.n_features_))
        #print("Selected Features: " + str(fit.support_))
        #print("Feature Ranking: " + str(fit.ranking_))
        i = 0
        indices = list()
        for included in fit.support_:
            if included:
                indices.append(i)
            i += 1
        print('Indeces selected by RFE:')
        print(indices)
        lmodel = rfe.estimator_
        X = rfe.transform(Xtrimmed)
        port = subportfolio.subportfolio(lmodel, X, indices, self.ydata, 'hedgeregressorRFE', 'RFE', \
            self.plotmodelresults, self.plt, self.catdata)
        port.evaluatemodelaccuracy(self.figurenr, self.listofsubportfolios)
    

    def bestportprint(self):
        print('Best model for ' + self.ydatalabel)
        self.bestport.keyportstats()
        self.bestport.analyseresovertime()
        print('Indices selected:')
        print(self.bestport.selectedindices)
        print('')


    def selectbestmodel(self):
        bestscore = 9999
        self.bestport = None
        for port in self.listofsubportfolios:
            if port.portscore < bestscore:
                self.bestport = port
                bestscore = port.portscore
        self.bestportprint()
        temp = self.bestport.plotmodelresults
        self.bestport.plotmodelresults = True
        self.bestport.plotportovertime(self.figurenr, self.ydatalabel)
        self.bestport.plotresovertime(self.figurenr, self.ydatalabel)
        self.bestport.plotmodelresults = temp


    def fitbestlinearmodel(self):
        global figurenr
        if self.plotmodelresults:
            fig = self.plt.figure(self.figurenr)
            self.figurenr += 1
            df = pd.DataFrame(self.ydata)
            df.columns = ['Reference ydata']
            df.plot()
            #fig.suptitle('Reference ydata', fontsize=20)

        indsortedftest = self.f_test_for_feature_selection()
        self.fitRidgeCV(indsortedftest, 'f_test feature selection')

        indsortedmi = self.mutual_info_for_feature_selection()
        self.fitRidgeCV(indsortedmi, 'mutual_info feature selection')

        indsortedpearson = self.pearson_r_for_feature_selection(globe.SecuritiesPerBasket)
        self.fitRidgeCV(indsortedpearson, 'pearson_r feature selection')

        self.fitpositiveLassoCV(indsortedpearson, 'pearson_r feature selection')

        self.fitnnls(indsortedpearson, 'pearson_r feature selection')

        indsortedrf = self.randomforrests_for_feature_selection()

        self.fitRidgeCV(indsortedrf, 'RandomForrest feature selection')

        self.fitpositiveLassoCV(indsortedrf, 'RandomForrest feature selection')

        self.fitpositiveLassoCVRFE()

        self.fithedgeregressorRFE()

        self.selectbestmodel()
        return self.bestport