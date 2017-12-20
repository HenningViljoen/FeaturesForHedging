import pandas as pd
import numpy as np
import globe as globe
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


class subportfolio(object):
    def __init__(self, lmodel, X, indselected, ydata, modelname, sourceofindeces, plotmodelresults, plt, catdata=None):
            #resmodel: the residual model typically non-linear.  plt: matplotlib plot
        self.lmodel = lmodel
        self.resmodel = None
        self.catdata = catdata
        self.X = X
        self.selectedindices = indselected
        self.ydata = ydata
        self.modelname = modelname
        self.sourceofindeces = sourceofindeces
        self.plotmodelresults = plotmodelresults
        self.portfoliomse = 0.0 #mean square error
        self.n = len(ydata) #length of dataset
        self.portfoliovariance = 0.0
        self.covmatrix = list()
        self.nrfeatures = X.shape[1]
        self.portriskreturnbalance = 0.0
        self.weights = list()
        self.normalisedweights = list()
        self.cormatrix = list()
        self.portfoliocor = 0.0
        self.capital = globe.Capital
        self.capfractions = list() #the fraction of total capital to be invested in each security
        self.portproper = False #will be true if all fitted coeffs are positive.
        self.capitalallocation = list()
        self.plt = plt
        self.r2 = 0.0 #R^2
        self.resr2 = 0.0 #R^2 of residual categorial model
        self.totalr2 = 0.0 #R^2 of the linear model plus the residual model
        self.multiple = 1.0 #scale of base fit, to the capital investment.
        self.portscore = 0.0


    def keyportstats(self):
        s1 = self.modelname + ', ' + self.sourceofindeces + ', R^2: ' + str(round(self.r2,globe.NrD)) + \
                ', pcor: ' + str(round(self.portfoliocor, globe.NrD)) + ' pscore: ' + str(round(self.portscore, globe.NrD))
        print(s1)
        self.allocatefunds()


    def evaluatemodelaccuracy(self, figurenr, listofsubportfolios):
        #global figurenr
        self.fitresmodel()
        print(self.sourceofindeces + ', ' + self.modelname + ' model score (R^2):')
        self.calcr2s()
        print(self.r2)
        self.weights = list(self.lmodel.coef_)
        #print(self.weights)
        #print(self.lmodel.coef_)
        self.testproper()
        self.calccovmatrix()
        print(self.sourceofindeces + ', ' + self.modelname + ' model covariance:')
        print(self.portfoliovariance)
        self.calccormatrix()
        self.calcportscore()
        print(self.sourceofindeces + ', ' + self.modelname + ' model correlation:')
        print(self.portfoliocor)
        print(self.sourceofindeces + ', ' + self.modelname + ' overall score:')
        print(self.portscore)
        print(self.sourceofindeces + ', ' + self.modelname + ' model coeff:')
        print(self.lmodel.coef_)
        print(self.sourceofindeces + ', ' + self.modelname + ' model normalised coeff:')
        print(self.normalisedweights)
        self.allocatefunds()
        print('')
        #if self.plotmodelresults:
        #    fig = self.plt.figure(figurenr)
        #    figurenr += 1
        #    df = pd.DataFrame(self.lmodel.predict(self.X))
        #    df.columns = [self.modelname + ', ' + self.sourceofindeces + ', R^2: ' + str(self.r2)]
        #    df.plot()
        self.plotportovertime(figurenr)
        listofsubportfolios.append(self)


    def fitresmodel(self):
        if self.catdata != None:
            rf = RandomForestRegressor()
            cv = TimeSeriesSplit()
            self.resmodel = GridSearchCV(rf, {'min_samples_leaf': range(1,10)}, cv=cv)
            res = self.ydata - self.lmodel.predict(self.X)
            self.resmodel.fit(self.catdata,res)
            self.resr2 = self.resmodel.score(self.catdata,res)
            #print('resr2: ' + str(resr2))

    def calcr2s(self):
        self.r2 = self.lmodel.score(self.X, self.ydata)
        if self.resmodel != None:
            self.totalr2 = r2_score(self.ydata, (self.lmodel.predict(self.X) + self.resmodel.predict(self.catdata)))


    def calcportscore(self):
        self.portscore = self.portfoliocor - self.r2


    def testproper(self):
        self.portproper = True
        for i in range(self.nrfeatures):
            if self.weights[i] < 0:
                self.portproper = False
                return 0

    def calccovmatrix(self):
        self.calcnormalisedweights()
        self.portfoliovariance = 0.0
        self.covmatrix = list()
        for i in range(self.nrfeatures):
            matrixrow = list()
            for j in range(self.nrfeatures):
                coeff = np.cov(self.X[i], self.X[j])[0][1]
                matrixrow.append(coeff)
                self.portfoliovariance += \
                    self.normalisedweights[i]*self.normalisedweights[j]*coeff
            self.covmatrix.append(matrixrow)
        #print('self.portfoliovariance : ',self.portfoliovariance)          
        #self.portfoliostd = math.sqrt(self.portfoliovariance)
        #print('self.portfoliostd : ',self.portfoliostd)


    def calcnormalisedweights(self):
        total = 0.0
        self.normalisedweights = []
        for i in range(self.nrfeatures):
            total += abs(self.weights[i])
        for i in range(self.nrfeatures):
            self.normalisedweights.append(abs(self.weights[i])/total)

    
    def calccormatrix(self):
        self.calcnormalisedweights()
        self.portfoliocor = 0.0
        self.cormatrix = list()
        for i in range(self.nrfeatures):
            matrixrow = list()
            for j in range(self.nrfeatures):
                r = pearsonr(self.X[i], self.X[j])
                coeff = r[0]
                matrixrow.append(coeff)
                self.portfoliocor += \
                    self.normalisedweights[i]*self.normalisedweights[j]*coeff
            self.cormatrix.append(matrixrow)


    def allocatefunds(self):
        if self.portproper:
            self.multiple = self.capital/self.ydata[0]
            den = sum([self.weights[i]*self.X[0,i] for i in range(self.nrfeatures)])
            self.capfractions = [self.weights[i]*self.X[0,i]/den for i in range(self.nrfeatures)]
            self.capitalallocation = [self.capfractions[i]*self.capital/globe.Mil for i in range(self.nrfeatures)]
            for i in range(self.nrfeatures):
                print('Capital for security ' + str(i) + ': ' + str(self.capitalallocation[i]))
            totalcapitalallocation = sum(self.capitalallocation)
            print('Total capital :' + str(totalcapitalallocation))

    
    def plotportovertime(self, figurenr, ydatalabel=''):
        if self.plotmodelresults and self.portproper:
            fig = self.plt.figure(figurenr)
            figurenr += 1
            df = pd.DataFrame(self.lmodel.predict(self.X)*self.multiple)
            df.columns = [self.modelname + ', ' + self.sourceofindeces + ', R^2: ' + str(round(self.r2,globe.NrD)) + \
                ', pcor: ' + str(round(self.portfoliocor, globe.NrD)) + ' pscore: ' + str(round(self.portscore, globe.NrD))]
            df['Reference ' + ydatalabel] = pd.Series(self.ydata*self.multiple)
            if self.resmodel != None:
                df['Linear model + non-linear residual model, R^2: ' + str(round(self.totalr2,globe.NrD))] = \
                    pd.Series((self.lmodel.predict(self.X) + self.resmodel.predict(self.catdata))*self.multiple)
            df.plot()


    def plotresovertime(self, figurenr, ydatalabel=''):
        if self.plotmodelresults and self.portproper:
            fig = self.plt.figure(figurenr)
            figurenr += 1
            res = (self.ydata - self.lmodel.predict(self.X))*self.multiple
            df = pd.DataFrame(res)
            df.columns = [ydatalabel + ' residuals']
            df.plot()

    def analyseresovertime(self):
        res = (self.ydata - self.lmodel.predict(self.X))*self.multiple
        maxres = res.max()
        minres = res.min()
        std = res.std()
        mean = res.mean()
        print('Residual max: ' + str(maxres))
        print('Residual min: ' + str(minres))
        print('Residual std: ' + str(std))
        print('Residual mean: ' + str(mean))

    


