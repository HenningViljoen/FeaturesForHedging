from sklearn import base
import subportfolio as subportfolio
from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LinearRegression
import globe as globe
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit

class hedgeregressor(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        #self.linreg = LinearRegression(fit_intercept=False)
        self.weights = list()
        self.coef_ = None
        self.ydata = None
        self.X = None
        self.n = 0
        self.nfeatures = 0
        self.portfoliorisktollerance = globe.RiskTollerance
        self.portriskreturnbalance = 0.0
        self.portfoliomse = 0.0
        self.dbaldw = list()    #the derivative of the risk return balance wrt weights
        self.normalisedweights = list()
        self.cormatrix = list()
        self.portfoliocor = 0.0


    def calcportfoliomse(self):
        self.portfoliomse = 0.0
        ypredict = self.predict(self.X) #local var
        for i in range(self.n):
            self.portfoliomse += (ypredict[i] - self.ydata[i])**2
        self.portfoliomse /= self.n
    

    def calccovmatrix(self):
        self.portfoliovariance = 0.0
        self.covmatrix = list()
        for i in range(self.nfeatures):
            matrixrow = list()
            for j in range(self.nfeatures):
                coeff = np.cov(self.X[:,i], self.X[:,j])[0][1]
                matrixrow.append(coeff)
                self.portfoliovariance += \
                    self.weights[i]*self.weights[j]*coeff
            self.covmatrix.append(matrixrow)

    def calccormatrix(self):
        self.calcnormalisedweights()
        self.portfoliocor = 0.0
        self.cormatrix = list()
        for i in range(self.nfeatures):
            matrixrow = list()
            for j in range(self.nfeatures):
                r = pearsonr(self.X[i], self.X[j])
                coeff = r[0]
                matrixrow.append(coeff)
                self.portfoliocor += \
                    self.normalisedweights[i]*self.normalisedweights[j]*coeff
            self.cormatrix.append(matrixrow)


    def calcnormalisedweights(self):
        total = 0.0
        self.normalisedweights = []
        for i in range(self.nfeatures):
            total += abs(self.weights[i])
        for i in range(self.nfeatures):
            self.normalisedweights.append(abs(self.weights[i])/(total + globe.Epsilon))


    def calcportriskreturnbalance(self):
        #self.calcportfoliomse()
        self.calccovmatrix()
        r2 = self.score(self.X,self.ydata)
        #self.portriskreturnbalance = self.portfoliostd - PortfolioRiskTollerance*self.portfolioreturn
        self.portriskreturnbalance = self.portfoliovariance - self.portfoliorisktollerance*r2
        #print('self.portriskreturnbalance : ',self.portriskreturnbalance)


    def funcriskreturn(self, x, sign=1.0):
        self.weights = x
        self.calcportriskreturnbalance()
        return self.portriskreturnbalance


    def funcriskreturnderiv(self, x, sign=1.0):
        self.weights = x
        self.calcportriskreturnbalance()
        J0 = self.portriskreturnbalance
        for i in range(self.nfeatures):
            oldweight = self.weights[i]
            self.weights[i] += globe.Epsilon
            self.calcportriskreturnbalance()
            J1 = self.portriskreturnbalance
            self.dbaldw[i] = (J1 - J0)/globe.Epsilon
            self.weights[i] = oldweight
        return np.array(self.dbaldw)



    def fit(self, X, y=None):
        self.X = X
        self.ydata = y
        self.n = X.shape[0]
        self.nfeatures = X.shape[1]
        self.weights = [0.0]*self.nfeatures
        self.dbaldw = [1.0]*self.nfeatures
        self.coef_ = np.zeros(self.nfeatures)
        #cons = ({'type': 'eq', \
        #    'fun' : lambda x: np.array(sum(x) - 1), \
        #    'jac' : lambda x: np.ones(len(self.assets))})
        bnds = [(0, None)] #was [(0, None)]
        for i in range(1, self.nfeatures):
            bnds.append((0, None))
        #print(bnds)
        #print(self.weights)
        cv = TimeSeriesSplit(n_splits=globe.NrHedgeRegressorCV)
        allweights = [[0.0]*self.nfeatures for i in range(self.nfeatures)] #average weights from all CV cases to be combined
        run = 0
        besttestscore = -1000
        bestrun = -1
        for train_index, test_index in cv.split(X):
            self.X = X[train_index, :]
            self.ydata = y[train_index]
            res = minimize(self.funcriskreturn, self.weights, args=(), \
                       jac=self.funcriskreturnderiv, \
                       bounds=bnds, method='SLSQP', options={'disp': False})
            for i in range(self.nfeatures):
                self.weights[i] = res.x[i]
                allweights[run][i] += self.weights[i]
                self.coef_[i] = self.weights[i]
            trainscore = self.score(self.X,self.ydata)
            testscore = self.score(X[test_index, :], y[test_index])
            print('trainscore: ' + str(trainscore) + ';  testscore: ' + str(testscore))
            if testscore > besttestscore:
                besttestscore = testscore
                bestrun = run
            run += 1
        #aveweights = [aveweights[i]/globe.NrHedgeRegressorCV for i in range(self.nfeatures)]
        self.weights = [allweights[bestrun][i] for i in range(self.nfeatures)]
        for i in range(self.nfeatures): self.coef_[i] = self.weights[i]
        print('final testscore: ' + str(self.score(X,y)))
        print('minimize iterated... self.nfeatures: ' + str(self.nfeatures))
        return self
    

    def transform(self, X):
        return X


    def score(self, X, ydata):
        ypredict = self.predict(X)
        return r2_score(ydata, ypredict)
    

    def predict(self, X):
        n = X.shape[0]
        y = np.zeros(n)
        for i in range(n):
            for j in range(self.nfeatures):
                y[i] += self.weights[j]*X[i,j]
        return y