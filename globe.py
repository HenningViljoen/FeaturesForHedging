#Global variables and global constants for the whole program.

RiskTollerance = 1.0
Epsilon = 0.00001
Capital = 10000000.0 #USD 10m
Mil = 1000000.0
NrD = 3
NrFeaturesStartHedgeRegressorRFE = 10 #20 #10
NrHedgeRegressorCV = 3 #The number of of time series splits to be used for testing and validation for the regressor
NrPCAComponents = 5 #5 #10 #The number of components to divide the data into
NrFeaturesPerComponent = 1 #The number of features to include in the hyperspace of fitting per principle component
SecuritiesPerBasket = 5