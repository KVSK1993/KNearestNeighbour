# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:19:35 2017

@author: Karan Vijay Singh
"""
import timeit
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#starting time of program
startTime = timeit.default_timer()

#Reading of training data
training_data = pd.read_csv('MNIST_X_train.csv',header=None,nrows=60000).as_matrix()
training_result = pd.read_csv('MNIST_y_train.csv',header=None,nrows=60000).as_matrix()

#reading of test data
testing_data = pd.read_csv('MNIST_X_test.csv',header=None).as_matrix()
testing_result = pd.read_csv('MNIST_y_test.csv',header=None).as_matrix()

#normalising both test and training data
training_data = preprocessing.normalize(training_data)
testing_data = preprocessing.normalize(testing_data)

#defining number of K folds
No_of_folds = 10
CV = KFold(n_splits=No_of_folds)

#function to get majority vote for predicting
def get_majority_vote(neighbour_indexes, training_result):
    voteArray = np.zeros((10,))
    for i in neighbour_indexes:
         voteArray[training_result[i]]+=1
    return np.argmax(voteArray)

#precomputing dot product of each training point with itself to be used again and again
dotProductTrain = np.zeros(training_data.shape[0]).astype(float)
for i in range(training_data.shape[0]):
    dotProductTrain[i] = np.dot(training_data[i],training_data[i])

#precomputing dot product of each test point with itself to be used again and again
dotproductTest = np.zeros(testing_data.shape[0]).astype(float)
for i in range(testing_data.shape[0]):
    dotproductTest[i] = np.dot(testing_data[i],testing_data[i])


#function for implementing knn
def knn(k,training_data,training_result,testing_data,testing_result, dotProductTrain,dotproductTest):
    error = 0
    number_of_dataPts=testing_data.shape[0]
    for i in range(number_of_dataPts):
        testRow = testing_data[i] # taking each test row
        XY = np.dot(training_data,testRow)# taking dot product of test data with whole training data
        YY = np.full((training_data.shape[0],),dotproductTest[i]) #filling dot product of the ith test row equal to number to training data row
        distance =  dotProductTrain + YY - 2*XY # calaculating euclidean distance with formaul x.x +y.y - 2.x.y
        neighbour_indexes = np.argpartition(distance, k)[:k]
        FinalpredClass = get_majority_vote(neighbour_indexes,training_result)
        #print("Pred class for datapt = %d is %d and K= %d "% (i+1,FinalpredClass, k))
        if FinalpredClass != testing_result[i]:
            error+=1
    return (error*100)/testing_result.shape[0]

#values of k for validation
valuesOfK = [1,5,10,27,50]
error_For_Each_k = np.zeros((len(valuesOfK),))
index = 0

for k in (valuesOfK):
    SumofError = 0.0
    
    #applying Kfold
    for training_index, validation_index in CV.split(training_data):
        #partN+=1
        #putting training data and validation data for a partition
        traindata_array,validationIArray = training_data[training_index], training_data[validation_index]
        #putting training data result and validation data result for a partition
        trainresult_array,validationOArray =training_result[training_index], training_result[validation_index]
        #putting precalculated dot product of training data and validation data for a partition
        XX,YYY = dotProductTrain[training_index], dotProductTrain[validation_index]
        #calculating error value
        error_Value = knn(k,traindata_array, trainresult_array, validationIArray,validationOArray,XX,YYY)
        #adding error for each partition
        SumofError+=error_Value
    #calculating avg error for each k
    CV_Error_K = SumofError/No_of_folds
    error_For_Each_k[index] = CV_Error_K
    index+=1

#plotting of graph for CV error        
graph = plt.figure()

plotCV = graph.add_subplot(221)
plotCV.plot(valuesOfK, error_For_Each_k, 'r-')
plotCV.set_xlabel("Value of 'k'")
plotCV.set_ylabel("CV error rate")



#finding the best k
Bestk = valuesOfK[np.argmin(error_For_Each_k)]

#finding the test error
TestingError = knn(Bestk, training_data,training_result, testing_data, testing_result,dotProductTrain, dotproductTest)
print("Error Rate of Testing Data for best 'k' %d is %f" % (Bestk,TestingError)) 

print("Final error for each K",error_For_Each_k)

stopTime = timeit.default_timer()
print ("TotalTime to run the code ", stopTime - startTime)









    
    
