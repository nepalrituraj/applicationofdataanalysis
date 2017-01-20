import pandas as pd
import math
import operator
import random

from sklearn import neighbors

import sys
#Testing
from scipy.spatial.distance import euclidean as distance

trainingdata_path = "/pathtomydata"
data_properties =["Sepal Length","Sepal Width","Petal Length","Petal Width","Class"]


# 1 .Handle Data
def load_data():
    pd_data = pd.read_csv(trainingdata_path)
    pd_data.columns = data_properties
    return pd_data

#1.1. Handle real Value
# This function loads only four characters of the entire dataset
def iris_data(loaddata):
    return loaddata.ix[:,:4] # This loads only four character of the iris data

#2. Distance between two instances
def euclidean_distance(data1, data2):
    leng = len(data1)-1 # This avoids including the class of the dataset, i.e. only have four attributes
    euc = 0.0
    square_item = 0
    for i in range(leng):
        x = data1[data_properties[i]]
        y = data2[data_properties[i]]
        z = x - y
        square_item = math.pow(z,2)
        euc += square_item
    return math.sqrt(euc)

#. Get Neighbour
def getNeighbour(k,alldata,mydata_instance):
    distances = [] # An array object with key value pair
    leng = len(alldata)
    for x in range(leng):
        dist = euclidean_distance(mydata_instance,alldata.ix[x])
        distances.append((alldata.ix[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#. 4. Getting class label for our object
# This concept is predicting the class of specific object based on higher frequency of the neighbours class
# This class needs getNeighbour method to show nearest neighbours
def getclassprediction(k,data_instance,alldata):
    neighbours = getNeighbour(k,alldata,data_instance)
    irissetosa = 0
    irisversicolor = 0
    irisvirginica = 0
    for i in range(k):
        comparableinstance = neighbours[i]
        if (comparableinstance["Class"] == "Iris-setosa"):
            irissetosa += 1
        elif (comparableinstance["Class"] == "Iris Versicolour"):
            irisversicolor += 1
        elif (comparableinstance["Class"] == "Iris Virginica"):
            irisvirginica += 1
    x = max(irissetosa,irisversicolor,irisvirginica)
    if x == irisvirginica:
        return "Iris Virginica"
    if x == irisversicolor:
        return "Iris Versicolour"
    if x == irissetosa:
        return "Iris-setosa"

#5. Spliting into training and test data set
# Testing if our kNN Algorithm predicts correctly
def gettrain_test(alldata,n=109):
    df = alldata.sample(frac=1).reset_index(drop=True)
    return df.ix[:n,:], df.ix[n+1:,:]

#6. Testing our test data set with all data as traininig set and predict the class
#This defination needs gettrain_test(alldata,n=100)
def predict_testcases(alldata, n=109):
    leng = len(alldata)
    train,test = gettrain_test(alldata)
    predictions = []
    true_prediction = 0
    for i in range(len(test)):
        j = i + n + 1
        predict = getclassprediction(20,test.ix[j],train)
        if predict == test.ix[j]["Class"]:
            true_prediction += 1
    return true_prediction,len(test)




def main():
    print("------------------Assignment 1 --------------------")
    loaddata = load_data()
    data_load1 = loaddata.ix[0]
    data_load2 = loaddata.ix[1]

    dist = euclidean_distance(data_load1,data_load2)
    data1_neighbour = getNeighbour(3,loaddata,data_load1)
    #print(data1_neighbour) # These are 5 neighbours of our dataset which is data1
    dataload1_class = getclassprediction(5,data_load1,loaddata)
    print(dataload1_class)
    print("********************************************")
    predict_testcases(loaddata)
    true_prediction, totaltests = predict_testcases(loaddata)
    print("True Prediction : " ,true_prediction)
    print( "Total Tests : ", totaltests)
    print("Success Rate : ", true_prediction/totaltests *100)

    print("  Code from sklearn.Neighbours ")
    df = loaddata.sample(frac=1).reset_index(drop=True)
    train_= df.ix[:, :4]
    target_= df.ix[:,4:]
    print("Rituraj")
    nbrs = neighbors.KNeighborsClassifier(n_neighbors=8,algorithm='ball_tree').fit(train_,target_)



if __name__=="__main__":
    main()

