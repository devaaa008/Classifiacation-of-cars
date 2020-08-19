import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import math
import joblib


class RADIAL_BASIS():

    #initializing variables
    def __init__(self):
        self.DIR_TRAIN=r"train200"
        self.DIR_TEST=r"test50"
        self.CATEGORIES=["audi_train200","benz_train200","gran_train200"]
        self.data_train=[]
        self.data_test=[]
        self.training=[]
        self.testing=[]
        self.centroids=[]
        self.w=[]
        self.k=6
        self.std=[]
        self.group=[]
        self.LR=0.01
        self.b=[]
        self.trainX=[]
        self.testX=[]
        self.trainY=[]
        self.testY=[]
        self.epochs=15

    """#data for training set
    def DRAWOUT_TRAIN(self):
        for category in self.CATEGORIES:
            c=0
            path=os.path.join(self.DIR_TRAIN,category)
            class_num=self.CATEGORIES.index(category)
            for img in os.listdir(path):
                if c<150:
                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    img_array=cv2.resize(img_array,(100,100))
                    img_array=cv2.GaussianBlur(img_array,(7,7),0)
                    img_array1=img_array[30:65,40:85]
                    #plt.imshow(img_array1,cmap="gray")
                    
                    self.data_train.append([img_array1,class_num])
                    #plt.imshow(img_array1,cmap="gray")
                    #plt.show()
                    c=c+1
        joblib.dump(self.data_train,"data_train.txt")
                    
    #extracting data for testing set
    def DRAWOUT_TEST(self):
        for category in self.CATEGORIES:
            c=0
            path=os.path.join(self.DIR_TEST,category)
            class_num=self.CATEGORIES.index(category)
            for img in os.listdir(path):
                if c<60:
                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    img_array=cv2.resize(img_array,(100,100))
                    img_array=cv2.GaussianBlur(img_array,(7,7),0)
                    img_array1=img_array[30:65,40:85]
                    self.data_test.append([img_array1,class_num])
                    #plt.imshow(img_array1,cmap="gray")
                    #plt.show()
                    c=c+1
        joblib.dump(self.data_test,"data_test.txt")
                
    """#training 
    def train(self):
        #random.shuffle(self.data)
        self.data_train=joblib.load("data_train.txt")
        #print(self.data_train)
        self.training=self.data_train[0:]
        for i in self.training:
            self.trainX.append(np.array(i[0],dtype='int32').flatten())
            #print(self.trainX)
            self.trainY.append(i[1])
        
    #onehot encoding of y
    def one_hot_encode(self,y, num_of_classes):
        arr = np.zeros((len(y), num_of_classes))
        for i in range(len(y)):
            c = int(y[i])
            arr[i][c] = 1
        return arr
        
    #calculate distance between data points and centroids
    def calculate_dist(self,c,x):
        sum = 0
        for i in range(len(c)):
            sum += (c[i] - x[i]) ** 2
        return np.sqrt(sum)
        
    #applying Kmeans
    def FIND_CENTERS(self):
        self.trainX=np.array(self.trainX)
        #print(self.trainX)
        np.random.seed(len(self.trainX))
        self.centroids = self.trainX[np.random.choice(range(len(self.trainX)), self.k, replace=False)]
        #print(self.centroids)
        #joblib.dump(self.centroids,"centroids.txt")
        found = False
        it = 0
        while (not found) and (it < 100):
            self.group= [[] for i in range(len(self.centroids))]
            for x in self.trainX:  
                dist= []
                for c in self.centroids:
                    dist.append(self.calculate_dist(c, x))
                self.group[int(np.argmin(dist))].append(x)
            self.group = list((filter(None, self.group)))
            centroids1 = self.centroids.copy()
            self.centroids = []
            for j in range(len(self.group)):
                self.centroids.append(np.mean(self.group[j], axis=0))
            pattern = np.abs(np.sum(centroids1) - np.sum(self.centroids))
            print('MEAN_VALUE: ', int(pattern))
            found = (pattern == 0)
            it += 1
        
        max=0
        for i in range(0,len(self.centroids)):
            for j in range(0,len(self.centroids)):
                d=self.calculate_dist(self.centroids[i],self.centroids[j])
            if(d>max):
                 max=d
        d=max
        self.std= [d/math.sqrt(2*self.k) for x in range(len(self.centroids))]

    #calculating gaussian function
    def gauss(self,x, c, s):
        d=(float)(self.calculate_dist(c,x))
        return np.exp(-(float)(d**2) / (2 * s**2))

    #training phase
    def train_rbf(self):
        self.trainY=self.one_hot_encode(self.trainY, 3)
        #b is 1*3 matrix and w is len(centroids)*3 matrix
        self.b=[np.random.randn(3)]
        self.w=np.array([np.random.rand(3) for j in range(len(self.centroids))])
        
        for epoch in range(self.epochs):
           
            for i in range(len(self.trainX)):
                x = np.array([self.gauss(self.trainX[i], c, s) for c, s, in zip(self.centroids, self.std)])
                y = x.dot(self.w) + self.b
                
                error = np.array((self.trainY[i] - y))
                x.resize(1,len(self.centroids))
                error.resize(1,3)
                self.w = self.w + self.LR *( x.T ).dot(error)
                self.b = self.b + self.LR * error
            print('Weight and bias after ',epoch+1,' epoch')
            print(self.w)
            print(self.b)
            
        
        
    #prediction/testing
    def predict(self):
        self.data_test=joblib.load("data_test.txt")
        
        self.testing=self.data_test[0:]
        c=0
        for i in self.testing:
            self.testX.append(np.array(i[0]).flatten())
            self.testY.append(i[1])
        y_pred = list()
        for i in range(len(self.testX)):
                x = np.array([self.gauss(self.testX[i], c, s) for c, s, in zip(self.centroids, self.std)])
                y = x.dot(self.w) + self.b
                y_pred.append(y.flatten())
        y_pred= np.array([np.argmax(j) for j in y_pred])
        print('prediction:',y_pred)
        diff = y_pred - self.testY
        self.testY=np.array(self.testY)
        for i in range(len(diff)):
            if diff[i]==0:
                c=c+1
        print('Accuracy: ', c / len(diff))
        

obj=RADIAL_BASIS()
#obj.DRAWOUT_TRAIN()
obj.train()
obj.FIND_CENTERS()
obj.train_rbf()
#obj.DRAWOUT_TEST()
obj.predict()

 