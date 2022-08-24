import cv2
from cv2 import *
import keras
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
ef knn(x_train_kn,y_train_kn,x_test_kn,y_test_kn):
    n = KNeighborsClassifier(n_neighbors=9)
    n.fit(x_train_kn, y_train_kn)
    y_predict_kn = n.predict(x_test_kn)
    accuracy = accuracy_score(y_test_kn, y_predict_kn)
    print("Accuracy: ",accuracy*100)
    pre=precision_score(y_test_kn,y_predict_kn,average="micro")
    rec = recall_score(y_test_kn, y_predict_kn, average='micro')
    measure = 2 * (pre * rec) / (pre + rec)
    print("recall: ", rec*100)
    print("precision: ", pre*100)
    print("Measure: ", measure)

def Tree(x_train_ds,y_train_ds,x_test_ds, y_test_ds):
    decisionTree = tree.DecisionTreeClassifier()
    # build and train the decisionTree model
    decisionTree = decisionTree.fit(x_train_ds, y_train_ds)
    # testing the model with test set
    y_predict_ds = decisionTree.predict(x_test_ds)
    #print(y_predict_ds)
    print("Accuracy: ",accuracy_score(y_test_ds, y_predict_ds) * 100)
    pre = precision_score(y_test_ds, y_predict_ds, average="micro")
    rec = recall_score(y_test_ds, y_predict_ds, average='micro')
    measure = 2 * (pre * rec) / (pre + rec)
    print("recall: ", rec * 100)
    print("precision: ", pre * 100)
    print("Measure: ", measure)
def logistic (X_train,y_train, X_test, y_test ):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    pre = precision_score(y_test,y_predict,average="micro")
    rec = recall_score(y_test, y_predict,average="micro")
    print("Accuracy: ",accuracy_score(y_test, y_predict)*100)
    measure = 2 * (pre * rec) / (pre + rec)
    print("recall: ", rec * 100)
    print("precision: ", pre * 100)
    print("Measure: ", measure)
    #print("result",y_predict)

path="D:\\Bioinformatics\\Fourth-Year\\Machine_Learning_And_Bioinformatics\\assignmt_2\\ASL_Alphabet_Dataset\\asl_alphabet_train"
files_train=os.listdir(path)
size=65
path2="D:\\Bioinformatics\\Fourth-Year\\Machine_Learning_And_Bioinformatics\\assignmt_2\ASL_Alphabet_Dataset\\asl_alphabet_test"
file_test=os.listdir(path2)
def read_image_test(file,path):
    x_test_list=[]
    y_test_list=[]
    for j in file:
        image = (cv2.imread(path + '\\' + j))
        image = cv2.resize(image, (size, size))
        y_test_list.append(j)
        x_test_list.append(image)
    for i in range(len(y_test_list)):
        y_test_list[i] = y_test_list[i].split("_")[0]
    print("reading test train finished ")
    return x_test_list,y_test_list
def read_image_train(file,path):
    x_train_list=[]
    y_train_list=[]
    for i in file:
        counter=0
        file_img = os.listdir(path+'\\'+i)
        temp_path=path+'\\'+i
        for j in range(0,len(file_img),3):
              img=(cv2.imread(temp_path+'\\'+file_img[j]))
              img = cv2.resize(img, (size, size))
              #img=img.flatten()
              x_train_list.append(img)
              y_train_list.append(i)
             # counter+=1
              #if(counter==10):
               #  break
    print("reading train finished ")
    return x_train_list , y_train_list
def image_proccesing(images):
    canny_list = []
    for i in range(len(images)):
        img=images[i]
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(image=img_blur, threshold1=4, threshold2=115)
        canny_list.append(edges.flatten())
    canny= np.array(canny_list)
    return canny


def main():

    flatten_train=[]# to store RGB train_images after flatten
    gray_images = [] #to store copy of images as gray
    binary_images = [] #to store copy of images as binary
    flatten_test=[] # to store the test images after flatten it and convert it to gray to be one channel
    x_train, y_train = read_image_train(files_train, path)
    x_test, y_test = read_image_test(file_test, path2)


    #converting copy of images into gray
    # for i in range(len(x_train)):
    #     img=(cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY))
    #     img=img.flatten()
    #     gray_images.append(img)

    #converting copy of images into binary based on the gray copy
    # for i in range(len(gray_images)):
    #     r, threshold = cv2.threshold(gray_images[i], 100, 255, cv2.THRESH_BINARY)
    #     binary_images.append(threshold)
    # cv2.imshow("kjfkjad",binary_images[1])
    # cv2.imshow("kj",gray_images[1])

    # flatten images
    for i in range(len(x_train)):
        flatten_train.append(x_train[i].flatten())

    #call image proccesing on the train images
    #cv2.imshow("addjvkj", binary_images[1])
    #canny_flatten_list=image_proccesing(binary_images)

    #converting the test images into gray and flatten it
    # for i in x_test:
    #     img=(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
    #     flatten_test.append(img.flatten())


    # flatten the test images to run with RGB
    for i in x_test:
        flatten_test.append(i.flatten())


    #calling different classifier

    #knn(flatten_train,y_train,flatten_test,y_test)
    #Tree(flatten_train,y_train,flatten_test,y_test)
    logistic(flatten_train,y_train,flatten_test,y_test)


main()


cv2.waitKey(0)
cv2.destroyAllWindows()




#####################
#continue problem 3


path='asl_alphabet_train//asl_alphabet_train'

files_train=os.listdir(path)
size=64

path2='asl_alphabet_test//asl_alphabet_test'

file_test=os.listdir(path2)

classes = {'A':0 ,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25,"del":26,"nothing":27,"space":28}

def getcode(n) : 
    for x , y in classes.items() : 
        if n == y : 
            return x

def read_image_test(file,path):
    x_test_list=[]
    y_test_list=[]
    for j in file:
        image = (cv2.imread(path + '//' + j))
        image = cv2.resize(image, (size, size))
        y_test_list.append(j)
        # image=image.flatten()
        x_test_list.append(image)
    for i in range(len(y_test_list)):
        y_test_list[i] = classes[y_test_list[i].split("_")[0]]
    return x_test_list,y_test_list
x_test,y_test=read_image_test(file_test,path2)

print(len(x_test))
print(y_test)
#print(sorted(y_test))

    #cv2.imshow("binary",r)

def read_image_train(file,path):
    x_train_list=[]
    y_train_list=[]
    for i in file:
        temp_path=''
        counter=0
        file_img=[]
        file_img = os.listdir(path+'//'+i)
        temp_path=path+'//'+i
        for j in range(0,len(file_img),3):
              img=(cv2.imread(temp_path+'//'+file_img[j]))
              img = cv2.resize(img, (size, size))
              x_train_list.append(img)
              y_train_list.append(classes[i])
              counter+=1
              if(counter==100):
                  break
    return x_train_list , y_train_list
x_train,y_train=read_image_train(files_train,path)
print(len(x_train))
print(len(y_train))
#print(sorted(y_train))

x_train_np = np.array(x_train)
x_test_np = np.array(x_test)
#X_pred_array = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f'X_train shape  is {x_train_np.shape}')
print(f'X_test shape  is {x_test_np.shape}')
#print(f'X_pred shape  is {X_pred_array.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')

KerasModel3 = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(size,size,3)),
        keras.layers.MaxPool2D(4,4),
        
        keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        
        keras.layers.Flatten() ,    
          
        keras.layers.Dense(29,activation='softmax') ,    
        ])

KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(size,size,3)),
        keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(100,kernel_size=(3,3),activation='relu'),    
        # keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'),    
        # keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten() ,    
        # keras.layers.Dense(120,activation='relu') ,    
        # keras.layers.Dense(100,activation='relu') ,    
        #keras.layers.Dense(50,activation='relu') ,        
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(29,activation='softmax') ,    
        ])

KerasModel.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

KerasModel2.compile(optimizer ='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

epochs = 30
ThisModel = KerasModel.fit(x_train_np, y_train, epochs=epochs,batch_size=64,verbose=1)

epochs = 40
ThisModel3 = KerasModel3.fit(x_train_np, y_train, epochs=epochs,batch_size=64,verbose=1)

ModelLoss2, ModelAccuracy2 = KerasModel.evaluate(x_test_np, y_test)

print('Test Loss is {}'.format(ModelLoss2))
print('Test Accuracy is {}'.format(ModelAccuracy2 ))

ModelLoss, ModelAccuracy = KerasModel3.evaluate(x_test_np, y_test)
# accuracy 0.9286 model 2  

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))

y_pred = KerasModel3.predict(x_test_np)

print('Prediction Shape is {}'.format(y_pred.shape))

y_pred = KerasModel.predict(x_test_np)

print('Prediction Shape is {}'.format(y_pred.shape))



##############################################################################
###Bonus


import cv2
from cv2 import *
import keras
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from sklearn import  tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import os
import tensorflow as tf
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from google.colab.patches import cv2_imshow
from keras.preprocessing import image
from keras.initializers import glorot_uniform

path='asl_alphabet_train//asl_alphabet_train'
files_train=os.listdir(path)
size=64
path2='asl_alphabet_test//asl_alphabet_test'
file_test=os.listdir(path2)
classes = {'A':0 ,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25,"del":26,"nothing":27,"space":28}

def getcode(n) :
    for x , y in classes.items() :
        if n == y :
            return x
def read_image_test(file,path):
    x_test_list=[]
    y_test_list=[]
    x_test1=[]
    x_test2=[]
    for j in file:
        image = (cv2.imread(path + '//' + j))
        image = cv2.resize(image, (size, size))
        y_test_list.append(j)
        # image=image.flatten()
        x_test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize images
        x_test_gray=(x_test_gray-np.min(x_test_gray))/(np.max(x_test_gray)-np.min(x_test_gray))
        #augmentation
        #x_test_R=cv2.rotate(x_train_gray,rotateCode=ROTATE_90_CLOCKWISE)
        # x_test1.append(x_test_gray)
        # x_test2.append(x_test_R)
        # x_test_list=x_test1+x_test2
        x_test_list.append(x_test_gray)
    for i in range(len(y_test_list)):
        y_test_list[i] = classes[y_test_list[i].split("_")[0]]
    return x_test_list,y_test_list
x_test,y_test=read_image_test(file_test,path2)

print(len(x_test))
print(y_test)
#print(sorted(y_test))
#cv2.imshow("binary",r)

def read_image_train(file,path):
    x_train_list=[]
    y_train_list=[]
    x_t=[]
    x_tr=[]
    for i in file:
        temp_path=''
        counter=0
        file_img=[]
        file_img = os.listdir(path+'//'+i)
        temp_path=path+'//'+i
        for j in range(0,len(file_img)):
              img=(cv2.imread(temp_path+'//'+file_img[j]))
              img = cv2.resize(img, (size, size))
              x_train_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              # Normalize images
              x_train_gray=(x_train_gray-np.min(x_train_gray))/(np.max(x_train_gray)-np.min(x_train_gray))
              # Augmentation
              #x_train_R=cv2.rotate(x_train_gray,rotateCode=ROTATE_90_CLOCKWISE)
              # x_t.append(x_train_gray)
              # x_tr.append(x_train_R)
              # x_train_list=x_t+x_tr
              x_train_list.append(x_train_gray)
              y_train_list.append(classes[i])
              counter+=1
              if(counter==200):
                  break
    return x_train_list , y_train_list
x_train,y_train=read_image_train(files_train,path)
print(len(x_train))
print(len(y_train))
#print(sorted(y_train))
x_train_np = np.array(x_train)
x_train_np=np.expand_dims(x_train_np,axis=3)
x_test_np = np.array(x_test)
x_test_np=np.expand_dims(x_test_np,axis=3)
#X_pred_array = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(f'x_train_np shape  is {x_train_np.shape}')
print(f'X_test shape  is {x_test_np.shape}')
#print(f'X_pred shape  is {X_pred_array.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')
##################################################################
#model2 for problem 3 Bonus
KerasModel = keras.models.Sequential([
    keras.layers.Conv2D(200, kernel_size=(3, 3), activation='relu', input_shape=(size, size, 1)),
    keras.layers.MaxPool2D(4, 4),

    keras.layers.Conv2D(120, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(4, 4),

    keras.layers.Flatten(),

    keras.layers.Dense(29, activation='softmax'),
])
KerasModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
epochs = 10
ThisModel = KerasModel.fit(x_train_np, y_train, epochs=epochs,batch_size=64,verbose=1)
ModelLoss, ModelAccuracy = KerasModel.evaluate(x_test_np, y_test)
print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))
###################################################################
##Model VGG16
KerasModel2 = keras.models.Sequential([
        keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(size,size,1),padding="same",),
        keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu',padding="same"),
        #keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(512,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.Conv2D(512,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.Conv2D(512,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(512,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.Conv2D(512,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.Conv2D(512,kernel_size=(3,3),activation='relu',padding="same"),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten() ,
        keras.layers.Dense(4096,activation='relu') ,
        keras.layers.Dense(4096,activation='relu') ,
        #keras.layers.Dense(4096,activation='relu') ,
        #keras.layers.Dropout(rate=0.5) ,
        keras.layers.Dense(29,activation='softmax') ,
        ])
KerasModel2.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
epochs = 10
ThisModel2 = KerasModel2.fit(x_train_np, y_train, epochs=epochs,batch_size=64,verbose=1)
ModelLoss2, ModelAccuracy2 = KerasModel2.evaluate(x_test_np, y_test)
print('Test Loss is {}'.format(ModelLoss2))
print('Test Accuracy is {}'.format(ModelAccuracy2))
