import pandas as pd
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pickle
import cv2
import os

# load mnist image data

def load_images_from_dir(dirname):
    
    objects = os.listdir(dirname)
    
    array = np.empty((len(objects),32,32,3),dtype = np.uint8)
    i= 0
    for objectname in objects:       
        im = cv2.imread(dirname+"/"+objectname)
        print("Image: ", i)
        plt.imshow(im)
        plt.show()

        array[i,:,:,:] = im
        i = i +1
        
        
    print( array.dtype)    
    data = {"X_test": array,"X_norm_test":gray_normalize(array),
               "y_test": np.array((13,25,1,12,11))}
    
    
    return data
    

def load_mnist():

    from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train           = shuffle(mnist.train.images, mnist.train.labels)
    X_validation, y_validation = shuffle(mnist.validation.images, mnist.validation.labels)
    X_test, y_test             = shuffle(mnist.test.images, mnist.test.labels)
    
    
    
    assert(len(X_train) == len(y_train))
    assert(len(X_validation) == len(y_validation))
    assert(len(X_test) == len(y_test))
    
    data = {"X_train" : X_train, 
            "y_train":y_train,
            "X_validation": X_validation,
            "y_validation":y_validation,
            "X_test":X_test,
            "y_test":y_test} 
    
    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))
    
    return data 

#load pickled image data (traffic signs)
def load_pickled_data():
    # Load pickled data


# TODO: Fill this in based on where you saved the training and testing data

    training_file = "C:/Users/Chris/SDC_projects/CarND-Traffic-Sign-Classifier-Project-P2/traffic-signs-data/train.p"
    validation_file="C:/Users/Chris/SDC_projects/CarND-Traffic-Sign-Classifier-Project-P2/traffic-signs-data/valid.p"
    testing_file = "C:/Users/Chris/SDC_projects/CarND-Traffic-Sign-Classifier-Project-P2/traffic-signs-data/test.p"
    
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
    X_train, y_train = shuffle(train['features'], train['labels'])
    X_validation, y_validation = shuffle(valid['features'], valid['labels'])
    X_test, y_test = shuffle(test['features'], test['labels'])
    
    assert(len(X_train) == len(y_train))
    assert(len(X_validation) == len(y_validation))
    assert(len(X_test) == len(y_test))
    
    data = {"X_train" : X_train, 
            "y_train":y_train,
            "X_validation": X_validation,
            "y_validation":y_validation,
            "X_test":X_test,
            "y_test":y_test} 
    
    
    # TODO: Number of training examples
    n_train = len(X_train)
    
    # TODO: Number of validation examples
    n_validation = len(X_validation)
    
    # TODO: Number of testing examples.
    n_test = len(X_test)
    
    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train[0].shape
    
    data_dtype = X_train.dtype
    
    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(range(min(y_train),max(y_train))) +1
    
    print("Number of training examples =", n_train)
    print("Shape of training examples =", X_train.shape)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print()
    print("Number of classes =", n_classes)
    print("Image data shape =", image_shape)
    print("Data type =", data_dtype)
    

    return data


#TODO  
def dict_to_pandas(dict_data, image_shape, data_length):
    
    
    col = range(image_shape[0])
    row = range(image_shape[1])
    channel = range(image_shape[2])
    data_length = range(data_length)
    index = pd.MultiIndex.from_product([col, channel, data_length],
                             names=['col', "channel", "frameno"])
                             
                             
    df = pd.DataFrame(dict_data, index=row, columns=index)                        
                             
                             
    return df
    
    
    
    

def viz_image_data(X_train, y_train):
    
    #%matplotlib inline
    
    
    
#    
#    index = random.randint(0, len(x_data))
#    image = x_data[index].squeeze()
#    
#    plt.figure(figsize=(1,1))
#    plt.imshow(image, cmap="gray")
#    print ("Index of label:", (y_data[index]))
#    
    n_classes =  max(y_train)-min(y_train)+1
    num_of_samples=[]
    plt.figure(figsize=(12, 16.5))
    for i in range(0, n_classes):
        plt.subplot(11, 4, i+1)
        x_selected = X_train[y_train == i]
        plt.imshow(x_selected[0, :, :, 0],cmap="gray") 
        plt.title(i)
        plt.axis('off')
        num_of_samples.append(len(x_selected))
    plt.show()
    
    #Plot number of images per class
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, n_classes), num_of_samples)
    plt.title("Class distribution of the loaded dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    
    print("Min number of images per class =", min(num_of_samples))
    print("Max number of images per class =", max(num_of_samples))
    
    
def eq_Hist(img):
    #Histogram Equalization
    img2=img.copy() 
    img2[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img2[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img2[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img2


def gray_normalize(data):
    data2 = data.copy()
    shape = data2.shape
    
    data3 = np.empty((shape[0],shape[1],shape[2],1),dtype=np.uint8)

    

    for i in range(len(data)):
        
        data3[i,:,:,0] = cv2.cvtColor(data2[i,:,:,:], cv2.COLOR_RGB2GRAY)
              
        data3[i,:,:,0] = cv2.equalizeHist(data3[i, :, :, 0])
        
    print  (data3.shape)
    return data3  

def normalize(data):
    data2 = data.copy()

    
    for i in range(len(data)):      
        data2[i,:,:,:] = eq_Hist(data[i,:,:,:])
        
 
    return data2   

if __name__ == "__main__":
#    data = load_pickled_data()
#    print (data["X_test"].shape, data["y_test"].shape)
#    data["X_train_norm"] = gray_normalize(data["X_train"])
#
#    viz_image_data(data["X_train_norm"], data["y_train"],max(data["y_train"])-min(data["y_train"])+1)
   
    #print (data["X_train_norm"])
#    
#    data = load_mnist()

#    viz_image_data(data["X_train"], data["y_train"])
#    
    data = load_images_from_dir("C:/Users/Chris/SDC_projects/CarND-Traffic-Sign-Classifier-Project-P2/traffic-signs-data/img")
    print (data["X_norm_test"].shape, data["y_test"].shape)
    viz_image_data(data["X_norm_test"], data["y_test"])
     
     
                          
     