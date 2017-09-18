# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:55:59 2017

@author: Chris
"""
import tensorflow as tf
import le_net
import le_net_loaddata
import matplotlib.pyplot as plt
import numpy as np


def load_and_test_model(X_test, y_test):

    
    
            x = tf.placeholder(tf.float32, (None, 32, 32, 1))
            y = tf.placeholder(tf.int32, (None))
            one_hot_y = tf.one_hot(y, 43)
            
            #learning rate 
            rate = 0.001
            BATCH_SIZE = 1
            ## training pipeline
            logits = le_net.LeNet(x)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
            loss_operation = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate = rate)
            training_operation = optimizer.minimize(loss_operation)
            
            # evaluation 
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
            accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
            def evaluate_all(X_data, y_data):
                num_examples = len(X_data)
                print(num_examples)
                total_accuracy = 0
                
                
                sess = tf.get_default_session()
                
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_test[offset:end], y_test[offset:end]
    
                    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                    print ("accuracy",accuracy)
                    total_accuracy += (accuracy * len(batch_x))
                    print ("total_accuracy",total_accuracy)
                    
 
                return total_accuracy / num_examples
            

            def top_prob(X_data, sess): 
                prob = sess.run(tf.nn.softmax(logits), feed_dict={x: X_data})    
                top_5 = tf.nn.top_k(prob, k=5)
                return sess.run(top_5) 
            
        
            saver = tf.train.Saver()
    
    
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess, \
            tf.device('/gpu:0'): 
                    saver.restore(sess, tf.train.latest_checkpoint('.'))
                
                    test_accuracy = evaluate_all(X_test, y_test)
                    print("Test Accuracy = {:.3f}".format(test_accuracy))
                    signs_top_5 = top_prob(X_test, sess)
                    print("Top five probabilities",signs_top_5)
                    
                    

                    
                    return test_accuracy, signs_top_5
                
                
                

                
                


if __name__ == "__main__":
    data = le_net_loaddata.load_images_from_dir("C:/Users/Chris/SDC_projects/CarND-Traffic-Sign-Classifier-Project-P2/traffic-signs-data/img")


    test_acc, signs_top_5 = load_and_test_model(data["X_norm_test"],data["y_test"])

   
    plt.figure(figsize=(16, 21))
    for i in range(5):
        plt.subplot(12, 2, 2*i+1)
        im = data["X_norm_test"][i,:,:,0]
        plt.imshow(im)
        plt.title(data["y_test"][i])
        plt.axis('off')
        plt.subplot(12, 2, 2*i+2)
        plt.barh(np.arange(1, 6, 1), signs_top_5.values[i, :])
        labs= signs_top_5.indices[i]
        plt.yticks(np.arange(1, 6, 1), labs)
    plt.show()


#    data["X_norm_test"] = le_net_loaddata.gray_normalize(data["X_test"]) 
#    print (data["X_norm_test"].shape, data["y_test"].shape)
        # load and test the model
#    load_and_test_model(data["X_train-norm"],data["y_train"])        
#    load_and_test_model(data["X_validation-norm"],data["y_validation"])
    
    
    