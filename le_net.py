import tensorflow as tf
from sklearn.utils import shuffle
import le_net_loaddata

from tensorflow.contrib.layers import flatten



                        

def LeNet_training(X_train,y_train, X_validation, y_validation, X_test, y_test):
    


        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, 43)
        
    
        
        #learning rate 
        rate = 0.001
        ## training pipeline
        logits = LeNet(x)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)
        
        # evaluation 
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        
        def evaluate(X_data, y_data):
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, BATCH_SIZE):
                batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
                accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples
        
    
        saver = tf.train.Saver()
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess, \
        tf.device('/gpu:0'):
            
            print("#training settings#")
                  
            EPOCHS = 2
            BATCH_SIZE = 125
            
            
            print ("Training Epochs", EPOCHS)
            print ("Training Batch Size", BATCH_SIZE)
            
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)
            
            print("Training...")
            print()
            for i in range(EPOCHS):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
                training_accuracy = evaluate(X_train, y_train)
                validation_accuracy = evaluate(X_validation, y_validation)
                print("EPOCH {} ...".format(i+1))
                print("Training Accuracy = {:.3f}".format(training_accuracy))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))

                print( 'LOSS = ' + str(cross_entropy))
                print()
                
            saver.save(sess, './lenet_test_e20_b500')
            print("Model saved")
            
            
            
            with tf.Session() as sess:
                saver.restore(sess, tf.train.latest_checkpoint('.'))
            
                test_accuracy = evaluate(X_test, y_test)
                print("Test Accuracy = {:.3f}".format(test_accuracy))     



def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.01

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x128.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 128), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(128))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x128. Output = 14x14x128.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x196.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 128, 196), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(196))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x1024. Output = 5x5x196.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x196. Output = 12800.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 300. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(4900, 1100), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1100))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 52.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1100, 43), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(43))
    logits   = tf.matmul(fc1, fc2_W) + fc2_b

#    # SOLUTION: Activation.
#    fc2    = tf.nn.relu(fc2)
#
#    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
#    fc3_W  = tf.Variable(tf.truncated_normal(shape=(80, 43), mean = mu, stddev = sigma))
#    fc3_b  = tf.Variable(tf.zeros(43))
#    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


if __name__ == "__main__":
    data = le_net_loaddata.load_pickled_data()
    
    data["X_train-norm"] = le_net_loaddata.gray_normalize(data["X_train"])
    data["X_validation-norm"] = le_net_loaddata.gray_normalize(data["X_validation"])
    data["X_test-norm"] = le_net_loaddata.gray_normalize(data["X_test"])
    
    LeNet_training(data["X_train-norm"],data["y_train"], data["X_validation-norm"],
                            data["y_validation"],data["X_test-norm"],data["y_test"])
    


    
    
