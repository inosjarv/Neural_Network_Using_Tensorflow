import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class NeuralNetwork(object):
    
    def __init__(self, filename):
        self.filename = filename

    def run(self):
        
        #Read csv file using pandas
        df = pd.read_csv(self.filename)

        #shuffled rows using reset index function of pandas
        df = df.sample(frac=1).reset_index(drop=True)


        #data for input layer
        x_data = df[["League", "Team", "S_own", "EX_own", "EN_own", "F_own", "I_own", "R_own","Opponent Team", 
            "S_opp", "EX_opp", "EN_opp", "F_opp", "I_opp", "R_opp", "Home Situation"]]

        #data for output layer
        y_labels = df['Result']

        #one hot encoding of output data
        y_labels = pd.get_dummies(pd.Series(y_labels))


        #Normalization of columns
        cols_to_norm = ['Team', 'S_own', 'EX_own', 'EN_own', 'F_own', 'I_own',
            'R_own', 'Opponent Team','S_opp', 'EX_opp', 'EN_opp', 'F_opp', 'I_opp', 'R_opp']

        x_data[cols_to_norm] = x_data[cols_to_norm].apply(lambda x: (x/x.max()))


        #Split train test data
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.1, random_state=101)


        #Dataframe to Numpy array conversion
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        y_train_cls = np.argmax(y_train,axis=1)
        X_test = np.array(X_test)
        y_test = np.array(y_test)



        n_features = 16
        n_hidden_1 = 64
        n_hidden_2 = 32
        n_classes = 2
        epochs = 200
        batch_size = 29
        learnig_rate = 0.001
        input_size = X_train.shape[0]

        #Hidden layers' weights and baises
        weights = {
            'Hidden1': tf.Variable(tf.random_normal([n_features, n_hidden_1])),
            'Hidden2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'Output': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }

        biases = {
            'Hidden1': tf.Variable(tf.random_normal([n_hidden_1])),
            'Hidden2': tf.Variable(tf.random_normal([n_hidden_2])),
            'Output': tf.Variable(tf.random_normal([n_classes]))
        }
        weights['Output'].shape

        #tensorflow Placeholder for x and y
        x = tf.placeholder(tf.float32, [None, n_features])
        y_true = tf.placeholder(tf.float32, [None, n_classes])
        y_true_cls = tf.placeholder(tf.int64,None)


        #Generate Graph for neural network

        ####matrice multipliation then addition and then activation function for both layers 
        hidden_layer_1 = tf.add(tf.matmul(x, weights["Hidden1"]), biases["Hidden1"])
        hidden_layer_1 = tf.sigmoid(hidden_layer_1)

        hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights["Hidden2"]), biases["Hidden2"])
        hidden_layer_2 = tf.sigmoid(hidden_layer_2)

        a = tf.add((tf.matmul(hidden_layer_2, weights['Output'])), biases['Output'])
        y_pred = tf.nn.softmax(a)
        y_pred_cls = tf.argmax(y_pred,axis=1)


        #Calculate error between computed outputs and the desired target outputs of the training data
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=a, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)

        #update and tune the Model's parameters in a direction so that we can minimize the Loss function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnig_rate).minimize(cost)

        #compare prediction of original output and computed outpy
        correct_prediction = tf.equal(y_true_cls,y_pred_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #Initialise all varibales declared above for session
        init = tf.global_variables_initializer()


        #Create batches of data
        def next_batch(num,data,labels):
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            idx = idx[:num]
            random_data = [data[i] for i in idx]
            random_label = [labels[i] for i in idx]
            return random_data,random_label

        #Session created
        with tf.Session() as sess:
            
            sess.run(init)
            for epoch in range(epochs):
                
                avg_cost = 0
                iteration = int(input_size/batch_size)
                for i in range(iteration):
                    x_batch, y_batch = next_batch(batch_size,X_train,y_train)
                    feed_dict = {x:x_batch,y_true:y_batch}
                    _,c = sess.run([optimizer,cost],feed_dict=feed_dict)
                
            #found cost value for each batch   
                if(epoch%50==0):
                    print("epoch====>", epoch, "    Cost====>", c)
                    accuracyv = sess.run(accuracy,feed_dict={x:X_train, y_true:y_train,y_true_cls:y_train_cls})
                    
                    #found accuracy of neural network
                    print ("accuracy: {}".format(accuracyv))

