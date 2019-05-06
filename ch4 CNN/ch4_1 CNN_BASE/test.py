import cnn_utils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
"""
定义卷积核
"""
def initialize_parameter():
    W1 = tf.get_variable('W1',shape=[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer())
    #tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.07)(W1))
    W2 = tf.get_variable('W2', shape=[2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer())
    #tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.07)(W2))
    parameters={'W1':W1,
                'W2':W2}
    return parameters
"""
 创建输入输出placeholder
"""
def creat_placeholder(n_xH,n_xW,n_C0,n_y):
    X=tf.placeholder(tf.float32,shape=(None,n_xH,n_xW,n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    return X,Y
"""
传播过程
"""
def forward_propagation(X,parameters):
    W1=parameters['W1']
    W2 = parameters['W2']
    Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    print('第一次卷积尺寸={}'.format(Z1.shape))
    A1=tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1, 8, 8, 1], padding='VALID')
    print('第一次池化尺寸={}'.format(P1.shape))
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    print('第二次卷积尺寸={}'.format(Z2.shape))
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    print('第二次池化尺寸={}'.format(P2.shape))
    P_flatten=tf.contrib.layers.flatten(P2)
    Z3=tf.contrib.layers.fully_connected(P_flatten,6,activation_fn=None)
    return Z3
"""
计算损失值
"""
def compute_cost(Z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    return cost
"""
模型应用过程
"""
def model(learning_rate,num_pochs,minibatch_size):
    train_set_x_orig, train_y_orig, test_set_x_orig, test_y_orig, classes=cnn_utils.load_dataset()
    train_x = train_set_x_orig / 255
    test_x = test_set_x_orig / 255
    # 转换成one-hot
    train_y=cnn_utils.convert_to_one_hot(train_y_orig,6).T
    test_y = cnn_utils.convert_to_one_hot(test_y_orig, 6).T
 
    m,n_xH, n_xW, n_C0=train_set_x_orig.shape
    n_y=train_y.shape[1]
    X, Y = creat_placeholder(n_xH, n_xW, n_C0, n_y)
    parameters = initialize_parameter()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    ##带正则项误差
    # tf.add_to_collection("losses", cost)
    # loss = tf.add_n(tf.get_collection('losses'))
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    costs=[]
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_pochs):
            minibatch_cost=0
            num_minibatches=int(m/minibatch_size)
            minibatchs=cnn_utils.random_mini_batches(train_x,train_y,)
            for minibatch in minibatchs:
                (mini_batch_X, mini_batch_Y)=minibatch
                _,temp_cost = sess.run([optimizer,cost], feed_dict={X:mini_batch_X , Y: mini_batch_Y})
                minibatch_cost+=temp_cost/num_minibatches
            if epoch%5==0:
                print('after {} epochs minibatch_cost={}'.format(epoch,minibatch_cost))
                costs.append(minibatch_cost)
        #predict_y=tf.argmax(Z3,1)####1 represent hang zuida
        corect_prediction=tf.equal(tf.argmax(Z3,1),tf.argmax(Y,1))
        accuarcy=tf.reduce_mean(tf.cast(corect_prediction,'float'))
        train_accuarcy=sess.run(accuarcy,feed_dict={X:train_x,Y:train_y})
        test_accuarcy = sess.run(accuarcy, feed_dict={X: test_x, Y: test_y})
        print('train_accuarcy={}'.format(train_accuarcy))
        print('test_accuarcy={}'.format(test_accuarcy))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations ')
    plt.title('learning rate={}'.format(learning_rate))
    plt.show()
 
def test_model():
    model(learning_rate=0.009,num_pochs=100,minibatch_size=32)
def test():
    ########test forward
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    with tf.Session() as sess:
        X,Y=creat_placeholder(64,64,3,6)
        parameters=initialize_parameter()
        Z3=forward_propagation(X,parameters)
        cost=compute_cost(Z3,Y)
        init = tf.global_variables_initializer()
        sess.run(init)
        Z3,cost=sess.run([Z3,cost],feed_dict={X:np.random.randn(2,64,64,3),Y:np.random.randn(2,6)})
        print('Z3={}'.format(Z3))
        print('cost={}'.format(cost))
    ################
 
 
if __name__=='__main__':
    #test()
    test_model()
