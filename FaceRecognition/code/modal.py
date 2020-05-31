import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import shift
from tqdm import trange
import ImageIO
import sys


def my_batch(x,y,batch_size):
    rand_index = np.random.permutation(len(x))
    n_batches = len(x)//batch_size
    for batch_index in np.array_split(rand_index,n_batches):
        x_batch,y_batch = x[batch_index],y[batch_index]
        yield x_batch,y_batch

#----Weight Initialization---#
#One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Convolution and Pooling
#Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
#Our pooling is plain old max pooling over 2x2 blocks
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def model(model_x, model_y, valid_x, valid_y, test_x,test_y,mode='train'):
    x = tf.placeholder(np.float32,[None, 64, 64])
    y_ = tf.placeholder(tf.float32, [None, 68])
    x_image = tf.reshape(x, [-1,64,64,1])


    W_conv1 = weight_variable([3,3,1,16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 64*64*16

    W_conv1_1 = weight_variable([3,3,16,32])
    b_conv1_1 = bias_variable([32])
    h_conv1_1 = tf.nn.relu(conv2d(h_conv1,W_conv1_1) + b_conv1_1) # 64*64*32
    h_pool1 = max_pool_2x2(h_conv1_1) # 32*32*32
    
    W_conv2 = weight_variable([3,3,32,48])
    b_conv2 = bias_variable([48])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 32*32*48
    
    W_conv2_1 = weight_variable([3,3,48,64])
    b_conv2_1 = bias_variable([64])
    h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1) # 32*32*64
    h_pool2 = max_pool_2x2(h_conv2_1) # 16*16*64

    W_conv3 = weight_variable([3,3,64,128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # 16*16*128
    
    W_conv3_1 = weight_variable([3,3,128,256])
    b_conv3_1 = bias_variable([256])
    h_conv3_1 = tf.nn.relu(conv2d(h_conv3, W_conv3_1) + b_conv3_1) # 16*16*256
    h_pool3 = max_pool_2x2(h_conv3_1) #8*8*256
    
    W_fc1 = weight_variable([8*8*256, 2048])
    b_fc1 = bias_variable([2048])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([2048,68])
    b_fc2 = bias_variable([68])
    y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2  

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))
    mark = tf.argmax(y_conv,1) 

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=100)
        # saver = tf.train.import_meta_graph('./my_net/epoch_18.ckpt.meta')
        if (mode == 'train'):
            batch_size = 32
            for epoch in range(30):
                get_batch = my_batch(model_x,model_y,batch_size)
                for i in range (len(model_x)//batch_size):
                    x_batch,y_batch = next(get_batch)
                    x_batch = ImageIO.shift_images(x_batch)
                    _, myloss,train_accuracy = sess.run([train_step,cross_entropy,accuracy],feed_dict = {x: x_batch, y_: y_batch, keep_prob: 0.5})
                    if i % 25 == 0:
                        print('epoch {} setp {},the train accuracy: {},the loss: {}'.format(epoch, i, train_accuracy,myloss))
                    # train_step.run(feed_dict = {x: x_batch, y_: y_batch, keep_prob: 0.5})
                print('*'*40) 
                test_correct = 0.0
                for i in trange(len(valid_x),ascii=True):
                    test_correct = test_correct + accuracy.eval(feed_dict = {x: valid_x[i].reshape((1,64,64)), y_: valid_y[i].reshape((1,68)), keep_prob: 1.})
                test_accuracy = round(test_correct/len(test_x),4)
                print('epoch {},the accuracy in valid set:{}'.format(epoch,test_accuracy))                         
                print('*'*40)
                test_correct = 0.0
                for i in trange(len(test_x),ascii=True):
                    test_correct = test_correct + accuracy.eval(feed_dict = {x: test_x[i].reshape((1,64,64)), y_: test_y[i].reshape((1,68)), keep_prob: 1.})
                test_accuracy = round(test_correct/len(test_x),4)
                print('epoch {},the accuracy in test set:{}'.format(epoch,test_accuracy))         
                print('*'*40)
                path = saver.save(sess, './my_net/epoch_{}.ckpt'.format(epoch))
                print('save path: {}'.format(path))
        elif (mode == 'test'):
            ckpt = tf.train.get_checkpoint_state("./my_net")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
            print('*'*40) 
            test_correct = 0
            for i in trange(len(test_x),ascii=True):
                test_correct = test_correct + accuracy.eval(feed_dict = {x: test_x[i].reshape((1,64,64)), y_: test_y[i].reshape((1,28)), keep_prob: 1.})
            test_accuracy = test_correct/len(test_x)
            print('the accuracy in fake test set:{}'.format(test_accuracy))         
            print('*'*40)

if __name__ == "__main__":
    if(sys.argv[1] == 'train'):
        x_train_use_ori, y_train_use_ori, x_valid_use_ori, y_valid_use_ori, ori_test_x, ori_test_y = ImageIO.loadTrainImage(0.1, False)
        '''print(x_train_use_ori.shape)
        print(y_train_use_ori.shape)
        print(x_valid_use_ori.shape)
        print(y_valid_use_ori.shape)
        print(ori_test_x.shape)
        print(ori_test_y.shape)'''
        model(x_train_use_ori,y_train_use_ori,x_valid_use_ori,y_valid_use_ori, ori_test_x, ori_test_y, mode='train')
    elif(sys.argv[1] == 'test'):
        x_train_use_ori, y_train_use_ori, x_valid_use_ori, y_valid_use_ori, ori_test_x, ori_test_y = ImageIO.loadTrainImage(0.1, False)
        model(x_train_use_ori,y_train_use_ori,x_valid_use_ori,y_valid_use_ori, ori_test_x, ori_test_y, mode='test')
