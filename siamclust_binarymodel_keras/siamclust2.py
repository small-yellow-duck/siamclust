'''
Train a siamese convolution neural network on trios of digits from the MNIST dataset.

This code was adapted by Small Yellow Duck (https://github.com/small-yellow-duck/) from 
https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py

The similarity between two images is calculated as per Hadsell-et-al.'06 [1] by 
computing the Euclidean distance on the output of the shared network and by 
optimizing the contrastive loss (see paper for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Dot, Concatenate, Average
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Convolution2D, Convolution1D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Input, Lambda, Reshape, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
import tensorflow as tf

from keras.layers.noise import AlphaDropout
from keras import regularizers
from keras.constraints import unit_norm

from keras.layers.noise import GaussianNoise

from keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture

from mpl_toolkits.mplot3d import Axes3D

nb_filters = 16



def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


'''
vecs = np.random.random(16*3*24).reshape((16, 3, 24))
y_true = np.zeros((16,3))
vecs = K.variable(value=vecs)
y_true = K.variable(value=y_true)

vecs = K.variable(value=preds[0:1])
y_true = K.variable(value=y[0:1])

'''

def pick_best_loss(y_true, y_pred):
    #y_true is only 001, 010, 100 if two images are the same
    #y_true is 000 if three different images were chosen


    pickbest = y_pred + K.epsilon()*K.ones_like(y_pred)

    pickbest = pickbest / K.tile(K.reshape(K.max(pickbest, axis=1), (-1,1)), (1,3))
    y_pred2 = tf.floor(pickbest)

    t = K.reshape(K.sum(y_true, axis=1), (-1,1))
    y_true = y_pred2*K.tile((K.ones_like(t) - t), (1,3)) + y_true*K.tile(t, (1,3))


    #loss = -K.mean(y_true*K.log(K.clip(y_pred, K.epsilon(), 1.0)) + (1-y_true)*K.log(K.clip(1-y_pred, K.epsilon(), 1.0)))
    #loss = -K.mean(y_true*K.log(K.clip(y_pred, K.epsilon(), 1.0)) + 0.5*(1-y_true)*K.log(K.clip(1-y_pred, K.epsilon(), 1.0)))
    #loss = -K.mean(y_true*K.log(K.clip(y_pred, K.epsilon(), 1.0)))
    #loss = np.log(0.5)/3.0 - K.mean(y_true*K.log(K.clip(y_pred, K.epsilon(), 1.0)) + 0.5*(1-y_true)*K.log(K.clip((1-y_pred)/K.repeat_elements(K.expand_dims(K.sum((1-y_true)*(1-y_pred), axis=-1)), 3, axis=-1), K.epsilon(), 1.0)))

    #numerator = K.sum(y_pred*y_true,axis=-1)
    #denom1 = numerator + K.sum(y_pred*(1.0-y_true), axis=-1) - K.max(y_pred*(1.0-y_true), axis=-1)
    #denom2 = numerator + K.max(y_pred*(1.0-y_true), axis=-1)
    #loss = -K.mean(K.log(K.clip(numerator/denom1, K.epsilon(), 1.0))) -K.mean(K.log(K.clip(numerator/denom2, K.epsilon(), 1.0)))

    #loss = -K.mean(K.log(K.clip(numerator/denom2, K.epsilon(), 1.0)))


    #loss = -K.mean((1-y_true)*K.log(K.clip(1-y_pred, K.epsilon(), 1.0)))

    #loss = np.log(0.5)/3.0 - K.mean(y_true*K.log(K.clip(y_pred, K.epsilon(), 1.0)) + 0.5*(1-y_true)*K.log(K.clip((1-y_pred)/K.repeat_elements(K.expand_dims(K.sum((1-y_true)*(1-y_pred), axis=-1)), 3, axis=-1), K.epsilon(), 1.0)))

    loss = K.mean(-y_true*K.log(K.clip(y_pred, K.epsilon(), 1.0)))

    #loss = K.mean(-(1.0-y_true)*K.log(K.clip((1.0-y_pred), K.epsilon(), 1.0)) )

    #margin = 1.0
    #return K.mean(y_true * K.square(y_pred) + 0.5*(1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    return loss





def pick_two_images(x, n):
    w = np.ones(n)
    weights = w/np.sum(w)

    #w = np.ones(overlaps.shape[0])
    #w[x] = 0
    #weights = overlaps[x]*w
    #weights /= np.sum(weights)
    return pd.Series(np.random.choice(n, size=2, replace=False, p=weights))

def augment(img):
    #if np.random.rand() > 0.6:
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), np.random.randint(-15, 15), 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode= cv2.BORDER_REPLICATE)
    #need to define pts
    #if np.random.rand() > 0.6:
    #	M = cv2.getPerspectiveTransform(pts1,pts2)
    #	img = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return img

def create_triplets(X, Y, same_frac, batch_size, mode='train'):
    #if overlaps is None:

    if mode=='train':
        while True:
            idx = pd.DataFrame(data=np.zeros((batch_size, 3)).astype(np.int), columns=['img0', 'img1', 'img2'])
            idx['img0'] = np.random.choice(np.arange(X.shape[0]), size=batch_size, replace=True)

            idx[['img1', 'img2']] = idx['img0'].apply(lambda x : pick_two_images(x, X.shape[0]))
            idx.loc[int(idx.shape[0]*(1-same_frac)):, 'img1'] = idx.loc[int(idx.shape[0]*(1-same_frac)):, 'img0']
            idx = idx.apply(lambda x: pd.Series(np.random.choice(x, size=3, replace=False), index=['img0', 'img1', 'img2']), axis=1)
            idx[['notinpair0', 'notinpair1', 'notinpair2']] = idx.apply(lambda x : pd.Series([x[1]==x[2] , x[0]==x[2], x[0]==x[1]], index=['notinpair0', 'notinpair1', 'notinpair2']), axis=1)

            for i in range(3):
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda j : X[j])
                #augment by random shifts
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda x : np.roll(x, np.random.randint(-1,2), axis=np.random.randint(2)))
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda x : augment(x))

            yield [ np.concatenate(idx['img0'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img1'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img2'].values).ravel().reshape((-1,) + X.shape[1:]) ], 1.0*idx.iloc[:, 3:].values.reshape((-1, 3))

    else:
        while True:
            idx = pd.DataFrame(data=np.zeros((batch_size, 3)).astype(np.int), columns=['img0', 'img1', 'img2'])
            idx['img0'] = np.random.choice(np.arange(X.shape[0]), size=batch_size, replace=True)

            idx[['img1', 'img2']] = idx['img0'].apply(lambda x : pick_two_images(x, X.shape[0]))
            idx.loc[int(idx.shape[0]*(1-same_frac)):, 'img1'] = idx.loc[int(idx.shape[0]*(1-same_frac)):, 'img0']
            idx = idx.apply(lambda x: pd.Series(np.random.choice(x, size=3, replace=False), index=['img0', 'img1', 'img2']), axis=1)
            idx[['notinpair0', 'notinpair1', 'notinpair2']] = idx.apply(lambda x : pd.Series([x[1]==x[2] , x[0]==x[2], x[0]==x[1]], index=['notinpair0', 'notinpair1', 'notinpair2']), axis=1)


            for i in range(3):
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda j : X[j])
                #augment by random shifts
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda x : np.roll(x, np.random.randint(-1,2), axis=np.random.randint(2)))
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda x : augment(x))

            yield [ np.concatenate(idx['img0'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img1'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img2'].values).ravel().reshape((-1,) + X.shape[1:]) ], 1.0*idx.iloc[:, 3:].values.reshape((-1, 3))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)



def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


'''
def create_base_network_dense(input_dim):
    #Base network to be shared (eq. to feature extraction).
    
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq
''' 


def create_base_network(input_dim):
    # input image dimensions
    img_colours, img_rows, img_cols = input_dim

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_colours, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    #model.add(Dropout(0.1)) #0.25 #too much dropout and loss -> nan

    model.add(Flatten())

    model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
    #model.add(Dropout(0.05)) #too much dropout and loss -> nan
    model.add(Dense(32, activation='relu'))



    return model


def create_base_network2(input):	
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    d = 0.05
    r = 0.001
    x0 = Dropout(0.05)(input)
    x = Convolution2D(4, (1,1), activation='selu', padding='valid')(x0)
    x = Convolution2D(16, (3,3), activation='selu', padding='valid')(x)

    encoded = Flatten()(x)
    #x = Dense(24, activation='selu', kernel_regularizer=regularizers.l2(r))(x)
    #x = AlphaDropout(d)(x)
    #x = Dense(24, activation='selu', kernel_regularizer=regularizers.l2(r))(x)
    #x = AlphaDropout(d)(x)

    #encoded = Dense(16, activation='tanh')(x)
    #encoded = GaussianNoise(0.1)(x)

    #encoded = Lambda(lambda z : z/K.tile(K.reshape(K.sum(K.pow(z,2),axis=1), (-1,1)), (1, K.int_shape(z)[1])))(x)
    #encoded = Reshape((1, -1))(x)


    return encoded


def convnet(input_shape):	
    # number of convolutional filters to use
    #nb_filters = 8 #10 #10
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    d = 0.25 #
    d2 = 0.25 #0.25 #0.125 #0.125
    r = 0.001
    r2 = 0.003

    convnet = Sequential()
    convnet.add(Dropout(d, input_shape=input_shape))
    convnet.add(Convolution2D(nb_filters, (2,2), activation='selu', padding='valid', input_shape=input_shape))
    convnet.add(Convolution2D(nb_filters, (2,2), activation='selu', padding='valid'))
    convnet.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))


    convnet.add(Dropout(d))
    convnet.add(Convolution2D(2*nb_filters, (2,2), activation='selu', padding='valid'))
    convnet.add(Convolution2D(2*nb_filters, (3,3), activation='selu', padding='valid'))

    convnet.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    #convnet.add(Dropout(d))
    #convnet.add(Convolution2D(4*nb_filters, (3,3), activation='selu', padding='valid'))
    #convnet.add(Convolution2D(4*nb_filters, (2,2), activation='linear', padding='valid'))

    #convnet.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    #convnet.add(Dropout(d))
    #convnet.add(Convolution2D(4*nb_filters, (3,3), activation='selu', padding='valid'))
    #convnet.add(Dropout(d))
    #convnet.add(Convolution2D(4*nb_filters, (3,3), activation='selu', padding='valid'))

    #convnet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    #convnet.add(Convolution2D(2*nb_filters, (2,2), activation='selu', padding='same'))
    #convnet.add(Convolution2D(2*nb_filters, (2,2), activation='selu', padding='valid'))


    convnet.add(Dropout(d))
    convnet.add(Convolution2D(4*nb_filters, (3,3), activation='selu', padding='valid'))
    convnet.add(Convolution2D(2*nb_filters, (3,3), activation='linear', padding='valid'))
    convnet.add(Dropout(d2))
    #convnet.add(Dropout(d))
    #convnet.add(Convolution2D(2*nb_filters, (4,4), activation='linear', padding='valid'))
    #convnet.add(Dropout(d2))

    #convnet.add(Dropout(d))
    #convnet.add(Convolution2D(4*nb_filters, (3,3), activation='selu', padding='valid'))
    #convnet.add(Convolution2D(2*nb_filters, (2,2), activation='linear', padding='valid'))
    #convnet.add(Dropout(d2))


    convnet.add(Flatten())


    convnet.add(GaussianNoise(0.003))


    #convnet.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

    #convnet.add(Lambda(lambda v: K.log(v+K.epsilon())))
    #convnet.add(GaussianNoise(0.5))
    #convnet.add(Lambda(lambda v: K.exp(v)))

    #convnet.add(Dense(2*nb_filters, activation='selu'))
    #convnet.add(Dense(2*nb_filters, activation='selu'))
    #convnet.add(AlphaDropout(d))
    #convnet.add(Dense(nb_filters, activation='selu'))


    return convnet

#X_train, y_train, X_test, y_test = get_data()	
def get_data():
    img_rows, img_cols = 28, 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, img_rows, img_cols, 1)
    X_test = X_test.reshape(-1, img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    return X_train, y_train, X_test, y_test



def train(X_train, X_test): 
    batch_size = 512
    same_frac = 0.125 #  0.0625 #0.25 # 0.125 #
    epochs = 40
    train_steps_per_epoch = 50

    f = 0


    # the data, shuffled and split between train and test sets


    # input image dimensions
    img_rows, img_cols = 28, 28

    #input_dim = 784
    input_dim = (img_rows, img_cols, 1)
    nb_epoch = 12

    # network definition

    input_a = Input(shape=(img_rows, img_cols, 1))
    input_b = Input(shape=(img_rows, img_cols, 1))
    input_c = Input(shape=(img_rows, img_cols, 1))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    #vec_a = create_base_network2(input_a)
    #vec_b = create_base_network2(input_b)
    #vec_c = create_base_network2(input_c)

    cnet = convnet((img_rows, img_cols, 1))
    vec_a = cnet(input_a)
    vec_b = cnet(input_b)
    vec_c = cnet(input_c)
    #vec_a = Activation('selu')(vec_a)
    #vec_b = Activation('selu')(vec_b)
    #vec_c = Activation('selu')(vec_c)

    #vec_a2 = GaussianNoise(50.0)(vec_a)
    #vec_b2 = GaussianNoise(50.0)(vec_b)
    #vec_c2 = GaussianNoise(50.0)(vec_c)
    #vec_a2 = Lambda(lambda v: K.tanh(v[0] - v[1]))([vec_b, vec_c])
    #vec_b2 = Lambda(lambda v: K.tanh(v[0] - v[1]))([vec_a, vec_c])
    #vec_c2 = Lambda(lambda v: K.tanh(v[0] - v[1]))([vec_a, vec_b])
    #vec_a2 = GaussianNoise(0.01)(vec_a2)
    #vec_b2 = GaussianNoise(0.01)(vec_b2)
    #vec_c2 = GaussianNoise(0.01)(vec_c2)
    #vec_a2 = Lambda(lambda v: K.clip(v, -1.0, 1.0))(vec_a2)
    #vec_b2 = Lambda(lambda v: K.clip(v, -1.0, 1.0))(vec_b2)
    #vec_c2 = Lambda(lambda v: K.clip(v, -1.0, 1.0))(vec_c2)

    #vec_a3 = Dot(axes=-1, normalize=False)([vec_b, vec_c])
    #vec_b3 = Dot(axes=-1, normalize=False)([vec_c, vec_a])
    #vec_c3 = Dot(axes=-1, normalize=False)([vec_a, vec_b])

    #x = Lambda(lambda v : K.stack([K.abs(v[1]-v[2]), K.abs(v[2]-v[0]), K.abs(v[0]-v[1])], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack([K.mean(K.square(v[1]-v[2]), axis=-1), K.mean(K.square(v[2]-v[0]), axis=-1), K.mean(K.square(v[0]-v[1]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack([K.log(K.mean(K.square(v[1]-v[2]), axis=-1)), K.log(K.mean(K.square(v[2]-v[0]), axis=-1)), K.log(K.mean(K.square(v[0]-v[1]), axis=-1))], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack([K.log(K.epsilon()+K.mean(K.square(v[1]-v[2]), axis=-1)), K.log(K.mean(K.epsilon()+K.square(v[2]-v[0]), axis=-1)), K.log(K.mean(K.epsilon()+K.square(v[0]-v[1]), axis=-1))], axis=-1))([vec_a, vec_b, vec_c])

    #x = Lambda(lambda v : K.stack([v[1]*v[2], v[2]*v[0], v[0]*v[1]], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack(v, axis=-1))([vec_a3, vec_b3, vec_c3])
    #x = Activation('selu')(x)
    #x = TimeDistributed(Dense(3, activation='selu'))(x)
    #x = TimeDistributed(Dense(3, activation='sigmoid'))(x)
    #x = Flatten()(x)
    #x = Dense(3*nb_filters, activation='selu')(x)
    #x = Dense(nb_filters, activation='selu')(x)

    #x = Convolution1D(2*nb_filters, 3, strides=1, activation='selu', padding='same')(x)
    #x = Convolution1D(2*nb_filters, 3, strides=2, activation='selu', padding='same')(x)
    #x = Convolution1D(3, 1, strides=1, activation='sigmoid', padding='same')(x)
    #x = GlobalAveragePooling1D()(x)

    #probs = Activation('softmax')(x)
    #x = Convolution1D(int(nb_filters/2), 2, strides=2, activation='selu', padding='same')(x)

    #x = Reshape((-1,1))(x)
    #x = TimeDistributed(Dense(nb_filters, activation='selu'))(x)
    #x = TimeDistributed(Dense(nb_filters, activation='selu'))(x)
    #x = TimeDistributed(Dense(1, activation='linear'))(x)
    #x = Flatten()(x)
    #probs = Lambda(lambda y : K.exp(-y)/(K.repeat_elements(K.expand_dims(K.epsilon() + K.sum(K.exp(-y), axis=-1)), 3, axis=-1)))(x)


    input_u = Input(shape=(3,))
    y = Dense(6, activation='selu')(input_u)
    y = Dense(6, activation='selu')(y)
    p = Dense(3, activation='softmax')(y)

    ntop = int(nb_filters/2)
    unit = Model(inputs=input_u, outputs=p)
    x0 = Lambda(lambda v : K.stack([K.mean(K.square(v[1]-v[2]), axis=-1), K.mean(K.square(v[2]-v[0]), axis=-1), K.mean(K.square(v[0]-v[1]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    x1 = Lambda(lambda v : K.stack([K.mean(K.square(v[2]-v[0]), axis=-1), K.mean(K.square(v[0]-v[1]), axis=-1), K.mean(K.square(v[1]-v[2]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    x2 = Lambda(lambda v : K.stack([K.mean(K.square(v[0]-v[1]), axis=-1), K.mean(K.square(v[1]-v[2]), axis=-1), K.mean(K.square(v[2]-v[0]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])

#	x0 = Lambda(lambda v : K.stack([K.mean(tf.nn.top_k(K.square(v[1]-v[2]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[2]-v[0]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[0]-v[1]), k=ntop)[0], axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
#	x1 = Lambda(lambda v : K.stack([K.mean(tf.nn.top_k(K.square(v[2]-v[0]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[0]-v[1]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[1]-v[2]), k=ntop)[0], axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
#	x2 = Lambda(lambda v : K.stack([K.mean(tf.nn.top_k(K.square(v[0]-v[1]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[1]-v[2]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[2]-v[0]), k=ntop)[0], axis=-1)], axis=-1))([vec_a, vec_b, vec_c])


    x0 = unit(x0)
    x1 = unit(x1)
    x2 = unit(x2)

    x1 = Lambda(lambda v : K.stack([v[:,2], v[:,0], v[:,1]], axis=-1))(x1)
    x2 = Lambda(lambda v : K.stack([v[:,1], v[:,2], v[:,0]], axis=-1))(x2)

    probs = Average()([x0, x1, x2])







    model = Model(inputs=[input_a, input_b, input_c], outputs=probs)
    optimizer = Adam(lr=0.0003, clipnorm=0.1, clipvalue=0.1) #
    model.compile(loss=pick_best_loss, optimizer=optimizer)

    #overlaps = np.ones((X_train.shape[0], X_train.shape[0]))
    train_gen = create_triplets(X_train, None, same_frac, batch_size, mode='train')

    es_gen = create_triplets(X_test, None, same_frac, batch_size, mode='val')
    es_steps = 100
    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=0),
                ModelCheckpoint('autoenc'+str(f)+'.h5', monitor='loss', save_best_only=True, verbose=0)]
    model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=epochs, validation_data=es_gen, validation_steps=es_steps, callbacks=callbacks) #, max_q_size=10, workers=1


    #model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=epochs, validation_data=val_gen, validation_steps=100) #, max_q_size=10, workers=1

    model.load_weights('autoenc'+str(f)+'.h5')

    #train_gen.close()
    #es_gen.close()

    '''
    preds = model.predict_generator(val_gen, steps=1)	
    
    x, y = val_gen.next() 
    preds = model.predict(x)
    
    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    '''

def make_preds(X_test, y_test):
    img_rows, img_cols = 28, 28
    f= 0

    input_a = Input(shape=(img_rows, img_cols, 1))
    input_b = Input(shape=(img_rows, img_cols, 1))
    input_c = Input(shape=(img_rows, img_cols, 1))


    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    #vec_a = create_base_network2(input_a)
    #vec_b = create_base_network2(input_b)
    #vec_c = create_base_network2(input_c)

    cnet = convnet((img_rows, img_cols, 1))
    vec_a = cnet(input_a)
    vec_b = cnet(input_b)
    vec_c = cnet(input_c)
    #vec_a = Activation('selu')(vec_a)
    #vec_b = Activation('selu')(vec_b)
    #vec_c = Activation('selu')(vec_c)

    #vec_a2 = GaussianNoise(50.0)(vec_a)
    #vec_b2 = GaussianNoise(50.0)(vec_b)
    #vec_c2 = GaussianNoise(50.0)(vec_c)
    #vec_a2 = Lambda(lambda v: K.tanh(v[0] - v[1]))([vec_b, vec_c])
    #vec_b2 = Lambda(lambda v: K.tanh(v[0] - v[1]))([vec_a, vec_c])
    #vec_c2 = Lambda(lambda v: K.tanh(v[0] - v[1]))([vec_a, vec_b])
    #vec_a2 = GaussianNoise(0.01)(vec_a2)
    #vec_b2 = GaussianNoise(0.01)(vec_b2)
    #vec_c2 = GaussianNoise(0.01)(vec_c2)
    #vec_a2 = Lambda(lambda v: K.clip(v, -1.0, 1.0))(vec_a2)
    #vec_b2 = Lambda(lambda v: K.clip(v, -1.0, 1.0))(vec_b2)
    #vec_c2 = Lambda(lambda v: K.clip(v, -1.0, 1.0))(vec_c2)

    #vec_a3 = Dot(axes=-1, normalize=False)([vec_b, vec_c])
    #vec_b3 = Dot(axes=-1, normalize=False)([vec_c, vec_a])
    #vec_c3 = Dot(axes=-1, normalize=False)([vec_a, vec_b])

    #x = Lambda(lambda v : K.stack([K.abs(v[1]-v[2]), K.abs(v[2]-v[0]), K.abs(v[0]-v[1])], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack([K.mean(K.square(v[1]-v[2]), axis=-1), K.mean(K.square(v[2]-v[0]), axis=-1), K.mean(K.square(v[0]-v[1]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack([K.log(K.epsilon()+K.mean(K.square(v[1]-v[2]), axis=-1)), K.log(K.mean(K.epsilon()+K.square(v[2]-v[0]), axis=-1)), K.log(K.mean(K.epsilon()+K.square(v[0]-v[1]), axis=-1))], axis=-1))([vec_a, vec_b, vec_c])

    #x = Lambda(lambda v : K.stack([K.mean(K.square(v[1]-v[2]), axis=-1), K.mean(K.square(v[2]-v[0]), axis=-1), K.mean(K.square(v[0]-v[1]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack([v[1]*v[2], v[2]*v[0], v[0]*v[1]], axis=-1))([vec_a, vec_b, vec_c])
    #x = Lambda(lambda v : K.stack(v, axis=-1))([vec_a3, vec_b3, vec_c3])
    #x = Activation('selu')(x)
    #x = TimeDistributed(Dense(3, activation='selu'))(x)
    #x = TimeDistributed(Dense(3, activation='sigmoid'))(x)
    #x = Flatten()(x)
    #x = Dense(3*nb_filters, activation='selu')(x)
    #x = Dense(nb_filters, activation='selu')(x)

    #x = Convolution1D(2*nb_filters, 3, strides=1, activation='selu', padding='same')(x)
    #x = Convolution1D(2*nb_filters, 3, strides=2, activation='selu', padding='same')(x)
    #x = Convolution1D(3, 1, strides=1, activation='sigmoid', padding='same')(x)
    #x = GlobalAveragePooling1D()(x)

    #probs = Activation('softmax')(x)
    #x = Convolution1D(int(nb_filters/2), 2, strides=2, activation='selu', padding='same')(x)

    #x = Reshape((-1,1))(x)
    #x = TimeDistributed(Dense(nb_filters, activation='selu'))(x)
    #x = TimeDistributed(Dense(nb_filters, activation='selu'))(x)
    #x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    #x = Flatten()(x)
    #probs = Lambda(lambda x : x/(K.repeat_elements(K.expand_dims(K.sum(x, axis=-1)), 3, axis=-1)+K.epsilon()))(x)

    #x = Reshape((-1,1))(x)
    #x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    #x = Flatten()(x)
    #x = Dense(int(nb_filters/2), activation='selu')(x)
    #x = Dense(int(nb_filters/2), activation='selu')(x)
    #probs = Dense(3, activation='softmax')(x)

    #input_u = Input(shape=(3,))
    #y = Dense(int(nb_filters/2), activation='selu')(input_u)
    #y = Dense(int(nb_filters/2), activation='selu')(y)
    #p = Dense(3, activation='softmax')(y)

    #unit = Model(inputs=input_u, outputs=p)
    #probs = unit(x)


    #x = Reshape((1,-1))(x)
    #x = Convolution1D(2*nb_filters, 1, strides=1, activation='selu', padding='same')(x)
    #x = Convolution1D(2*nb_filters, 1, strides=1, activation='selu', padding='same')(x)
    #x = Convolution1D(3, 1, strides=1, activation='softmax', padding='same')(x)
    #probs = Flatten()(x)

    #probs = Lambda(lambda x : x/(K.repeat_elements(K.expand_dims(K.sum(x, axis=-1)), 3, axis=-1)+K.epsilon()))(x)
    #probs = Lambda(lambda x : 1.0-x/(K.repeat_elements(K.expand_dims(K.sum(x, axis=-1)), 3, axis=-1)+K.epsilon()))(x)


    input_u = Input(shape=(3,))
    y = Dense(6, activation='selu')(input_u)
    y = Dense(6, activation='selu')(y)
    p = Dense(3, activation='softmax')(y)

    ntop = int(nb_filters/2)
    unit = Model(inputs=input_u, outputs=p)
    x0 = Lambda(lambda v : K.stack([K.mean(K.square(v[1]-v[2]), axis=-1), K.mean(K.square(v[2]-v[0]), axis=-1), K.mean(K.square(v[0]-v[1]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    x1 = Lambda(lambda v : K.stack([K.mean(K.square(v[2]-v[0]), axis=-1), K.mean(K.square(v[0]-v[1]), axis=-1), K.mean(K.square(v[1]-v[2]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    x2 = Lambda(lambda v : K.stack([K.mean(K.square(v[0]-v[1]), axis=-1), K.mean(K.square(v[1]-v[2]), axis=-1), K.mean(K.square(v[2]-v[0]), axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    #x0 = Lambda(lambda v : K.stack([K.mean(tf.nn.top_k(K.square(v[1]-v[2]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[2]-v[0]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[0]-v[1]), k=ntop)[0], axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    #x1 = Lambda(lambda v : K.stack([K.mean(tf.nn.top_k(K.square(v[2]-v[0]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[0]-v[1]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[1]-v[2]), k=ntop)[0], axis=-1)], axis=-1))([vec_a, vec_b, vec_c])
    #x2 = Lambda(lambda v : K.stack([K.mean(tf.nn.top_k(K.square(v[0]-v[1]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[1]-v[2]), k=ntop)[0], axis=-1), K.mean(tf.nn.top_k(K.square(v[2]-v[0]), k=ntop)[0], axis=-1)], axis=-1))([vec_a, vec_b, vec_c])


    x0 = unit(x0)
    x1 = unit(x1)
    x2 = unit(x2)

    x1 = Lambda(lambda v : K.stack([v[:,2], v[:,0], v[:,1]], axis=-1))(x1)
    x2 = Lambda(lambda v : K.stack([v[:,1], v[:,2], v[:,0]], axis=-1))(x2)

    probs = Average()([x0, x1, x2])


    #x = Reshape((-1,1))(x)
    #x = TimeDistributed(Dense(nb_filters, activation='selu'))(x)
    #x = TimeDistributed(Dense(nb_filters, activation='selu'))(x)
    #x = TimeDistributed(Dense(1, activation='linear'))(x)
    #x = Flatten()(x)
    #probs = Lambda(lambda y : K.exp(-y)/(K.repeat_elements(K.expand_dims(K.epsilon() + K.sum(K.exp(-y), axis=-1)), 3, axis=-1)))(x)


    model = Model(inputs=[input_a, input_b, input_c], outputs=probs)
    optimizer = Adam(lr=0.0003) #RMSprop()
    model.compile(loss=pick_best_loss, optimizer=optimizer, metrics=['accuracy'])

    model.load_weights('autoenc'+str(f)+'.h5')

    #model = [model.layers[0], model.layers[-2]]


    batch_size = 256
    test_gen = create_triplets(X_test, None, 1.0, batch_size, mode='val')

    print('log loss, accuracy')
    print(model.evaluate_generator(test_gen, steps=20))



    input = Input(shape=(img_rows, img_cols, 1))
    output = model.layers[3](input)
    model2 = Model(inputs=input, outputs=output)


    vecs = model2.predict(1.0*X_test)

    print(np.std(vecs, axis=0))

    pca = PCA(n_components=3, svd_solver='arpack', copy=True, whiten=True)
    pca.fit(vecs[:, :])

    pca_vecs = pca.transform(vecs[:, :])

    kmeans_dims = 11


    clf = KMeans(n_clusters=kmeans_dims, random_state=0, max_iter=1000).fit(vecs)
    preds = clf.predict(vecs)
    #preds = DBSCAN(eps=5.0).fit_predict(vecs)
    #preds = MeanShift().fit_predict(pca_vecs)
    #preds = KMeans(n_clusters=kmeans_dims, random_state=0).fit_predict(vecs)
    #gmm = BayesianGaussianMixture(n_components=4).fit(pca_vecs)
    #preds = gmm.predict(pca_vecs)
    #preds = AgglomerativeClustering(n_clusters=kmeans_dims, linkage='average', affinity='euclidean').fit_predict(vecs)

    print(len(set(preds)))

    print('0')
    idx = np.where(y_test==0)
    print(preds[idx][0:40])
    print('1')
    idx = np.where(y_test==1)
    print(preds[idx][0:40])
    print('2')
    idx = np.where(y_test==2)
    print(preds[idx][0:40])
    print('3')
    idx = np.where(y_test==3)
    print(preds[idx][0:40])
    print('4')
    idx = np.where(y_test==4)
    print(preds[idx][0:40])
    print('5')
    idx = np.where(y_test==5)
    print(preds[idx][0:40])
    print('6')
    idx = np.where(y_test==6)
    print(preds[idx][0:40])
    print('7')
    idx = np.where(y_test==7)
    print(preds[idx][0:40])
    print('8')
    idx = np.where(y_test==8)
    print(preds[idx][0:40])
    print('9')
    idx = np.where(y_test==9)
    print(preds[idx][0:40])


    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=pca_vecs[y_test==1][:, 0], ys=pca_vecs[y_test==1][:, 1], zs=pca_vecs[y_test==1][:, 2], zdir='z', s=5, c='k', depthshade=True, label='1')
    ax.scatter(xs=pca_vecs[y_test==7][:, 0], ys=pca_vecs[y_test==7][:, 1], zs=pca_vecs[y_test==7][:, 2], zdir='z', s=5, c='r', depthshade=True, label='7')
    ax.scatter(xs=pca_vecs[y_test==4][:, 0], ys=pca_vecs[y_test==4][:, 1], zs=pca_vecs[y_test==4][:, 2], zdir='z', s=5, c='b', depthshade=True, label='4')
    ax.scatter(xs=pca_vecs[y_test==9][:, 0], ys=pca_vecs[y_test==9][:, 1], zs=pca_vecs[y_test==9][:, 2], zdir='z', s=5, c='g', depthshade=True, label='9')
    ax.scatter(xs=pca_vecs[y_test==0][:, 0], ys=pca_vecs[y_test==0][:, 1], zs=pca_vecs[y_test==0][:, 2], zdir='z', s=5, c='m', depthshade=True, label='0')
    plt.legend()


    ''' 	
    n = vecs.shape[0]
    q = int(vecs.shape[1]/2)
    overlaps = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            overlaps[i,j] = np.mean(sort((vecs[i] - vecs[j])**2)[q:])
            overlaps[i,j] = overlaps[j,i]

    clf = KMeans(n_clusters=kmeans_dims, random_state=0, max_iter=1000, precomputed=True).fit(overlaps)
    preds = clf.predict(overlaps)	
    
    o = np.argsort(np.sum(overlaps, axis=1))
    



    pca = PCA(n_components=4, svd_solver='arpack', copy=True, whiten=False)
    pca.fit(vecs[:, :])
    
    pca_vecs = pca.transform(vecs[:, :])
    
    ax0 = 1
    ax1 = 3
    
    plt.close('all')
    plt.plot(pca_vecs[y_test==1][:, ax0], pca_vecs[y_test==1][:, ax1], 'k+')
    plt.plot(pca_vecs[y_test==7][:, ax0], pca_vecs[y_test==7][:, ax1], 'r+')
    plt.plot(pca_vecs[y_test==4][:, ax0], pca_vecs[y_test==4][:, ax1], 'b+')
    plt.plot(pca_vecs[y_test==9][:, ax0], pca_vecs[y_test==9][:, ax1], 'g+')
    plt.plot(pca_vecs[y_test==0][:, ax0], pca_vecs[y_test==0][:, ax1], 'm+')


    vecs = model2.predict(1.0*X_test)
    kmeans_dims = 10
    kmeans = KMeans(n_clusters=kmeans_dims, random_state=0).fit(vecs)
    
    diff = np.mean(np.sqrt((np.tile(vecs.reshape((vecs.shape[0], 1, vecs.shape[1])), (1, kmeans_dims, 1)) / np.tile(kmeans.cluster_centers_, (vecs.shape[0], 1, 1)) - 1.0)**2), axis=2)


    i = 9
    plt.close()
    plt.plot(diff[y_test!=i][:, 0], diff[y_test!=i][:, 2], 'k+')
    plt.plot(diff[y_test==i][:, 0], diff[y_test==i][:, 2], 'r+')	
    
    i = 4
    plt.close()
    plt.plot(vecs[y_test!=i][:, 0], vecs[y_test!=i][:, 4], 'k+')
    plt.plot(vecs[y_test==i][:, 0], vecs[y_test==i][:, 4], 'r+')	
    
    
    
    i = 1
    plt.close()
    plt.plot(vecs[y_test!=i][:, 7], vecs[y_test!=i][:, 6], 'k+')
    plt.plot(vecs[y_test==i][:, 7], vecs[y_test==i][:, 6], 'r+')		

    
    ax0 = 0
    ax1 = 1
    
    plt.close('all')
    plt.plot(pca_vecs[y_test==1][:, ax0], pca_vecs[y_test==1][:, ax1], 'k+')
    plt.plot(pca_vecs[y_test==7][:, ax0], pca_vecs[y_test==7][:, ax1], 'r+')
    plt.plot(pca_vecs[y_test==4][:, ax0], pca_vecs[y_test==4][:, ax1], 'b+')
    plt.plot(pca_vecs[y_test==9][:, ax0], pca_vecs[y_test==9][:, ax1], 'g+')
    plt.plot(pca_vecs[y_test==0][:, ax0], pca_vecs[y_test==0][:, ax1], 'm+')


    plt.close('all')
    plt.plot(pca_vecs[:, 3], pca_vecs[:, 1], 'k+')
    plt.plot(pca_vecs[y_test==1][:, ax0], pca_vecs[y_test==1][:, ax1], 'r+')
    plt.plot(pca_vecs[y_test==3][:, ax0], pca_vecs[y_test==3][:, ax1], 'b+')
    plt.plot(pca_vecs[y_test==8][:, ax0], pca_vecs[y_test==8][:, ax1], 'g+')
        
    i = 0
    plt.close('all')
    plt.plot(pca_vecs[y_test!=i][:, 0], pca_vecs[y_test!=i][:, 2], 'k+')
    plt.plot(pca_vecs[y_test==i][:, 0], pca_vecs[y_test==i][:, 2], 'r+')	


    ax0 = 2
    ax1 = 3
    
    plt.close('all')
    plt.plot(vecs[y_test==1][:, ax0], vecs[y_test==1][:, ax1], 'k+')
    plt.plot(vecs[y_test==7][:, ax0], vecs[y_test==7][:, ax1], 'r+')
    plt.plot(vecs[y_test==4][:, ax0], vecs[y_test==4][:, ax1], 'b+')
    plt.plot(vecs[y_test==9][:, ax0], vecs[y_test==9][:, ax1], 'g+')
    plt.plot(vecs[y_test==0][:, ax0], vecs[y_test==0][:, ax1], 'm+')	
    '''