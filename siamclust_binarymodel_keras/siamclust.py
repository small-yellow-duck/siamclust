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
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Dropout, Input, Lambda, Reshape, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
import tensorflow as tf

from keras.layers.noise import AlphaDropout
from keras import regularizers

from keras.layers.noise import GaussianNoise

from keras.callbacks import EarlyStopping, ModelCheckpoint

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

def pick_best_loss(y_true, vecs):
    #y_true is only 001, 010, 100 if two images are the same
    #y_true is 000 if three different images were chosen


    y_true = K.reshape(y_true, (-1, 3))


    overlaps = []
    for i in range(3):
        j = (i+1) % 3
        k = (i+2) % 3
        #print(str(i), str(j), str(k))
        overlaps += [(K.mean(K.pow(vecs[:, j, :] - vecs[:, k, :], 2), axis=-1))]
        #overlaps += [(K.mean(K.abs(vecs[:, j, :] - vecs[:, k, :]), axis=-1))]
        #overlaps += [-K.mean(vecs[:, j, :]*vecs[:, k, :],axis=-1)/K.sqrt(K.mean(K.square(vecs[:, j, :]), axis=-1))/K.sqrt(K.mean(K.square(vecs[:, k, :]), axis=-1))]
    overlaps = K.stack(overlaps, axis = 1)


    '''
    overlaps = []
    overlaps += [(K.mean(K.pow(vecs[:, 1, :] - vecs[:, 2, :], 2), axis=-1))]
    overlaps += [(K.mean(K.pow(vecs[:, 2, :] - vecs[:, 0, :], 2), axis=-1))]
    overlaps += [(K.mean(K.pow(vecs[:, 0, :] - vecs[:, 1, :], 2), axis=-1))]
    overlaps = K.stack(overlaps, axis = 1)
    '''


    pickbest = overlaps + K.epsilon()*K.ones_like(overlaps)

    pickbest = K.tile(K.reshape(K.min(pickbest, axis=1), (-1,1)), (1,3)) / pickbest
    y_pred = tf.floor(pickbest)

    t = K.reshape(K.sum(y_true, axis=1), (-1,1))
    y_pred = y_pred*K.tile((K.ones_like(t) - t), (1,3)) + y_true*K.tile(t, (1,3))


    #loss = K.mean(K.tile(y_pred, (1,1, EMB))*overlaps - 0.5*K.tile((K.ones_like(y_pred)-y_pred), (1,1, EMB))*overlaps)
    loss = K.mean(K.sum(y_pred*overlaps - 0.5*(K.ones_like(y_pred)-y_pred)*overlaps, axis=1))
    #loss = K.mean(y_pred*overlaps - 0.5*(K.ones_like(y_pred)-y_pred)*(overlaps - K.tile(K.reshape(K.sum(y_pred*overlaps, axis=-1), (-1,1)), (1,3))) )

    #t = K.sum((1-y_pred)*overlaps, axis=1)/2
    #loss = K.mean(K.sum((1-y_pred)*(overlaps - K.tile(K.reshape(t, (-1, 1)), (1, 3))) - (1-y_pred)*overlaps, axis=1))

    #loss = K.mean(-K.square(vecs[:, :, :]))/3.0 + K.mean(y_pred*overlaps)
    #margin = 2.0
    #loss = K.mean(y_pred * K.square(overlaps) - 0.5*(1 - y_pred) * K.square(overlaps - K.tile(K.reshape(K.sum(y_pred*overlaps, axis=-1), (-1,1)), (1,3))))
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

            yield [np.concatenate(idx['img0'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img1'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img2'].values).ravel().reshape((-1,) + X.shape[1:]) ], idx.iloc[:, 3:].values.reshape((-1, 3, 1))

    else:
        while True:
            idx = pd.DataFrame(data=np.zeros((batch_size, 3)).astype(np.int), columns=['img0', 'img1', 'img2'])
            idx['img0'] = np.random.choice(np.arange(X.shape[0]), size=batch_size, replace=True)

            idx[['img1', 'img2']] = idx['img0'].apply(lambda x : pick_two_images(x, X.shape[0]))
            idx.loc[:, 'img1'] = idx.loc[:, 'img0']

            idx = idx.apply(
                lambda x: pd.Series(np.random.choice(x, size=3, replace=False), index=['img0', 'img1', 'img2']), axis=1)

            idx[['notinpair0', 'notinpair1', 'notinpair2']] = idx.apply(
                lambda x: pd.Series([x[1] == x[2], x[0] == x[2], x[0] == x[1]],
                                    index=['notinpair0', 'notinpair1', 'notinpair2']), axis=1)

            for i in range(3):
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda j: X[j])
                #augment by random shifts
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda x: np.roll(x, np.random.randint(-1,2), axis=np.random.randint(2)))
                idx['img'+str(i)] = idx['img'+str(i)].apply(lambda x: augment(x))

            yield [np.concatenate(idx['img0'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img1'].values).ravel().reshape((-1,) + X.shape[1:]), np.concatenate(idx['img2'].values).ravel().reshape((-1,) + X.shape[1:]) ], idx.iloc[:, 3:].values.reshape((-1, 3, 1))


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
    nb_filters = 8 #10 #10
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    d = 0.05
    r = 0.0
    r2 = 0.001

    convnet = Sequential()
    convnet.add(Dropout(d, input_shape=input_shape))
    convnet.add(Convolution2D(nb_filters, (2,2), activation='selu', padding='same', input_shape=input_shape))
    convnet.add(Convolution2D(nb_filters, (3,3), activation='selu',  padding='same'))
    convnet.add(MaxPooling2D(pool_size=(2, 2)))


    #convnet.add(AlphaDropout(d))
    convnet.add(Convolution2D(2*nb_filters, (3,3), activation='selu', padding='same'))
    convnet.add(MaxPooling2D(pool_size=(2, 2)))

    convnet.add(Convolution2D(4*nb_filters, (3,3), activation='tanh', padding='same'))


    #convnet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


    #convnet.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    convnet.add(GlobalAveragePooling2D())

    convnet.add(GaussianNoise(0.1))





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



def train(X_train):	
    batch_size = 256
    same_frac = 0.25 #0.0625 #0.125
    epochs = 50
    train_steps_per_epoch = 100

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

    #vecs = concatenate([vec_a, vec_b, vec_c])
    #vecs = Reshape((3,-1))(vecs)

    vecs = Lambda(lambda v : K.stack(v, axis=1))([vec_a, vec_b, vec_c])
    model = Model(inputs=[input_a, input_b, input_c], outputs=vecs)
    optimizer = Adam(lr=0.0003) #RMSprop()
    model.compile(loss=pick_best_loss, optimizer=optimizer)

    #overlaps = np.ones((X_train.shape[0], X_train.shape[0]))
    train_gen = create_triplets(X_train, None, same_frac, batch_size, mode='train')

    es_gen = create_triplets(X_train, None, same_frac, batch_size, mode='val')
    es_steps = 100
    callbacks = [EarlyStopping(monitor='loss', patience=4, verbose=0),
                ModelCheckpoint('autoenc'+str(f)+'.h5', monitor='loss', save_best_only=True, verbose=0)]
    model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=epochs, validation_data=es_gen, validation_steps=es_steps, callbacks=callbacks) #, max_q_size=10, workers=1


    #model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=epochs, validation_data=val_gen, validation_steps=100) #, max_q_size=10, workers=1

    model.load_weights('autoenc'+str(f)+'.h5')

    train_gen.close()
    es_gen.close()

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

    cnet = convnet((img_rows, img_cols, 1))
    vec_a = cnet(input_a)
    vec_b = cnet(input_b)
    vec_c = cnet(input_c)


    vecs = Lambda(lambda v : K.stack(v, axis=1))([vec_a, vec_b, vec_c])
    model = Model(inputs=[input_a, input_b, input_c], outputs=vecs)
    optimizer = Adam(lr=0.0003) #RMSprop()
    model.compile(loss=pick_best_loss, optimizer=optimizer)

    model.load_weights('autoenc'+str(f)+'.h5')

    #model = [model.layers[0], model.layers[-2]]

    preds = model.predict([X_test[0:24:3], X_test[1:24:3], X_test[2:24:3]])
    preds = preds.reshape((preds.shape[0]*preds.shape[1], -1))

    print(y_test[0:25])
    print('different')
    print('0 - 1:', np.mean((preds[0] - preds[1])**2))
    print('0 - 2:', np.mean((preds[0] - preds[2])**2))
    print('7 - 8:', np.mean((preds[7] - preds[8])**2))
    print('8 - 9:', np.mean((preds[8] - preds[9])**2))

    print('same')
    print('4 - 6:', np.mean((preds[4] - preds[6])**2))
    print('7 - 9:', np.mean((preds[7] - preds[9])**2))
