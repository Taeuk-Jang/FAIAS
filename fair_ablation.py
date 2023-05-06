'''
This code is written by Xiaoqian Joy Wang,
based on the code by Jinsung Yoon
Date: Jan 1th 2019
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "IINVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu

---------------------------------------------------

Instance-wise Variable Selection (INVASE) - without baseline networks
'''

# %% Necessary packages
# 1. tf.keras
import tensorflow as tf
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K
import keras
#from keras.backend.tensorflow_backend import set_session

# 2. Others
import sklearn.metrics as sklm
import numpy as np
import os
# from function_0 import *
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score



# %% Define FAIR class
class FAIR():
    # 1. Initialization
    '''
    x_train: training selector
    data_type: Syn1 to Syn 6
    '''

    def __init__(self, x_train, y_train, sens_idx, sess, batch_size = 100, epochs=800, learning_rate = 1e-3, learning_rate_theta = 1e-3, logging = None):
        self.latent_dim_sel = 100  # Dimension of the sampling (selector) network
        self.latent_dim_pred = 200  # Dimension of fair (predictor) network
        self.logging = logging
        self.batch_size = batch_size  # Batch size
        self.epochs = epochs  # Epoch size (large epoch is needed due to the policy gradient framework)
        # self.lamda = lamda  # Hyper-parameter for the number of selected features

        self.input_shape = x_train.shape[1]  # Input dimension
        self.output_shape = y_train.shape[1]  # Input dimension

        # Activation (For Syn1 and 2, relu, others, selu)
        # self.activation = 'relu' if data_type in ['Syn1', 'Syn2'] else 'selu'
        self.activation = 'selu'

        # Use Adam optimizer with learning rate = 0.0001
        self.learning_rate_theta = learning_rate_theta
        self.learning_rate = learning_rate
#        self.learning_rate_theta = 1e-3
#        self.learning_rate = 1e-3
        
        optimizer = Adam(self.learning_rate)
        
        from keras import backend as K
        K.set_session(sess)
        
        # Build and compile the predictor (for fairness)
        self.predictor = self.build_predictor()
        # Use categorical cross entropy as the loss
        self.predictor.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        self.classes_ = np.unique(y_train)
        # Build the selector
#        self.theta = np.random.rand(self.input_shape,)
#        self.best_theta = np.zeros(self.input_shape,)
        self.sens_idx = sens_idx

        


    # %% Predictor (Fair)
    def build_predictor(self, module = None):
        
        
        model = Sequential()
        '''
        model.add(Dense(self.latent_dim_pred, activation= 'relu', name='dense1', kernel_regularizer=regularizers.l2(1e-3),
                        input_dim=self.input_shape))
        model.add(Dense(self.latent_dim_pred, activation= 'relu', name='dense2', kernel_regularizer=regularizers.l2(1e-3)))
        # model.add(BatchNormalization())  # Use Batch norm for preventing overfitting
        # model.add(Dense(self.latent_dim_pred, activation=self.activation, name='dense2', kernel_regularizer=regularizers.l2(1e-3)))
        # model.add(BatchNormalization())
        # model.add(Dense(self.latent_dim_pred, activation=self.activation, name='dense3',
        #                 kernel_regularizer=regularizers.l2(1e-3)))
        # model.add(BatchNormalization())
        model.add(Dense(self.output_shape, activation='softmax', name='dense4', kernel_regularizer=regularizers.l2(1e-3)))

        '''
        model.add(Dense(self.latent_dim_pred, activation= 'selu', name='dense1', kernel_regularizer=regularizers.l2(1e-3),
                        input_dim=self.input_shape))
        model.add(Dense(self.latent_dim_pred, activation= 'selu', name='dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.latent_dim_pred, activation= 'selu', name='dense3', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.latent_dim_pred, activation= 'selu', name='dense4', kernel_regularizer=regularizers.l2(1e-3)))
        # model.add(BatchNormalization())  # Use Batch norm for preventing overfitting
        # model.add(Dense(self.latent_dim_pred, activation=self.activation, name='dense2', kernel_regularizer=regularizers.l2(1e-3)))
        # model.add(BatchNormalization())
        # model.add(Dense(self.latent_dim_pred, activation=self.activation, name='dense3',
        #                 kernel_regularizer=regularizers.l2(1e-3)))
        # model.add(BatchNormalization())
        model.add(Dense(self.output_shape, activation='softmax', name='dense5', kernel_regularizer=regularizers.l2(1e-3)))

        #model.summary()
        
        #model = keras.models.load_model('image_model/base_300_2.hdf5')
        # There are two inputs to be used in the predictor
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
#        select = Input(shape=(self.input_shape,), dtype='float32')

        # Element-wise multiplication
#        input = Multiply()([feature, select])
        pred = model(feature)

        return Model(feature, pred)
        # return Model([feature, select_with, select_without], tf.concat(axis=1,values=[pred_with, pred_without]))

        # return Model([feature, select_with, select_without], tf.concat(axis=1,values=[pred_with, pred_without]))

    # %% Training procedure
    def train(self, x_train, y_train):

        # For each epoch (actually iterations)
        for epoch in range(self.epochs):

            # %% Train Predictor
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx, :]
            y_batch = y_train[idx, :]

            self.predictor.train_on_batch(x_batch, y_batch) #l_pred

    # %% Prediction Results
    def get_prediction(self, x_test):

        out_pred = self.predictor.predict(x_test)

        return np.asarray(out_pred)
              


def fair_shapley(x_train, y_train, x_test, sens_idx, session=None, batch_size=200, epochs=5000, gpu_id='2', learning_rate = 1e-3, learning_rate_theta = 1e-3, logging = None, best_eq = 100):

    # Use CUDA
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # tf.tf.keras.backend.set_session(session)

    # Hyperparameter
    lamda = 3
    print(epochs)
    # 1. FAIR Class call
    FAIR_Alg = FAIR(x_train, y_train, x_test, sens_idx, session, batch_size, epochs, learning_rate, learning_rate_theta)

    # 2. Algorithm training
    FAIR_Alg.train(x_train, y_train, x_test)

    # 1. Get the selection probability on the testing set


    # 3. Prediction
    out_predict = FAIR_Alg.get_prediction(x_test)

    return out_predict