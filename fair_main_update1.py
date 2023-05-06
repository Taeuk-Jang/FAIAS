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
# 1. Keras
from tensorflow.keras.layers import Input, Dense, Multiply
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.backend import set_session
from tensorflow import keras
# 2. Others
import tensorflow as tf
import numpy as np
import os
# from function_0 import *
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import sklearn.metrics as sklm

# %% Define FAIR class
class FAIR():
    # 1. Initialization
    '''
    x_train: training selector
    data_type: Syn1 to Syn 6
    '''

    def __init__(self, x_train, y_train, sens_idx, sess, batch_size, epochs, lr, lr_t):
        self.latent_dim_sel = 100  # Dimension of the sampling (selector) network
        self.latent_dim_pred = 200  # Dimension of fair (predictor) network

        self.batch_size = batch_size  # Batch size
        self.epochs = epochs  # Epoch size (large epoch is needed due to the policy gradient framework)
        # self.lamda = lamda  # Hyper-parameter for the number of selected features

        self.input_shape = x_train.shape[1]  # Input dimension
        self.output_shape = y_train.shape[1]  # Input dimension

        # Activation (For Syn1 and 2, relu, others, selu)
        # self.activation = 'relu' if data_type in ['Syn1', 'Syn2'] else 'selu'
        self.activation = 'selu'

        # Use Adam optimizer with learning rate = 0.0001
#        self.learning_rate = 0.0001
#        optimizer = Adam(0.0005)
        self.lr_t = lr_t
        optimizer = Adam(lr)
        # Build and compile the predictor (for fairness)
        self.predictor = self.build_predictor()
        # Use categorical cross entropy as the loss
        self.predictor.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        #self.best_predictor = self.predictor
        #self.selector = self.build_selector()

        # Build the selector
#         self.theta = np.random.rand(self.input_shape,)
        self.theta = np.ones(self.input_shape,)
        self.best_theta = 0
        self.sens_idx = sens_idx
        self.best_saved = 0

        K.set_session(sess)


    # %% Predictor (Fair)
    def build_predictor(self):

        model = Sequential()

        model.add(Dense(self.latent_dim_pred, activation= 'selu', kernel_initializer='he_normal', name='dense1', kernel_regularizer=regularizers.l2(1e-4),
                        input_dim=self.input_shape))
        model.add(Dense(self.latent_dim_pred, activation= 'selu', kernel_initializer='he_normal', name='dense2', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(self.latent_dim_pred, activation= 'selu', kernel_initializer='he_normal', name='dense3', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(self.latent_dim_pred, activation= 'selu', kernel_initializer='he_normal', name='dense4', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(self.output_shape, activation='softmax', kernel_initializer='he_normal',name='dense5', kernel_regularizer=regularizers.l2(1e-4)))


#        model.summary()

        # There are two inputs to be used in the predictor
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
        select = Input(shape=(self.input_shape,), dtype='float32')

        # Element-wise multiplication
        input = Multiply()([feature, select])
        pred = model(input)

        return Model([feature, select], pred)
        # return Model([feature, select_with, select_without], tf.concat(axis=1,values=[pred_with, pred_without]))
        
    def build_selector(self):

        model = Sequential()
#         model.add(Dense(self.input_shape, activation='sigmoid', kernel_initializer='he_normal',name='dense5',\
#                         kernel_regularizer=regularizers.l2(1e-4), input_dim = self.input_shape))

        model.add(Dense(self.latent_dim_pred, activation= 'relu', kernel_initializer='he_normal', name='dense1', kernel_regularizer=regularizers.l2(1e-4),
                        input_dim=self.input_shape))
        model.add(Dense(self.latent_dim_pred, activation= 'relu', kernel_initializer='he_normal', name='dense2', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(self.input_shape, activation='sigmoid', kernel_initializer='he_normal',name='dense5', kernel_regularizer=regularizers.l2(1e-4)))

        return model
#        model.summary()

#         input = Input(shape=(self.input_shape,), dtype='float32')

#         # Element-wise multiplication
        
#         prob = model(input)

#         return Model(input, prob)
        # return Model([feature, select_with, select_without], tf.concat(axis=1,values=[pred_with, pred_without]))


    # %% Sampling the features based on the output of the selector
    def Selector_M(self, gen_prob):

        samples = np.random.binomial(1, gen_prob, (self.batch_size, self.input_shape))

        return samples
    
    def update_selector(self, x_input, g_out, loss, sel_with):
        weights = [self.selector.get_layer(layer.name).get_weights() for layer in self.selector.layers]

        features = [x_input]
        feature = x_input

        dev = g_out * (1 - g_out) # N x C
        dev_pi = ((sel_with - g_out)/(dev+ 1e-8))
        
        updated_weights = weights.copy()

        for weight, bais in (weights):
            feature = np.matmul(feature, weight) + bais
            features.append(feature)

            
        dev = dev_pi * dev

        for i in range(len(weights)-1, -1, -1):
            #print(i)
            bais = weights[i][1]
            weight = weights[i][0]

            updated_weights[i][0] = weight - self.lr_t *  np.matmul(features[i].T, dev) * loss
            updated_weights[i][1] = bais - self.lr_t * np.mean(dev, 0) * loss

            dev = np.matmul(dev, weight.T)

        idx=0
        for layer in self.selector.layers:
            self.selector.get_layer(layer.name).set_weights(updated_weights[idx])
            idx += 1       


    # %% Training procedure
    def train(self, x_train, y_train, x_test, y_test):
        best_eq = 100
        # For each epoch (actually iterations)
        for epoch in range(self.epochs):

            # %% Train Predictor
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            
            x_batch = x_train[idx, :]
            y_batch = y_train[idx, :]

            # Generate a batch of probabilities of feature selection
            # gen_prob = np.exp(self.theta)/(np.sum(np.exp(self.theta)))
            
            gen_prob = np.exp(self.theta) / (np.exp(self.theta)+1)

            #print(gen_prob)

            # Sampling the features based on the generated probability
            sel_prob = self.Selector_M(gen_prob)
            sel_with = np.copy(sel_prob)
            sel_without = np.copy(sel_prob)
            sel_with[:,self.sens_idx] = 1
            sel_without[:,self.sens_idx] = 0

            # Compute the prediction of the fairness based on the sampled features (used for selector training)
            y_pred_with = self.predictor.predict([x_batch, sel_with])
            y_pred_without = self.predictor.predict([x_batch, sel_without])

            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            # loss_pred_sel = np.array([0,0])
#             loss_pred_all = self.predictor.test_on_batch([x_batch, np.ones(x_batch.shape)], y_batch)
            loss_pred_y = self.predictor.train_on_batch([x_batch, sel_without], y_batch)
            loss_pred_sel = self.predictor.train_on_batch([x_batch, sel_without], y_pred_with)

            # %% Train the selector
            
            Reward =  np.mean(- np.sum(y_pred_with * np.log(y_pred_without + 1e-8), axis=1)) \
            
            dev = np.mean(sel_with, axis=0) - gen_prob
            # dev =  (tmp_prob/(gen_prob+1e-8) + (1-tmp_prob)/(1-gen_prob+1e-8))
            self.theta = self.theta + self.lr_t * Reward * dev
            
            if epoch % 100 == 0:

                Sel_Prob_Test = self.output()

                score = 1. * (Sel_Prob_Test > .5)
                #print(score)
                score = np.array([score, ] * x_test.shape[0])
                pred = self.get_prediction(x_test, score)
                pred = pred.argmax(axis = 1)

                idx = np.random.randint(0, x_test.shape[0], self.batch_size)
                x_batch = x_test[idx, :] 

                y_pred_without = self.predictor.predict([x_batch, sel_without])
                loss = self.predictor.test_on_batch([x_batch, sel_without], y_test[idx])

                pred_male_label = pred[x_test[:,self.sens_idx]==2]
                pred_female_label = pred[x_test[:,self.sens_idx]==1]

                male_label = y_test[x_test[:,self.sens_idx]==2]
                female_label = y_test[x_test[:,self.sens_idx]==1]
                male_label = male_label.argmax(axis = 1)
                female_label = female_label.argmax(axis = 1)
                
                tpr_male = np.mean(pred_male_label[male_label == 1] == 1)
                tpr_female = np.mean(pred_female_label[female_label == 1] == 1)
                fpr_male = np.mean(pred_male_label[male_label != 1] == 1)
                fpr_female = np.mean(pred_female_label[female_label != 1] == 1)
                
#                 print(x_test[:,self.sens_idx])
#                 print(pred_female_label)

                recall_m = sklm.recall_score(male_label, pred_male_label, zero_division = 0)
                recall_f = sklm.recall_score(female_label, pred_female_label, zero_division = 0)
#                 eq = abs(np.round(recall_f - recall_m, 4))
                eq = abs(tpr_male - tpr_female) + abs(fpr_male - fpr_female)

                dialog = 'Epoch: ' + str(epoch) \
                     + ', pred_acc: ' + str(loss_pred_y[1]) + ', diff_acc: ' + str(loss_pred_sel[1]) \
                     + ', pred_loss: ' + str(np.round(loss_pred_y[0], 4)) \
                     + ', diff_loss: ' + str(np.round(loss_pred_sel[0], 4)) \
                     + ', acc: ' + str(np.round(loss[1], 4)) \
                     + ', TPR female: ' + str(np.round(recall_f, 4)) \
                     + ', TPR male: ' + str(np.round(recall_m, 4)) \
                     + ', Eq_opp: ' + str(eq)

                print(dialog)
                if eq < best_eq and epoch != 0 and eq < 0.17: 
                    print('\n\n best model saved \n\n ')
                    print(dialog)
                    #model_name = 'save_fair/fair_eq_{}_acc_{}.hdf5'.format(np.round(eq, 4), np.round(loss_pred_y[1], 4))
                    self.best_predictor = keras.models.clone_model(self.predictor)
                    self.best_predictor.set_weights(self.predictor.get_weights())

                    self.best_theta = self.theta
                    best_eq = eq
                    self.best_saved = 1

    # %% Selected Features
    def output(self):
        gen_prob = np.exp(self.theta) / (np.exp(self.theta)+1)
        return np.asarray(gen_prob)


    # %% Prediction Results
    def get_prediction(self, x_test, m_test):
        pred = self.predictor.predict([x_test, m_test])
        return np.asarray(pred)
    
    def test_output(self):
        if self.best_saved != 0:
            gen_prob = np.exp(self.best_theta) / (np.exp(self.best_theta)+1)
        else:
            gen_prob = self.output()
   
        return np.asarray(gen_prob)

    def test_prediction(self, x_test, m_test):
        if self.best_saved != 0:
            pred = self.best_predictor.predict([x_test, m_test])
        else:
            pred = self.predictor.predict([x_test, m_test])
   
        return pred

def fair_shapley(x_train, y_train, x_val, y_val, x_test, sens_idx, session=None, batch_size=200, epochs=5000, lr = 1e-4, lr_t = 1e-3):

    # Use CUDA
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # tf.keras.backend.set_session(session)

    # Hyperparameter
    lamda = 3

    # 1. FAIR Class call
    FAIR_Alg = FAIR(x_train, y_train, sens_idx, session, batch_size, epochs, lr, lr_t)

    # 2. Algorithm training
    best_eq = FAIR_Alg.train(x_train, y_train, x_val, y_val)
    #FAIR_Alg.predictor = keras.models.load_model('best_model.hdf5')
    
    Sel_Prob_Test = FAIR_Alg.test_output()
    Sel_Prob_Test[sens_idx] = 0

    # 2. Selected features
    score = 1. * (Sel_Prob_Test > .5)
    #print(score)
    score = np.array([score, ] * x_test.shape[0])

    pred = FAIR_Alg.test_prediction(x_test,score)
    return pred


