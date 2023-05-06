
############################## imports ##############################
import os
import sys
import timeit
from collections import OrderedDict

import numpy as np
from scipy import linalg
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
# import h5py
import scipy.ndimage as ndi
if sys.version_info[0]==3:
    import _pickle as cPickle
else:
    import cPickle

# import theano as th
# import theano.tensor as T
# import lasagne
import time
import socket
import inspect
from scipy.io import loadmat


def cc_coef_mean(y_true, y_pred):
    mu_y_true = np.mean(y_true, axis=-1, keepdims=True)
    mu_y_pred = np.mean(y_pred, axis=-1, keepdims=True)
    return 2 * np.mean(np.multiply((y_true - mu_y_true), (y_pred - mu_y_pred)), axis=-1) / \
           (np.var(y_true, axis=-1) + np.var(y_pred, axis=-1) + np.mean(np.square(mu_y_pred - mu_y_true), axis=-1))


def evaluation_metric(pred=None, y=None):
    errors = np.absolute(pred - y).mean(axis=0)
    error = errors.mean()
    error_std = np.std(errors)

    ccs = np.zeros((errors.shape[0]))
    for i in range(errors.shape[0]):
        ccs[i] = cc_coef_mean(y[:, i], pred[:, i])
    cc= ccs.mean()
    cc_std = np.std(ccs)

    return error, error_std, cc, cc_std

def rampup(current_epoch=None, loss_old=None, current_loss=None, old_output=None,
             max_output=None, min_output=None,  start_slope=None, stop_slope=None, type='linear'):

    if current_epoch < start_slope:
        output = min_output
    elif current_epoch > stop_slope:
        output = max_output
    else:
        if type == 'linear':
            output = ((min_output-max_output) * current_epoch + min_output*stop_slope + max_output*start_slope) / (stop_slope - start_slope)

    return np.asarray(output, dtype=np.float32)



def rampdown(current_epoch=None, loss_old=None, current_loss=None, old_output=None, num_step=None,
             max_output=None, min_output=None,  start_slope=None, stop_slope=None, type='linear'):

    if current_epoch < start_slope:
        output = max_output
    elif current_epoch > stop_slope:
        output = min_output
    else:
        if type == 'linear':
            output = ((min_output-max_output) * current_epoch + max_output*stop_slope - min_output*start_slope) / (stop_slope - start_slope)

        elif type == 'step':
            temp = (max_output-min_output)/num_step
            if old_output-temp > min_output:
                output = old_output - temp
            else:
                output = min_output

        elif type == 'adaptive':
            if (loss_old < current_loss) and (old_output > min_output):
                output = old_output * 0.9

    return np.asarray(output, dtype=np.float32)


def print_results(ep=None, t=None, l_tot=None, l_sup=None, l_con=None, l_reg=None, l_rec=None, l_disc=None, l_gen=None,
                  l_inv=None, l_cly=None, gan_acc=None, tr_err=None, va_err=None, te_err=None, te_std=None, tr_cc=None,
                  va_cc=None, te_cc=None, te_cc_std=None, best_val_flag=None, va_err2=None, te_err2=None, te_std2=None, best_val_flag2=None, tr_cc2=None, va_cc2=None,
                  te_cc2=None, te_cc_std2=None, tr_err_u=None):
    color_flag_2 = False


    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    if values['best_val_flag'] is not None:
        color_flag = True
        values['best_val_flag'] = None
    else:
        color_flag = False

    if values['best_val_flag2'] is not None:
        color_flag_2 = True
        values['best_val_flag2'] = None
    else:
        color_flag_2 = False

    prt_str = ""
    for i in args:
        if values[i] is not None:
            if i == 't':
                values[i] = int(t)
            elif i[:2] == 'l_':
                values[i] = round(values[i], 3)
            elif i[3:] == '_acc':
                values[i] = round(values[i], 2)
            elif i[2:6] == '_err':
                values[i] = round(values[i], 4)
            elif i[2:5] == '_cc':
                values[i] = round(values[i], 4)
            elif i[2:6] == '_std':
                values[i] = round(values[i], 4)


            if (color_flag) and (i == 'va_err') or (color_flag_2) and (i == 'va_err2'):
                prt_str += '\x1b[6;30;42m'

            prt_str += str(i) + ' = ' + str(values[i]) + ', '

            if (color_flag) and (i == 'te_err') or (color_flag_2) and (i == 'te_err2'):
                prt_str += '\x1b[0m'

    print(prt_str)









def logging(file_name=None, verbose=0):
    output_path = './results/' + file_name.split('.')[0] + '/' + time.strftime("%d-%m-%Y_") +\
                  time.strftime("%H:%M:%S_")  + socket.gethostname()
    create_result_dirs(output_path, file_name, verbose)
    sys.stdout = Logger(output_path)
    if verbose > 1: print(sys.argv)

    return output_path



def normalization_1(x):
    MAX = np.max(x, axis=0)
    MIN = np.max(x, axis=0)
    return (2*x - MIN - MAX) / (MAX-MIN)

def normalization_01(x):
    MAX = np.max(x, axis=0)
    MIN = np.max(x, axis=0)
    return (x - MIN) / (MAX - MIN)


def normalization_gauss(x):
    MEAN = np.mean(x, axis=0)
    STD = np.std(x, axis=0) + np.finfo(float).eps
    return (x - MEAN) / STD


def load_data(data_dir=None, verbose=0, normaliztion='[-1,1]', ratio=0.1, random_state=0):

    if verbose > 0: print('\n*** Loading data ***')

    in_data = loadmat(data_dir+'/MCI_test.mat')
    x = in_data['testX']
    y = in_data['testY']
    n = len(y)
    inds = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(inds)
    nTe = round(n*ratio)
    x_te = x[inds[:nTe]]
    y_te = y[inds[:nTe]]
    x_va = x[inds[nTe:]]
    y_va = y[inds[nTe:]]

    in_data = loadmat(data_dir+'/MRI_reg.mat')
    x1_tr_reg = in_data['X1']
    x2_tr_reg = in_data['X2']

    in_data = loadmat(data_dir+'/MCI_cly.mat')
    x_tr_cly = in_data['X']
    y_tr_cly = in_data['Y']

    in_data = loadmat(data_dir+'/MCI_adv.mat')
    x1_tr_adv = in_data['X1']
    x2_tr_adv = in_data['X2']
    y_tr_adv = in_data['Y']

    if normaliztion == '[-1,1]':
        x1_tr_adv = normalization_1(x1_tr_adv)
        x2_tr_adv = normalization_1(x2_tr_adv)
        x_tr_cly = normalization_1(x_tr_cly)
        x1_tr_reg = normalization_1(x1_tr_reg)
        x2_tr_reg = normalization_1(x2_tr_reg)
        x_va = normalization_1(x_va)
        x_te = normalization_1(x_te)
    elif normaliztion == '[0,1]':
        x1_tr_adv = normalization_01(x1_tr_adv)
        x2_tr_adv = normalization_01(x2_tr_adv)
        x_tr_cly = normalization_01(x_tr_cly)
        x1_tr_reg = normalization_01(x1_tr_reg)
        x2_tr_reg = normalization_01(x2_tr_reg)
        x_va = normalization_01(x_va)
        x_te = normalization_01(x_te)
    elif normaliztion == 'gaussian':
        x1_tr_adv = normalization_gauss(x1_tr_adv)
        x2_tr_adv = normalization_gauss(x2_tr_adv)
        x_tr_cly = normalization_gauss(x_tr_cly)
        x1_tr_reg = normalization_gauss(x1_tr_reg)
        x2_tr_reg = normalization_gauss(x2_tr_reg)
        x_va = normalization_gauss(x_va)
        x_te = normalization_gauss(x_te)

    return x1_tr_adv, x2_tr_adv, y_tr_adv, x_tr_cly, y_tr_cly, x1_tr_reg, x2_tr_reg, x_va, y_va, x_te, y_te


def create_result_dirs(output_path, file_name, verbose=0):
    if not os.path.exists(output_path):
        if verbose > 0:  print('\n*** Creating logging folder ***')
        os.makedirs(output_path)
        func_file_name = os.path.basename(__file__)
        if func_file_name.split('.')[1] == 'pyc':
            func_file_name = func_file_name[:-1]
        functions_full_path = os.path.join(output_path, func_file_name)
        cmd = 'cp ' + func_file_name + ' "' + functions_full_path + '"'
        os.popen(cmd)
        run_file_full_path = os.path.join(output_path, file_name)
        cmd = 'cp ' + file_name + ' "' + run_file_full_path + '"'
        os.popen(cmd)








# ########################################## nn functions ##########################################
#
# def robust_adam(loss, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
#     # Convert NaNs to zeros.
#     def clear_nan(x):
#         return T.switch(T.isnan(x), np.float32(0.0), x)
#
#     new = OrderedDict()
#     pg = zip(params, lasagne.updates.get_or_compute_grads(loss, params))
#     t = th.shared(lasagne.utils.floatX(0.))
#
#     new[t] = t + 1.0
#     coef = learning_rate * T.sqrt(1.0 - beta2**new[t]) / (1.0 - beta1**new[t])
#     for p, g in pg:
#         value = p.get_value(borrow=True)
#         m = th.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
#         v = th.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
#         new[m] = clear_nan(beta1 * m + (1.0 - beta1) * g)
#         new[v] = clear_nan(beta2 * v + (1.0 - beta2) * g**2)
#         new[p] = clear_nan(p - coef * new[m] / (T.sqrt(new[v]) + epsilon))
#
#     return new
#
# # T.nnet.relu has some issues with very large inputs, this is more stable
# def relu(x):
#     return T.maximum(x, 0)
#
#
# def lrelu(x, a=0.1):
#     return T.maximum(x, a * x)
#
# class my_lrelu(lasagne.layers.Layer):
#     def get_output_for(self, input, **kwargs):
#         return lrelu(input, **kwargs)
#
# def log_sum_exp(x, axis=1):
#     m = T.max(x, axis=axis)
#     return m + T.log(T.sum(T.exp(x - m.dimshuffle(0, 'x')), axis=axis))
#
#
# def adamax_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
#     updates = []
#     grads = T.grad(cost, params)
#     for p, g in zip(params, grads):
#         mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
#         v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
#         if mom1 > 0:
#             v_t = mom1 * v + (1. - mom1) * g
#             updates.append((v, v_t))
#         else:
#             v_t = g
#         mg_t = T.maximum(mom2 * mg, abs(g))
#         g_t = v_t / (mg_t + 1e-6)
#         p_t = p - lr * g_t
#         updates.append((mg, mg_t))
#         updates.append((p, p_t))
#     return updates
#
#
# def adam_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
#     updates = []
#     grads = T.grad(cost, params)
#     t = th.shared(np.cast[th.config.floatX](1.))
#     for p, g in zip(params, grads):
#         v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
#         mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
#         v_t = mom1 * v + (1. - mom1) * g
#         mg_t = mom2 * mg + (1. - mom2) * T.square(g)
#         v_hat = v_t / (1. - mom1 ** t)
#         mg_hat = mg_t / (1. - mom2 ** t)
#         g_t = v_hat / T.sqrt(mg_hat + 1e-8)
#         p_t = p - lr * g_t
#         updates.append((v, v_t))
#         updates.append((mg, mg_t))
#         updates.append((p, p_t))
#     updates.append((t, t + 1))
#     return updates
#
#
# def softmax_loss(p_true, output_before_softmax):
#     output_before_softmax -= T.max(output_before_softmax, axis=1, keepdims=True)
#     if p_true.ndim == 2:
#         return T.mean(
#             T.log(T.sum(T.exp(output_before_softmax), axis=1)) - T.sum(p_true * output_before_softmax, axis=1))
#     else:
#         return T.mean(T.log(T.sum(T.exp(output_before_softmax), axis=1)) - output_before_softmax[
#             T.arange(p_true.shape[0]), p_true])
#
#
# class BatchNormLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
#                  W=lasagne.init.Normal(0.05), nonlinearity=relu, **kwargs):
#         super(BatchNormLayer, self).__init__(incoming, **kwargs)
#         self.nonlinearity = nonlinearity
#         k = self.input_shape[1]
#         if b is not None:
#             self.b = self.add_param(b, (k,), name="b", regularizable=False)
#         if g is not None:
#             self.g = self.add_param(g, (k,), name="g")
#         self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean",
#                                              regularizable=False, trainable=False)
#         self.avg_batch_var = self.add_param(lasagne.init.Constant(1.), (k,), name="avg_batch_var", regularizable=False,
#                                             trainable=False)
#         incoming.W.set_value(W.sample(incoming.W.get_value().shape))
#         if len(self.input_shape) == 4:
#             self.axes_to_sum = (0, 2, 3)
#             self.dimshuffle_args = ['x', 0, 'x', 'x']
#         else:
#             self.axes_to_sum = 0
#             self.dimshuffle_args = ['x', 0]
#
#     def get_output_for(self, input, deterministic=False, **kwargs):
#         if deterministic:
#             norm_features = (input - self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)) / T.sqrt(
#                 1e-6 + self.avg_batch_var).dimshuffle(*self.dimshuffle_args)
#         else:
#             batch_mean = T.mean(input, axis=self.axes_to_sum).flatten()
#             centered_input = input - batch_mean.dimshuffle(*self.dimshuffle_args)
#             batch_var = T.mean(T.square(centered_input), axis=self.axes_to_sum).flatten()
#             batch_stdv = T.sqrt(1e-6 + batch_var)
#             norm_features = centered_input / batch_stdv.dimshuffle(*self.dimshuffle_args)
#
#             # BN updates
#             new_m = 0.9 * self.avg_batch_mean + 0.1 * batch_mean
#             new_v = 0.9 * self.avg_batch_var + T.cast((0.1 * input.shape[0]) / (input.shape[0] - 1.),
#                                                       th.config.floatX) * batch_var
#             self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]
#
#         if hasattr(self, 'g'):
#             activation = norm_features * self.g.dimshuffle(*self.dimshuffle_args)
#         else:
#             activation = norm_features
#         if hasattr(self, 'b'):
#             activation += self.b.dimshuffle(*self.dimshuffle_args)
#
#         return self.nonlinearity(activation)
#
#
# def batch_norm(layer, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), **kwargs):
#     """
#     adapted from https://gist.github.com/f0k/f1a6bd3c8585c400c190
#     """
#     nonlinearity = getattr(layer, 'nonlinearity', None)
#     if nonlinearity is not None:
#         layer.nonlinearity = lasagne.nonlinearities.identity
#     if hasattr(layer, 'b'):
#         del layer.params[layer.b]
#         layer.b = None
#     return BatchNormLayer(layer, b, g, nonlinearity=nonlinearity, **kwargs)
#
#
# class GlobalAvgLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, **kwargs):
#         super(GlobalAvgLayer, self).__init__(incoming, **kwargs)
#
#     def get_output_for(self, input, **kwargs):
#         return T.mean(input, axis=(2, 3))
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape[:2]
#
#
# class MeanOnlyBNLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
#                  W=lasagne.init.Normal(0.05), nonlinearity=relu, **kwargs):
#         super(MeanOnlyBNLayer, self).__init__(incoming, **kwargs)
#         self.nonlinearity = nonlinearity
#         k = self.input_shape[1]
#         if b is not None:
#             self.b = self.add_param(b, (k,), name="b", regularizable=False)
#         if g is not None:
#             self.g = self.add_param(g, (k,), name="g")
#         self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean",
#                                              regularizable=False, trainable=False)
#         if len(self.input_shape) == 4:
#             self.axes_to_sum = (0, 2, 3)
#             self.dimshuffle_args = ['x', 0, 'x', 'x']
#         else:
#             self.axes_to_sum = 0
#             self.dimshuffle_args = ['x', 0]
#
#         # scale weights in layer below
#         incoming.W_param = incoming.W
#         incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
#         if incoming.W_param.ndim == 4:
#             if isinstance(incoming, Deconv2DLayer):
#                 W_axes_to_sum = (0, 2, 3)
#                 W_dimshuffle_args = ['x', 0, 'x', 'x']
#             else:
#                 W_axes_to_sum = (1, 2, 3)
#                 W_dimshuffle_args = [0, 'x', 'x', 'x']
#         else:
#             W_axes_to_sum = 0
#             W_dimshuffle_args = ['x', 0]
#         if g is not None:
#             incoming.W = incoming.W_param * (
#             self.g / T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
#         else:
#             incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum, keepdims=True))
#
#     def get_output_for(self, input, deterministic=False, init=False, **kwargs):
#         if deterministic:
#             activation = input - self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)
#         else:
#             m = T.mean(input, axis=self.axes_to_sum)
#             activation = input - m.dimshuffle(*self.dimshuffle_args)
#             self.bn_updates = [(self.avg_batch_mean, 0.9 * self.avg_batch_mean + 0.1 * m)]
#             if init:
#                 stdv = T.sqrt(T.mean(T.square(activation), axis=self.axes_to_sum))
#                 activation /= stdv.dimshuffle(*self.dimshuffle_args)
#                 self.init_updates = [(self.g, self.g / stdv)]
#         if hasattr(self, 'b'):
#             activation += self.b.dimshuffle(*self.dimshuffle_args)
#
#         return self.nonlinearity(activation)
#
#
# def mean_only_bn(layer, **kwargs):
#     nonlinearity = getattr(layer, 'nonlinearity', None)
#     if nonlinearity is not None:
#         layer.nonlinearity = lasagne.nonlinearities.identity
#     if hasattr(layer, 'b'):
#         del layer.params[layer.b]
#         layer.b = None
#     return MeanOnlyBNLayer(layer, nonlinearity=nonlinearity, **kwargs)
#
#
# class WeightNormLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
#                  W=lasagne.init.Normal(0.05), nonlinearity=relu, **kwargs):
#         super(WeightNormLayer, self).__init__(incoming, **kwargs)
#         self.nonlinearity = nonlinearity
#         k = self.input_shape[1]
#         if b is not None:
#             self.b = self.add_param(b, (k,), name="b", regularizable=False)
#         if g is not None:
#             self.g = self.add_param(g, (k,), name="g")
#         if len(self.input_shape) == 4:
#             self.axes_to_sum = (0, 2, 3)
#             self.dimshuffle_args = ['x', 0, 'x', 'x']
#         else:
#             self.axes_to_sum = 0
#             self.dimshuffle_args = ['x', 0]
#
#         # scale weights in layer below
#         incoming.W_param = incoming.W
#         incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
#         if incoming.W_param.ndim == 4:
#             if isinstance(incoming, Deconv2DLayer):
#                 W_axes_to_sum = (0, 2, 3)
#                 W_dimshuffle_args = ['x', 0, 'x', 'x']
#             else:
#                 W_axes_to_sum = (1, 2, 3)
#                 W_dimshuffle_args = [0, 'x', 'x', 'x']
#         else:
#             W_axes_to_sum = 0
#             W_dimshuffle_args = ['x', 0]
#         if g is not None:
#             incoming.W = incoming.W_param * (
#             self.g / T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
#         else:
#             incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum, keepdims=True))
#
#     def get_output_for(self, input, init=False, **kwargs):
#         if init:
#             m = T.mean(input, self.axes_to_sum)
#             input -= m.dimshuffle(*self.dimshuffle_args)
#             stdv = T.sqrt(T.mean(T.square(input), axis=self.axes_to_sum))
#             input /= stdv.dimshuffle(*self.dimshuffle_args)
#             self.init_updates = [(self.b, -m / stdv), (self.g, self.g / stdv)]
#         elif hasattr(self, 'b'):
#             input += self.b.dimshuffle(*self.dimshuffle_args)
#
#         return self.nonlinearity(input)
#
#
# def weight_norm(layer, **kwargs):
#     nonlinearity = getattr(layer, 'nonlinearity', None)
#     if nonlinearity is not None:
#         layer.nonlinearity = lasagne.nonlinearities.identity
#     if hasattr(layer, 'b'):
#         del layer.params[layer.b]
#         layer.b = None
#     return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)
#
#
# class InitLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), nonlinearity=relu, **kwargs):
#         super(InitLayer, self).__init__(incoming, **kwargs)
#         self.nonlinearity = nonlinearity
#         k = self.input_shape[1]
#         if b is not None:
#             self.b = self.add_param(b, (k,), name="b", regularizable=False)
#         if g is not None:
#             self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=False)
#         if len(self.input_shape) == 4:
#             self.axes_to_sum = (0, 2, 3)
#             self.dimshuffle_args = ['x', 0, 'x', 'x']
#         else:
#             self.axes_to_sum = 0
#             self.dimshuffle_args = ['x', 0]
#
#         # scale weights in layer below
#         incoming.W_param = incoming.W
#         if incoming.W_param.ndim == 4:
#             if isinstance(incoming, Deconv2DLayer):
#                 W_dimshuffle_args = ['x', 0, 'x', 'x']
#             else:
#                 W_dimshuffle_args = [0, 'x', 'x', 'x']
#         else:
#             W_dimshuffle_args = ['x', 0]
#         incoming.W = self.g.dimshuffle(*W_dimshuffle_args) * incoming.W_param
#
#     def get_output_for(self, input, init=False, **kwargs):
#         if init:
#             m = T.mean(input, self.axes_to_sum)
#             input -= m.dimshuffle(*self.dimshuffle_args)
#             stdv = T.sqrt(T.mean(T.square(input), axis=self.axes_to_sum))
#             input /= stdv.dimshuffle(*self.dimshuffle_args)
#             self.init_updates = [(self.b, -m / stdv), (self.g, self.g / stdv)]
#         elif hasattr(self, 'b'):
#             input += self.b.dimshuffle(*self.dimshuffle_args)
#
#         return self.nonlinearity(input)
#
#
# def no_norm(layer, **kwargs):
#     nonlinearity = getattr(layer, 'nonlinearity', None)
#     if nonlinearity is not None:
#         layer.nonlinearity = lasagne.nonlinearities.identity
#     if hasattr(layer, 'b'):
#         del layer.params[layer.b]
#         layer.b = None
#     return InitLayer(layer, nonlinearity=nonlinearity, **kwargs)
#
#
# class Deconv2DLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, target_shape, filter_size, stride=(2, 2),
#                  W=lasagne.init.Normal(0.05), b=lasagne.init.Constant(0.), nonlinearity=relu, **kwargs):
#         super(Deconv2DLayer, self).__init__(incoming, **kwargs)
#         self.target_shape = target_shape
#         self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
#         self.filter_size = lasagne.layers.dnn.as_tuple(filter_size, 2)
#         self.stride = lasagne.layers.dnn.as_tuple(stride, 2)
#         self.target_shape = target_shape
#
#         self.W_shape = (incoming.output_shape[1], target_shape[1], filter_size[0], filter_size[1])
#         self.W = self.add_param(W, self.W_shape, name="W")
#         if b is not None:
#             self.b = self.add_param(b, (target_shape[1],), name="b")
#         else:
#             self.b = None
#
#     def get_output_for(self, input, **kwargs):
#         op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=self.target_shape, kshp=self.W_shape, subsample=self.stride, border_mode='half')
#         activation = op(self.W, input, self.target_shape[2:])
#
#         if self.b is not None:
#             activation += self.b.dimshuffle('x', 0, 'x', 'x')
#
#         return self.nonlinearity(activation)
#
#     def get_output_shape_for(self, input_shape):
#         return self.target_shape
#
# def analyze_function(func, verbose = False):
#     assert isinstance(func, th.compile.Function)
#     topo = func.maker.fgraph.toposort()
#
#     # Print stats.
#
#     if verbose:
#         op_names = [type(apply.op).__name__ for apply in topo]
#         op_dict = {op: 0 for op in op_names}
#         for op in op_names:
#             op_dict[op] += 1
#
#         op_list = op_dict.items()
#         op_list.sort(key = lambda x: -x[1])
#
#         print
#         for op, num in op_list:
#             print("  %-8d%s" % (num, op))
#         print
#
#     # Check for float64 use.
#
#     for apply in topo:
#         dtype = getattr(apply.outputs[0].type, 'dtype', '')
#         acc_dtype = getattr(apply.op, 'acc_dtype', '')
#         if dtype == 'float64' or acc_dtype == 'float64':
#             print('WARNING: th float64:', apply)
#             if verbose:
#                 print
#                 th.printing.debugprint(apply)
#                 print
#
#     # Check for excess GPU=>CPU transfers.
#
#     for apply in topo:
#         op = type(apply.op).__name__
#         if op == 'HostFromGpu':
#             for parent in topo:
#                 parent_inputs = [var.owner for var in parent.inputs]
#                 if apply in parent_inputs:
#                     print('WARNING: th CPU fallback:', parent)
#                     if verbose:
#                         print
#                         th.printing.debugprint(parent)
#                         print
#
# # Compile and check Theano function.
# def th_function(*args, **kwargs):
#     func = th.function(*args, **kwargs)
#     analyze_function(func, verbose = False)
#     return func




########################################## data functions ##########################################

# class ZCA(object):
#     def __init__(self, regularization=1e-5, x=None):
#         self.regularization = regularization
#         if x is not None:
#             self.fit(x)
#
#     def fit(self, x):
#         s = x.shape
#         x = x.copy().reshape((s[0], np.prod(s[1:])))
#         m = np.mean(x, axis=0)
#         x -= m
#         sigma = np.dot(x.T, x) / x.shape[0]
#         U, S, V = linalg.svd(sigma)
#         tmp = np.dot(U, np.diag(1. / np.sqrt(S + self.regularization)))
#         tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
#         self.ZCA_mat = th.shared(np.dot(tmp, U.T).astype(th.config.floatX))
#         self.inv_ZCA_mat = th.shared(np.dot(tmp2, U.T).astype(th.config.floatX))
#         self.mean = th.shared(m.astype(th.config.floatX))
#
#     def apply(self, x):
#         s = x.shape
#         if isinstance(x, np.ndarray):
#             return np.dot(x.reshape((s[0], np.prod(s[1:]))) - self.mean.get_value(), self.ZCA_mat.get_value()).reshape(
#                 s)
#         elif isinstance(x, T.TensorVariable):
#             return T.dot(x.flatten(2) - self.mean.dimshuffle('x', 0), self.ZCA_mat).reshape(s)
#         else:
#             raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")
#
#     def invert(self, x):
#         s = x.shape
#         if isinstance(x, np.ndarray):
#             return (
#             np.dot(x.reshape((s[0], np.prod(s[1:]))), self.inv_ZCA_mat.get_value()) + self.mean.get_value()).reshape(s)
#         elif isinstance(x, T.TensorVariable):
#             return (T.dot(x.flatten(2), self.inv_ZCA_mat) + self.mean.dimshuffle('x', 0)).reshape(s)
#         else:
#             raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")

def best_map_acc(L1, L2):
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2)

def kmeans(x, y=None, nClusters=None, weight_initilization='k-means++', seed=42, n_init=40, max_iter=300, verbose=0):
    # weight_initilization = { 'kmeans-pca', 'kmean++', 'random', None }

    if weight_initilization == 'kmeans-pca':
        start_time = timeit.default_timer()
        pca = PCA(n_components=nClusters).fit(x)
        kmeans_model = KMeans(init=pca.components_, n_clusters=nClusters, n_init=1, max_iter=300, random_state=seed)
        y_pred = kmeans_model.fit_predict(x)

        centroids = kmeans_model.cluster_centers_.T
        centroids = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))
        end_time = timeit.default_timer()

    elif weight_initilization == 'k-means++':
        start_time = timeit.default_timer()
        kmeans_model = KMeans(init='k-means++', n_clusters=nClusters, n_init=n_init, max_iter=max_iter, n_jobs=n_init, random_state=seed)
        y_pred = kmeans_model.fit_predict(x)

        # D = 1.0 / euclidean_distances(x, kmeans_model.cluster_centers_, squared=True)
        # D **= 2.0 / (2 - 1)
        # D /= np.sum(D, axis=1)[:, np.newaxis]

        centroids = kmeans_model.cluster_centers_.T
        centroids = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))
        end_time = timeit.default_timer()

    if verbose > 0:
        print('k-means: \t nmi =', normalized_mutual_info_score(y, y_pred), '\t acc = {:.4f} '.format(best_map_acc(y, y_pred)),
              'K-means objective = {:.1f} '.format(kmeans_model.inertia_), '\t runtime =', end_time - start_time)


    return y_pred, centroids, kmeans_model.inertia_


def balanced_sample_extraction(x=None, y=None, num_query=None, shuffle_flag=True):
    if shuffle_flag:
        inds = np.random.permutation(y.shape[0])
        x = x[inds]
        y = y[inds]

    labels = np.unique(y)
    num_class = labels.shape[0]
    for i in range(num_class):
        if i == 0:
            sampled_x = x[y == labels[i]][:int(num_query / num_class)]
            remianed_x = x[y == labels[i]][int(num_query / num_class):]
            sampled_y = y[y == labels[i]][:int(num_query / num_class)]
            remianed_y = y[y == labels[i]][int(num_query / num_class):]
        else:
            sampled_x = np.append(sampled_x, x[y == labels[i]][:int(num_query / num_class)], axis=0)
            remianed_x = np.append(remianed_x, x[y == labels[i]][int(num_query / num_class):], axis=0)
            sampled_y = np.append(sampled_y, y[y == labels[i]][:int(num_query / num_class)], axis=0)
            remianed_y =  np.append(remianed_y, y[y == labels[i]][int(num_query / num_class):], axis=0)

    if shuffle_flag:
        inds = np.random.permutation(num_query)
        sampled_x = sampled_x[inds]
        sampled_y = sampled_y[inds]
        inds = np.random.permutation(remianed_y.shape[0])
        remianed_x = remianed_x[inds]
        remianed_y = remianed_y[inds]

    return remianed_x, remianed_y, sampled_x, sampled_y


# def load_preprocess_data_clustering(dataset_name=None, preprocess_inputs=None, data_dir=None):
#
#     if dataset_name == 'MNIST-full':
#         data = np.load(os.path.join(data_dir, 'mnist.npz'))
#         trainx = data['x_train'].astype(th.config.floatX)
#         testx = data['x_test'].astype(th.config.floatX)
#         trainy = data['y_train'].astype(np.int32)
#         testy = data['y_test'].astype(np.int32)
#         #
#         x = np.concatenate((trainx, testx), axis=0)
#         x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#         y = np.concatenate((trainy, testy), axis=0)
#
#     if dataset_name == 'MNIST-test':
#         hf = h5py.File(data_dir + '/data.h5', 'r')
#         x = np.asarray(hf.get('data'), dtype='float32')
#         y = np.asarray(hf.get('labels'), dtype='int32')
#
#
#     if preprocess_inputs == 'whiten':
#         x = whiten_norm2d(x)
#     elif preprocess_inputs == 'zca':
#         whitener = ZCA(x=x)
#         x = whitener.apply(x)
#     elif preprocess_inputs == '0_1':
#         x = x / 255.0
#
#
#     x = np.asarray(x, dtype='float32')
#     y = np.asarray(y, dtype='int32')
#     return x, y

# def load_preprocess_data_hashing(dataset_name=None, num_query=1000, preprocess_inputs=None, data_dir=None):
#
#     if dataset_name == 'MNIST-full':
#         data = np.load(os.path.join(data_dir, 'mnist.npz'))
#         trainx = data['x_train'].astype(th.config.floatX)
#         testx = data['x_test'].astype(th.config.floatX)
#         trainy = data['y_train'].astype(np.int32)
#         testy = data['y_test'].astype(np.int32)
#         #
#         trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1], trainx.shape[2]))
#         testx = np.reshape(testx, (testx.shape[0], 1, testx.shape[1], testx.shape[2]))
#
#     elif dataset_name == 'CIFAR-10':
#         def load_cifar_batches(filenames):
#             if isinstance(filenames, str):
#                 filenames = [filenames]
#             images = []
#             labels = []
#             for fn in filenames:
#                 with open(os.path.join(data_dir, 'cifar-10', fn), 'rb') as f:
#                     data = cPickle.load(f, encoding='latin1')
#                 images.append(np.asarray(data['data'], dtype='float32').reshape(-1, 3, 32, 32))
#                 labels.append(np.asarray(data['labels'], dtype='int32'))
#             return np.concatenate(images), np.concatenate(labels)
#
#         trainx, trainy = load_cifar_batches(['data_batch_%d' % i for i in (1, 2, 3, 4, 5)])
#         testx, testy = load_cifar_batches('test_batch')
#
#     if preprocess_inputs == 'whiten':
#         trainx = whiten_norm2d(trainx)
#         testx = whiten_norm2d(testx)
#     elif preprocess_inputs == 'zca':
#         whitener = ZCA(x=trainx)
#         trainx = whitener.apply(trainx)
#         testx = whitener.apply(testx)
#     elif preprocess_inputs == '0_1':
#         trainx = trainx / 255.0
#         testx = testx / 255.0
#
#
#     x = np.concatenate((trainx, testx), axis=0)
#     y = np.concatenate((trainy, testy), axis=0)
#     trainx, trainy, testx, testy = balanced_sample_extraction(x=x, y=y, num_query=num_query)
#
#     trainx = np.asarray(trainx, dtype='float32')
#     trainy = np.asarray(trainy, dtype='int32')
#     testx = np.asarray(testx, dtype='float32')
#     testy = np.asarray(testy, dtype='int32')
#
#     return trainx, trainy, testx, testy

class Logger(object):
    def __init__(self, output_path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(output_path, "log.txt"), "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



# def load_preprocess(dataset_name=None, preprocess_inputs=None, data_dir=None):
#
#     if dataset_name == 'MNIST-full':
#         data = np.load(os.path.join(data_dir, 'mnist.npz'))
#         trainx = data['x_train'].astype(th.config.floatX)
#         testx = data['x_test'].astype(th.config.floatX)
#         trainy = data['y_train'].astype(np.int32)
#         testy = data['y_test'].astype(np.int32)
#
#     if preprocess_inputs == 'whiten':
#         trainx = whiten_norm2d(trainx)
#         testx = whiten_norm2d(testx)
#     elif preprocess_inputs == 'zca':
#         whitener = ZCA(x=trainx)
#         trainx = whitener.apply(trainx)
#         testx = whitener.apply(testx)
#     elif preprocess_inputs == '0_1':
#         trainx = trainx / 255.0
#         testx = testx / 255.0
#
#     if dataset_name == 'MNIST-full':
#         x = np.concatenate((trainx, testx), axis=0)
#         x = np.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#         y = np.concatenate((trainy, testy), axis=0)
#
#     return x, y


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)

    return transform_matrix


def augment_data(imgs, augment_mirror=False, augment_translation=2, augment_rotation=0, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    n, h, w = imgs.shape[0], imgs.shape[2], imgs.shape[3]

    if augment_translation > 0:
        padded_imgs = np.pad(imgs, ((0, 0), (0, 0), (augment_translation, augment_translation),
                             (augment_translation, augment_translation)), 'reflect')

    for i in range(n):
        img = padded_imgs[i]
        #
        if augment_mirror and np.random.uniform() > 0.5:
            img = img[:, :, ::-1]
        #
        if augment_translation > 0:
            t = augment_translation
            ofs0 = np.random.randint(-t, t + 1) + t
            ofs1 = np.random.randint(-t, t + 1) + t
            img = img[:, ofs0:ofs0 + h, ofs1:ofs1 + w]
        #
        if augment_rotation > 0:
            theta = np.pi / 180 * np.random.uniform(-augment_rotation, augment_rotation)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])

            transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
            img = apply_transform(img, transform_matrix, channel_axis, fill_mode, cval)

        imgs[i] = img

    return imgs