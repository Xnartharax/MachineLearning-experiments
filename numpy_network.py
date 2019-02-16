import pandas as pd
import numpy as np
import json
from math import exp, log, floor, sqrt
import sys
import matplotlib.pyplot as plt
from random import random
from sklearn import datasets
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle
error_plt = []
global epsilon
epsilon = 1e-4


def sqrt_plus(x):
    z = []
    for y in x:
        q = []
        for v in y:
            q.append(np.sqrt(v))
        z.append(q)
    return z


def sigmoid(x):
    try:
        return 1/(1+exp(x))
    except OverflowError:
        if x < 0:
            return 0.0
        else:
            return 1.0


def softmax(ins):
    d = max(ins)
    z = sum([exp(x-d)for x in ins])
    return [exp(y-d)/z for y in ins]


class NetworkEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (DenseLayer, DropoutLayer)):
            return obj.w, obj.b
        return json.JSONEncoder.default(self, obj)


class Debugger:
    def __init__(self, model):
        self.model = model

    def gradient_checking(self, loss, x, y):
        shape = self.extract_structure()
        param_num = sum([x*y+x for x, y in zip(shape[:-1], shape[1:])])
        grd = self.model.backpropagate_gradient(loss, x[0], y[0], 0)
        check_grd = grd.copy()
        for q, layer in enumerate(self.model.structure):
            for i, n in enumerate(layer.w):
                for j, w in enumerate(n):
                    layer.w[i, j] += epsilon
                    err1 = self.model.get_error(loss, x, y)
                    layer.w[i, j] -= 2*epsilon
                    err2 = self.model.get_error(loss, x, y)
                    layer.w[i, j] += epsilon
                    check_grd[q][0][i, j] = (err1-err2)/2*epsilon
            for i, b in enumerate(layer.b):
                layer.b[i] += epsilon
                err1 = self.model.get_error(loss, x, y)
                layer.b[i] -= 2 * epsilon
                err2 = self.model.get_error(loss, x, y)
                check_grd[q][1][i] = (err1 - err2) / 2 * epsilon
        deviation = 0
        for layer, check_layer in zip(grd, check_grd):
            deviation += np.sum(layer[0]-check_layer[0])
            deviation += np.sum(layer[1] - check_layer[1])
        return deviation/param_num

    def extract_structure(self):
        shape = [self.model.structure[0].w.shape[1]]
        for layer in self.model.structure:
            shape.append(layer.w.shape[0])
        return shape


class ActFunc:
    def normal(self, ins):
        pass

    def der(self, ins):
        pass


class Sigmoid(ActFunc):
    def normal(self, ins):
        return np.array([sigmoid(x)for x in ins])

    def der(self, ins):
        return [x*(1-x)for x in self.normal(ins)]


class LeakyReLU(ActFunc):
    def __init__(self, leak=0.001):
        self.leak = leak

    def normal(self, ins):
        return [x if x > 0 else x*self.leak for x in ins]

    def der(self, ins):
        return [1 if x > 0 else self.leak for x in ins]


class Logit(ActFunc):
    def normal(self, ins):
        return ins

    def der(self, ins):
        return np.ones_like(ins)


class ReLU(ActFunc):
    def normal(self, ins):
        return [x if x>0 else 0 for x in ins]

    def der(self, ins):
        return [1 if x > 0 else 0 for x in ins]


class Loss:
    def normal(self, outs, targets):
        pass

    def der(self, outs, targets):
        pass


class SoftmaxCrossEntropy(Loss):
    def normal(self, outs, targets):
        _outs = softmax(outs)
        return sum([t*log(y+epsilon)for y, t in zip(_outs, targets)])*(-1)

    def der(self, outs, targets):
        _outs = softmax(outs)
        return np.subtract(_outs, targets)


class DenseLayer:
    def __init__(self, in_len, out_len, activation_func):
        self.w = (np.random.rand(out_len, in_len)-0.5)/sqrt(in_len)
        self.act_func = activation_func
        self.b = np.zeros(out_len)

    def compute(self, ins):
        return self.act_func.normal(np.add(np.matmul(self.w, ins), self.b))

    def run(self, ins):
        return self.compute(ins)

    def delta_compute(self, ins):
        return self.act_func.der(np.add(np.matmul(self.w, ins), self.b))

    def der_run(self, ins):
        return self.compute(ins), self.delta_compute(ins)


class DropoutLayer(DenseLayer):
    def __init__(self, in_len, out_len, activation_func, drop):
        super(DropoutLayer, self).__init__(in_len, out_len, activation_func)
        self.drop = drop

    def run(self, ins):
        outs = self.act_func.normal(np.add(np.matmul(self.w, ins), self.b))
        return np.multiply(outs, (1-self.drop))

    def der_run(self, ins):
        mask = [0 if random() < self.drop else 1 for x in range(self.w.shape[0])]
        return np.multiply(self.compute(ins), mask), np.multiply(self.delta_compute(ins), mask)


class Model:
    def __init__(self, structure):
        self.structure = structure

    def run(self, ins):
        steps = [ins]
        for layer in self.structure:
            steps.append(layer.run(steps[-1]))
        return steps

    def der_run(self, ins):
        steps = [(ins, np.zeros_like(ins))]
        for layer in self.structure:
            steps.append(layer.der_run(steps[-1][0]))
        return steps

    def fit_adam(self, loss, x, y, x_val, y_val, epochs, learning_rate=0.1, ß1=0.9, ß2=0.99999,
                 steps_per_epoch=3000, error_plot=error_plt, l2_reg=0.02):
        batch_size = floor(x.shape[0]/steps_per_epoch)
        batch_grd = self.backpropagate_gradient(loss, x[0], y[0], l2_reg)
        m = batch_grd
        v = np.abs(batch_grd)
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                err = self.get_error(loss, x_val, y_val)
                sys.stdout.write('\r{} | {} | {} | {}'.format(round(step/steps_per_epoch, 3), epoch,
                                                              self.validate(x_val, y_val), err))
                error_plot.append(err)
                sys.stdout.flush()
                x_batch = x[step*batch_size:(step+1)*batch_size]
                y_batch = y[step*batch_size:(step+1)*batch_size]
                batch_grd = self.get_batch_gradient(loss, x_batch, y_batch, l2_reg)
                m = np.add(np.multiply(m, ß1), np.multiply(batch_grd, 1-ß1))
                v = np.add(np.multiply(v, ß2), np.multiply(np.square(batch_grd), 1-ß2))
                m_ = m/(1-ß1)
                v_ = v/(1-ß2)
                update = np.divide(np.multiply(learning_rate, m_), np.add([[np.sqrt(vj)for vj in vi]for vi in v_], epsilon))
                self.update(update)

    def fit_momentum(self, loss, x_train, y_train, x_val, y_val, epochs, learning_rate=0.001,
                     steps_per_epoch=3000, error_plot=error_plt, l2_reg=0.02, decay_start=0.5, decay_max=0.9, decay_growth=0.5):
        batch_size = floor(x_train.shape[0] / steps_per_epoch)
        decay = decay_start
        momentum = self.get_batch_gradient(loss, x_train[0:batch_size], y_train[0:batch_size], l2_reg)
        momentum = np.multiply(momentum, learning_rate)
        self.update(momentum)
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                decay += (decay_max-decay)*decay_growth
                grd = self.get_batch_gradient(loss, x_train[step*batch_size:(step+1)*batch_size],
                                              y_train[step*batch_size:(step+1)*batch_size], l2_reg)
                momentum = np.add(np.multiply(momentum, decay), np.multiply(grd, learning_rate*(decay)))
                self.update(momentum)
                err = self.get_error(loss, x_val, y_val)
                sys.stdout.write(
                    '\r{} | {} | {} | {}'.format(round(step / steps_per_epoch, 3), epoch, self.validate(x_val, y_val),
                                                 err))
                error_plot.append(err)
                sys.stdout.flush()

    def fit_adaptive(self, loss, x_train, y_train, x_val, y_val, epochs, learning_rate=0.001,
                     steps_per_epoch=3000, error_plot=error_plt, l2_reg=0.02):
        gains = self.backpropagate_gradient(loss, x_train[0], y_train[0])
        batch_size = floor(x_train.shape[0] / steps_per_epoch)
        grd_bef = gains
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                grd = self.get_batch_gradient(loss, x_train[step * batch_size:(step + 1) * batch_size],
                                              y_train[step * batch_size:(step + 1) * batch_size], l2_reg)
                update = np.multiply(grd, gains)
                grd_bef = grd

    def fit_sgd(self, loss, x_train, y_train, x_val, y_val, epochs, learning_rate=0.001,
                     steps_per_epoch=3000, error_plot=error_plt, l2_reg=0.02):
        batch_size = floor(x_train.shape[0] / steps_per_epoch)
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                update = self.get_batch_gradient(loss, x_train[step*batch_size:(step+1)*batch_size],
                                              y_train[step*batch_size:(step+1)*batch_size], l2_reg) * learning_rate
                self.update(update)
                err = self.get_error(loss, x_val, y_val)
                sys.stdout.write(
                    '\r{} | {} | {} | {}'.format(round(step / steps_per_epoch, 3), epoch,
                                                 self.validate(x_val, y_val), err))
                error_plot.append(err)
                sys.stdout.flush()

    def fit_nesterov(self, loss, x_train, y_train, x_val, y_val, epochs, learning_rate=0.001,
                     decay=0.9, steps_per_epoch=3000, error_plot=error_plt, l2_reg=0.02):
        batch_size = floor(x_train.shape[0] / steps_per_epoch)
        for epoch in range(epochs):
            momentum = self.get_batch_gradient(loss, x_train[0:batch_size], y_train[0:batch_size], l2_reg)
            momentum = np.multiply(momentum, learning_rate)
            self.update(momentum)
            for step in range(steps_per_epoch):
                grd = self.get_batch_gradient(loss, x_train[step * batch_size:(step + 1) * batch_size],
                                              y_train[step * batch_size:(step + 1) * batch_size], l2_reg)
                self.update(momentum)
                momentum = np.add(np.multiply(momentum, decay), np.multiply(grd, learning_rate * (decay)))

                err = self.get_error(loss, x_val, y_val)
                sys.stdout.write(
                    '\r{} | {} | {} | {}'.format(round(step / steps_per_epoch, 3), epoch, self.validate(x_val, y_val),
                                                 err))
                error_plot.append(err)
                sys.stdout.flush()

    def validate(self, x_val, y_val):
        right = 0
        for x, y in zip(x_val, y_val):
            if np.argmax(self.run(x)[-1]) == np.argmax(y):
                right += 1
        return round(right/x_val.shape[0], 3)

    def update(self, update):
        for u_layer, m_layer in zip(update, self.structure):
            m_layer.w = np.subtract(m_layer.w, u_layer[0])
            m_layer.b = np.subtract(m_layer.b, u_layer[1])

    def get_batch_gradient(self, loss, x, y, l2_reg):
        avg_grd = self.backpropagate_gradient(loss, x[0, :], y[0, :], l2_reg)
        for ins, targets in zip(x[1:, :], y[1:, :]):
            grd = self.backpropagate_gradient(loss, ins, targets, l2_reg)
            for i, (add_layer, avg_layer) in enumerate(zip(grd, avg_grd)):
                avg_grd[i][0] = np.add(add_layer[0], avg_layer[0])
                avg_grd[i][1] = np.add(add_layer[1], avg_layer[1])
        return np.divide(avg_grd, x.shape[0])

    def backpropagate_gradient(self, loss, x, y, l2_reg):
        steps = self.der_run(x)
        act_ders = loss.der(steps[-1][0], y)
        gradient = []
        for act_before, act_now, layer in zip(reversed(steps[:-1]), reversed(steps[1:]), reversed(self.structure)):
            dY = np.array(act_now[1]) * act_ders
            gradient.insert(0, [np.outer(dY, act_before[0]) + layer.w * l2_reg,
                                dY + layer.b * l2_reg])

            act_ders = np.dot(np.rollaxis(layer.w, 1, 0), dY)
        return gradient

    def save_weights(self, path='./network.json'):
        with open(path, 'w') as fp:
            json.dump(self.structure, fp, cls=NetworkEncoder)

    def load_weights(self, path='./network.json'):
        with open(path, 'r') as fp:
            weights = json.load(fp)
            for load_layer, model_layer in zip(weights, self.structure):
                model_layer.w = np.array(load_layer[0])
                model_layer.b = np.array(load_layer[1])

    def get_error(self, loss, x_test, y_test):
        return sum([loss.normal(self.run(x)[-1], y) for x, y in zip(x_test, y_test)])/len(x_test)


# model initialization

test_model = Model(
    [DenseLayer(784, 300, LeakyReLU()),
     DenseLayer(300, 100, LeakyReLU()),
     DenseLayer(100, 10, Logit())]
)
# data loading and preprocessing
raw = pd.read_csv('./mnist_train.csv').values
data = {'data': [(row[1:]/255)-0.5 for row in raw], 'target': [row[0]for row in raw]}

x_train, x_test, y_train, y_test = train_test_split(data.get('data'), data.get('target'), test_size=0.02)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array([[1 if x == label else 0 for x in range(10)]for label in y_train])
y_test = np.array([[1 if x == label else 0 for x in range(10)]for label in y_test])

# dbg = Debugger(test_model)

print('{} datasets loaded'.format(x_train.shape[0]))
# print(dbg.gradient_checking(SoftmaxCrossEntropy(), [x_test[0]], [y_test[0]]))
print(test_model.get_error(SoftmaxCrossEntropy(), x_train[:50, :], y_train[:50, :]))
print('begin training')
test_model.load_weights(path='./network1.json')
# actual training
test_model.fit_adam(SoftmaxCrossEntropy(), x_train, y_train, x_test, y_test, 1,
                    steps_per_epoch=400, learning_rate=0.01, l2_reg=0)

test_model.save_weights(path='./network1.json')

# performance test
print(test_model.run(x_train[0])[-1])

#print(y_train[0])
plt.plot(error_plt)
plt.show()
print('\nfinished training')
print(test_model.get_error(SoftmaxCrossEntropy(), x_test, y_test))
print(test_model.validate(x_test, y_test))

