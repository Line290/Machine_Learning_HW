import pandas as pd
import os
import numpy as np
import torch as t
import matplotlib.pylab as pl


TRAINING_PATH = './crime-train.txt'
TEST_PATH = './crime-test.txt'
df_train = pd.read_table(TRAINING_PATH)
df_test = pd.read_table(TEST_PATH)
print df_train.head()
print df_test.head()

y_train, X_train = df_train.values[:,0].reshape(-1,1), df_train.values[:,1:]
y_test, X_test = df_test.values[:,0].reshape(-1,1), df_test.values[:,1:]
print 'load training data:\n X_train: {}\n y_train: {}'.format(
    X_train.shape, y_train.shape)
print 'load test data:\n X_test: {}\n y_test: {}'.format(
    X_test.shape, y_test.shape)

# add one col as bias
X_train = np.hstack((np.ones(X_train.shape[0]).reshape(-1,1), X_train))
X_test = np.hstack((np.ones(X_test.shape[0]).reshape(-1,1), X_test))
print X_train.shape, X_test.shape

def loss_gradient(X, y, W, reg, Is_train=True):
    '''Input: 
            X: training data, shape is (N, D+1), the extra one dimenstion is use to add bias, all equal to 1
            y: training data, shape is (N, 1)
            W: weight, shape is (D+1, 1), the extra one dimenstion is bias b
            reg: L2 regularization strength, is a hyperparameter
            Is_train: get loss without regularization 
       Output:
            loss: scala
            dW: shape as W
    '''
    N, D = X.shape
    y_pred = X.dot(W)
    loss = 0.5 * np.sum((y - y_pred)**2)
    if Is_train == True:
        loss += reg * np.sum(W**2)
        dW = -X.T.dot(y - y_pred) 
        dW += 2 * reg * W
        return loss, dW
    else:
        return loss

def train_LS(X, y, learning_rate, reg, num_iters, stop, batch_size, verbose=False):
    N, D = X.shape
    # initialize W
    W = 0.001 * np.random.randn(D, 1)
    # save loss 
    loss_history = []
    
    for it in range(num_iters):
        
        mask = np.random.choice(N, batch_size)
        X_batch = X[mask]
        y_batch = y[mask]
        
        loss, dW = loss_gradient(X_batch, y_batch, W, reg)
#         print loss.shape
        loss_history.append(float(loss))
        # update W
        delta_W = -learning_rate*dW
        W += delta_W
        # verbose
        if verbose and it % 1000 == 0:
            print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
#         print np.linalg.norm(dW)
        if np.max(abs(delta_W)) < stop:
            break
    return loss_history, W

def K_folds_idx(X_train, K):
    N = X_train.shape[0]
    pos = N/K
    idices = np.array(range(N))
    folds = []
    for i in range(K):
        start = i * pos
        end = start + pos
        left_idx, mid_idx, right_idx = np.split(idices, [start, end])
        train_idx = np.append(left_idx, right_idx)
        folds.append((train_idx, mid_idx))
    return folds

# from sklearn.model_selection import StratifiedKFold
# K=10
# folds = list(StratifiedKFold(n_splits=K, 
#                              shuffle=True, 
#                              random_state=16).split(X_train, y_train))
learning_rate = 0.00001
# reg = 0.01
regs = [10**i for i in range(-6, 3)]
num_iters = 3000
stop = 7e-5
threshold = 1e-8

K = 10
folds = K_folds_idx(X_train, K)
best_reg = None
best_W_all = None
best_r2 = 0
best_test_loss = None
results_train = []
results_test = []
num_small_coefs = []

for reg in regs:
    print '==================================regularizzation strength: ', reg
    best_W = None
    best_val_loss = 1000
    for j, (train_idx, val_idx) in enumerate(folds):
        print('\n==============================FOLD=',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val= y_train[val_idx]

        batch_size = X_train_cv.shape[0]
        loss_history, W = train_LS(X=X_train_cv, 
                                   y=y_train_cv,
                                   learning_rate = learning_rate, 
                                   reg = reg, 
                                   num_iters = num_iters, 
                                   stop = stop, 
                                   batch_size = batch_size, 
                                   verbose=True)
        val_loss = loss_gradient(X_val, y_val, W, reg, Is_train=False)
        print 'Train loss: ', loss_history[-1]
        print 'Validation loss: ', val_loss
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_W = W
    y_p = X_train.dot(best_W)
    # number of small coefficients
    num_small_coef = np.argwhere(best_W < threshold).shape[0]
    num_small_coefs.append(num_small_coef)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_train.reshape(-1), y_p.reshape(-1))
    print '======================================R-square score: ', r2
    
    train_loss = loss_gradient(X_train, y_train, best_W, reg, Is_train=False)
    test_loss = loss_gradient(X_test, y_test, best_W, reg, Is_train=False)
    if best_test_loss is None or best_test_loss > test_loss:
        best_r2 = r2
        best_reg = reg
        best_W_all = best_W
        best_test_loss = test_loss
    results_train.append(train_loss)
    results_test.append(test_loss)
    
print "######################### Question 1 ######################"
pl.plot(np.log(np.asarray(regs))/np.log(10), results_train, 'r-')
pl.xlabel('log$(\lambda)$')
pl.ylabel('Squared Error in Training Data')
pl.title('Q1: log$(\lambda)$ VS Squared Error in Training Data')
pl.show()

print "######################### Question 2 ######################"
pl.plot(np.log(np.asarray(regs))/np.log(10), results_test, 'r-')
pl.xlabel('log$(\lambda)$')
pl.ylabel('Squared Error in Test Data')
pl.title('Q2: log$(\lambda)$ VS Squared Error in Test Data')
pl.show()

print "######################### Question 3 ######################"
pl.plot(np.log(np.asarray(regs))/np.log(10), num_small_coefs, 'r-')
pl.xlabel('log$(\lambda)$')
pl.ylabel('Number of Small Coefficients')
pl.title('Q3: log$(\lambda)$ VS Number of Small Coefficients')
pl.show()

print "######################### Question 4 ######################"
print "The best test set proformance: lambda = {}, test loss = {}".format(best_reg, best_test_loss)
print "The largest coefficient: ", best_W_all.max()
print "The smallest coefficient: ", best_W_all.min()