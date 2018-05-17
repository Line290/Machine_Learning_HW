import numpy as np
from SVM import SVM
import matplotlib.pyplot as plt

def read_data(PATH):
    f = open(PATH)
    lines = f.readlines()
    # find the number of features
    MAX_FEATURES = 0
    for line in lines:
        for i in line.split()[1:]:
            n = int(i[:i.find(':')])
            if n > MAX_FEATURES:
                MAX_FEATURES = n

    N_train = len(lines)
    X = np.zeros((N_train, MAX_FEATURES+1))
    y = np.zeros(N_train, np.int32)
    for i, line in enumerate(lines):
        y[i] = int(line.split()[0])
        for j in line.split()[1:]:
            n = int(j[:j.find(':')])
            X[i, n] = 1
    return X, y

TRAIN_PATH = 'a3a'
TEST_PATH = 'a3a.t'
X, y = read_data(TRAIN_PATH)
X_test, y_test = read_data(TEST_PATH)
X_test = X_test[:,:-1]

batch = int(X.shape[0]/100)
PASS = [i*batch for i in range(1,25)]
ACC = []
for i in PASS:
    print i
    X_part, y_part = X[:i], y[:i]
    model = SVM(max_iter=10, kernel_type='quadratic', C=1.0, epsilon=0.001)
    model.fit(X_part, y_part)
    y_p = model.predict(X_test)
    acc = np.mean(y_p==y_test)
    print "PASS : {0}, accuracy : {1}".format(i, acc)
    ACC.append(acc)

plt.plot(PASS, ACC)
plt.title("SMO for SVM with data $a3a$")
plt.xlabel("Number of training data in $a3a$")
plt.ylabel("Accuracy in test data $a3a.t$")
plt.show()