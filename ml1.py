from sklearn import neighbors, svm, ensemble, tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


digits = load_digits();
ov = fetch_olivetti_faces()
print('done')  # here to provide an update when the dataset finished downloading.
dataset = ov
setRange = 50;

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)

cv_list = []
tr_list = []
tst_list = []

# For Decision Trees:

'''for max_depth in range(setRange):
    classifier = tree.DecisionTreeClassifier(max_depth=max_depth+1)
    # print("CV Accuracy: ", cross_val_score(classifier, X_train, y_train, cv=5).mean())  # cross validation score
    cv_list.append(cross_val_score(classifier, X_train, y_train, cv=5).mean())

    classifier = classifier.fit(X_train, y_train)

    #print("TR Accurary: ", classifier.score(X_train, y_train))  #
    tr_list.append(classifier.score(X_train, y_train))
    #print("TST Accurary: ", classifier.score(X_test, y_test))
    tst_list.append(classifier.score(X_test, y_test))

    # y_predicted = classifier.predict(X_test)
    print('iteration ', max_depth)
    
plt.plot(cv_list, label="Cross Validation")
plt.plot(tr_list, label="Training")
plt.plot(tst_list, label="Testing")
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Trees on OV')
plt.legend()
plt.show()'''
    


# For Boosting:

'''for maxdepth in range(setRange):
    classifier = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=maxdepth+1))
    # print("CV Accuracy: ", cross_val_score(classifier, X_train, y_train, cv=5).mean())  # cross validation score
    cv_list.append(cross_val_score(classifier, X_train, y_train, cv=5).mean())

    classifier = classifier.fit(X_train, y_train)

    #print("TR Accurary: ", classifier.score(X_train, y_train))  #
    tr_list.append(classifier.score(X_train, y_train))
    #print("TST Accurary: ", classifier.score(X_test, y_test))
    tst_list.append(classifier.score(X_test, y_test))

    y_predicted = classifier.predict(X_test)
    print(maxdepth)
    
plt.plot(cv_list, label="Cross Validation")
plt.plot(tr_list, label="Training")
plt.plot(tst_list, label="Testing")
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Boosting on OV')
plt.legend()
plt.show()'''

# For K-nearest Neighbors

'''for n in range(setRange):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=n+1)
    # print("CV Accuracy: ", cross_val_score(classifier, X_train, y_train, cv=5).mean())  # cross validation score
    cv_list.append(cross_val_score(classifier, X_train, y_train, cv=5).mean())

    classifier = classifier.fit(X_train, y_train)

    #print("TR Accurary: ", classifier.score(X_train, y_train))  #
    tr_list.append(classifier.score(X_train, y_train))
    #print("TST Accurary: ", classifier.score(X_test, y_test))
    tst_list.append(classifier.score(X_test, y_test))
    print(n)

    y_predicted = classifier.predict(X_test)
    
plt.plot(cv_list, label="Cross Validation")
plt.plot(tr_list, label="Training")
plt.plot(tst_list, label="Testing")
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('K-nearest Neighbors on OV')
plt.legend()
plt.show()'''

# For Support Vector Machines

'''SVC = svm.SVC(gamma='auto');
SigmoidSVC = svm.SVC(kernel='sigmoid', gamma='auto');
LinearSVC = svm.LinearSVC();

#RBF
cv_list.append(cross_val_score(SVC, X_train, y_train, cv=5).mean())
SVC.fit(X_train, y_train)
tr_list.append(SVC.score(X_train, y_train))
tst_list.append(SVC.score(X_test, y_test))

#Sigmoid
cv_list.append(cross_val_score(SigmoidSVC, X_train, y_train, cv=5).mean())
SigmoidSVC.fit(X_train, y_train)
tr_list.append(SigmoidSVC.score(X_train, y_train))
tst_list.append(SigmoidSVC.score(X_test, y_test))

#Linear
cv_list.append(cross_val_score(LinearSVC, X_train, y_train, cv=5).mean())
LinearSVC.fit(X_train, y_train)
tr_list.append(LinearSVC.score(X_train, y_train))
tst_list.append(LinearSVC.score(X_test, y_test))

N = 3

ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, tr_list, width)

rects2 = ax.bar(ind+width, cv_list, width)

rects3 = ax.bar(ind+width+width, tst_list, width)

# add some
ax.set_ylabel('Scores')
ax.set_title('Support Vector Machines on OV')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels( ('RBF', 'Sigmoid', 'Linear') )

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Training', 'Cross Validation', 'Testing'))

plt.show()'''

# For Neural Network -- Neuron Tuning

'''scaler = StandardScaler()
scaler.fit(X_train)
input_list = scaler.transform(X_train)
test_input = scaler.transform(X_test)

for neuron in range(50):
    layers = [neuron+1]

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1)

    cv_list.append(cross_val_score(classifier, X_train, y_train, cv=5).mean())

    classifier = classifier.fit(X_train, y_train)

    tr_list.append(classifier.score(X_train, y_train))

    tst_list.append(classifier.score(X_test, y_test))

    y_predicted = classifier.predict(X_test)

    print(neuron)

plt.plot(cv_list, label="Cross Validation")
plt.plot(tr_list, label="Training")
plt.plot(tst_list, label="Testing")
plt.xlabel('Neurons')
plt.ylabel('Accuracy')
plt.title('Neural Network on OV')
plt.legend()
plt.show()'''

# For Neuron Network -- Layer Tuning

'''scaler = StandardScaler()
scaler.fit(X_train)
input_list = scaler.transform(X_train)
test_input = scaler.transform(X_test)

dIdeal = 19
OVIdeal = 46

for neurons in range(setRange):
    layers = []
    for neuron in range(neurons):
        layers.append(OVIdeal)
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1)

    cv_list.append(cross_val_score(classifier, X_train, y_train, cv=5).mean())

    classifier.fit(X_train, y_train)

    tr_list.append(classifier.score(X_train, y_train))

    tst_list.append(classifier.score(X_test, y_test))

    y_predicted = classifier.predict(X_test)

    print(neurons)

plt.plot(cv_list, label="Cross Validation")
plt.plot(tr_list, label="Training")
plt.plot(tst_list, label="Testing")
plt.xlabel('Layers')
plt.ylabel('Accuracy')
plt.title('Neural Network on OV')
plt.legend()
plt.show()'''

# For algorithm comparison -- Digits

'''scaler = StandardScaler()
layers = []
for neuron in range(5):
    layers.append(19)

tree_cv_list = []
tree_tr_list = []
boost_cv_list = []
boost_tr_list = []
k_cv_list = []
k_tr_list = []
svc_cv_list = []
svc_tr_list = []
nn_cv_list = []
nn_tr_list = []

for input_size in range(1, int(len(X_train) / 100)):
    input_partition = X_train[:input_size * 100]
    input_nn_partition = X_train[:input_size * 100]
    scaler.fit(input_nn_partition)
    input_nn_partition = scaler.transform(input_nn_partition)
    output_partition = y_train[:input_size * 100]

    dTree = tree.DecisionTreeClassifier(max_depth=6)
    aBoost = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=3))
    kn = neighbors.KNeighborsClassifier(n_neighbors=1)
    LSVC = svm.LinearSVC();
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1)

    tree_cv_list.append(cross_val_score(dTree, input_partition, output_partition, cv=5).mean())
    boost_cv_list.append(cross_val_score(aBoost, input_partition, output_partition, cv=5).mean())
    k_cv_list.append(cross_val_score(kn, input_partition, output_partition, cv=5).mean())
    svc_cv_list.append(cross_val_score(LSVC, input_partition, output_partition, cv=5).mean())
    nn_cv_list.append(cross_val_score(nn, input_nn_partition, output_partition, cv=5).mean())

    dTree.fit(input_partition, output_partition)
    aBoost.fit(input_partition, output_partition)
    kn.fit(input_partition, output_partition)
    LSVC.fit(input_partition, output_partition)
    nn.fit(input_nn_partition, output_partition)

    tree_tr_list.append(dTree.score(input_partition, output_partition))
    boost_tr_list.append(aBoost.score(input_partition, output_partition))
    k_tr_list.append(kn.score(input_partition, output_partition))
    svc_tr_list.append(LSVC.score(input_partition, output_partition))
    nn_tr_list.append(nn.score(input_nn_partition, output_partition))
    print(input_size * 100)

plt.plot(tree_cv_list, label="Decision Tree CV")
plt.plot(tree_tr_list, label="Decision Tree Train")
plt.plot(boost_cv_list, label="Boosting CV")
plt.plot(boost_tr_list, label="Boosting Tree Train")
plt.plot(k_cv_list, label="K Neighbors CV")
plt.plot(k_tr_list, label="K Neighbors Train")
plt.plot(svc_cv_list, label="SVC CV")
plt.plot(svc_tr_list, label="SVC Train")
plt.plot(nn_cv_list, label="NN CV")
plt.plot(nn_tr_list, label="NN Train")

plt.xlabel('Input Size /100')
plt.ylabel('Accuracy')
plt.title('Algorithm Accuracy Comparison per Input Size on Digits')
plt.legend()
plt.show()'''

# For algorithm comparison on OV

'''scaler = StandardScaler()
layers = []
for neuron in range(4):
    layers.append(46)

tree_cv_list = []
tree_tr_list = []
boost_cv_list = []
boost_tr_list = []
k_cv_list = []
k_tr_list = []
svc_cv_list = []
svc_tr_list = []
nn_cv_list = []
nn_tr_list = []

print(len(X_train))

for input_size in range(6, int(len(X_train) / 20)):
    input_partition = X_train[:input_size * 20]
    input_nn_partition = X_train[:input_size * 20]
    scaler.fit(input_nn_partition)
    input_nn_partition = scaler.transform(input_nn_partition)
    output_partition = y_train[:input_size * 20]

    dTree = tree.DecisionTreeClassifier(max_depth=41)
    aBoost = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=21))
    kn = neighbors.KNeighborsClassifier(n_neighbors=6)
    LSVC = svm.LinearSVC();
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1)

    tree_cv_list.append(cross_val_score(dTree, input_partition, output_partition, cv=5).mean())
    boost_cv_list.append(cross_val_score(aBoost, input_partition, output_partition, cv=5).mean())
    k_cv_list.append(cross_val_score(kn, input_partition, output_partition, cv=5).mean())
    svc_cv_list.append(cross_val_score(LSVC, input_partition, output_partition, cv=5).mean())
    nn_cv_list.append(cross_val_score(nn, input_nn_partition, output_partition, cv=5).mean())

    dTree.fit(input_partition, output_partition)
    aBoost.fit(input_partition, output_partition)
    kn.fit(input_partition, output_partition)
    LSVC.fit(input_partition, output_partition)
    nn.fit(input_nn_partition, output_partition)

    tree_tr_list.append(dTree.score(input_partition, output_partition))
    boost_tr_list.append(aBoost.score(input_partition, output_partition))
    k_tr_list.append(kn.score(input_partition, output_partition))
    svc_tr_list.append(LSVC.score(input_partition, output_partition))
    nn_tr_list.append(nn.score(input_nn_partition, output_partition))
    print(input_size * 20)

plt.plot(tree_cv_list, label="Decision Tree CV")
plt.plot(tree_tr_list, label="Decision Tree Train")
plt.plot(boost_cv_list, label="Boosting CV")
plt.plot(boost_tr_list, label="Boosting Tree Train")
plt.plot(k_cv_list, label="K Neighbors CV")
plt.plot(k_tr_list, label="K Neighbors Train")
plt.plot(svc_cv_list, label="SVC CV")
plt.plot(svc_tr_list, label="SVC Train")
plt.plot(nn_cv_list, label="NN CV")
plt.plot(nn_tr_list, label="NN Train")

plt.xlabel('Input Size /20')
plt.ylabel('Accuracy')
plt.title('Algorithm Accuracy Comparison per Input Size on OV')
plt.legend()
plt.show()'''


# sklearn using information gain by default