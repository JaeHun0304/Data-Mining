import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# import dataset using pandas read_csv with whitespace separated and divide into columns (PORT, SIZE, CLASS)
data = pd.read_csv('dataset.txt', sep=" ", header=None)
data.columns = ['port', 'size', 'class']
data['port'] = data['port'].astype(int)
data['size'] = data['size'].astype(int)
data['class'] = pd.factorize(data['class'])[0].astype(np.uint16)

X = np.array(data[['port', 'size']])
Y = np.array(data['class'])


# Selected split points
split_points = [[20000, 0, 0, 100, 30000, 60000, 1000, 25000, 10000, 0], [0, 500, 1500, 0, 0, 0, 0, 0, 0, 2000]]

# create two subplot for scatter plots with linear and log scale port(x-axis)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(data['port'], data['size'], c=data['class'], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
ax1.scatter(split_points[0], split_points[1], color='blue')
ax1.set_xlabel("port")
ax1.set_ylabel("size")
ax1.grid()
ax1.set_title('Scatter plot of PORT vs SIZE for each class')

ax2.scatter(data['port'], data['size'], c=data['class'], cmap=matplotlib.colors.ListedColormap(['red', 'green']))
ax2.scatter(split_points[0], split_points[1], color='blue')
ax2.set_xlabel("port")
ax2.set_ylabel("size")
ax2.set_xscale("log")
ax2.grid()

#using full same dataset for both training and testing
model = DecisionTreeClassifier(max_features=2)
x_train = data[['port', 'size']]
# fit model with input features(port, size) and class label(tcp, udp)
model.fit(X, Y)
# predict accuracy of the model with same full dataset
y_predict = model.predict(X)
print("100%_train_0%_test: " + str(accuracy_score(Y, y_predict)))
# Visulaize generated decision tree and convert output file into .png image file
tree.export_graphviz(model, out_file='tree.dot', feature_names=x_train.columns)
import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

# use 70% for training and 30% for testing by slicing the array
X_train = X[:7000]
Y_train = Y[:7000]
X_test = X[7000:10000]
Y_test = Y[7000:10000]

model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
print("70%_train_30%_test: " + str(accuracy_score(Y_test, y_predict)))

# create different split of data distribution to compare accuracy
X_train = X[:8000]
Y_train = Y[:8000]
X_test = X[8000:10000]
Y_test = Y[8000:10000]

model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
print("80%_train_20%_test: " + str(accuracy_score(Y_test, y_predict)))

X_train = X[:9000]
Y_train = Y[:9000]
X_test = X[9000:10000]
Y_test = Y[9000:10000]

model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
print("90%_train_10%_test: " + str(accuracy_score(Y_test, y_predict)))

X_train = X[:6000]
Y_train = Y[:6000]
X_test = X[6000:10000]
Y_test = Y[6000:10000]

model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
print("60%_train_40%_test: " + str(accuracy_score(Y_test, y_predict)))

X_train = X[:5000]
Y_train = Y[:5000]
X_test = X[5000:10000]
Y_test = Y[5000:10000]

model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
print("50%_train_50%_test: " + str(accuracy_score(Y_test, y_predict)))