import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from matplotlib.colors import Normalize, ListedColormap

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


df = pd.read_csv('nba_rookie_data.csv')

#to display all columns
pd.set_option('display.max_columns', None)

#print(df.head(),'\n')

# to view the correllation relationship
#print(df.corr())

# a summary of the data set
#print(df.info(), '\n')

# to view the number of null value in the variable
#print(df.isnull().sum())

# dropping the name variable since its a string column
df = df.drop(["Name"], axis=1)

# dropping the nan row in the variable
df = df.dropna(subset=["3 Point Percent"])

#print(df.head(),'\n')


X = df.iloc[:, [0,8]].values      #all columns after the first column
y = df.iloc[:, -1].values       #the first column


# we normalise selective X column in the data to improve the model
#X[:,[1]] = StandardScaler().fit_transform(X[:,[1]])


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size= 1/5, random_state=42)

# we normalise the X columns in the data to improve the model
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#print(X_test)


########################
## LOGISTIC REGRESSION MODEL
########################

# construct model and fit to the training data
logre = LogisticRegression()
logre.fit(X_train, y_train)

print("For LOGISTIC REGRESSION")
# output accuracy score
print('Our Accuracy is %.2f' % logre.score(X_test, y_test))
# output the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
		% (X_test.shape[0], (y_test != logre.predict(X_test)).sum()))


# visualise the model in sigmoid
# fig1, ax1 = plt.subplots()

# ax1.scatter(X_test, y_test, color='blue')
# ax1.scatter(X_test, logre.predict(X_test), color='red', marker='*')
# ax1.scatter(X_test, logre.predict_proba(X_test)[:,1], color='green', marker='.')

# ax1.set_xlabel('Games Played')
# ax1.set_ylabel('TARGET_5Yrs')
# ax1.set_title('LOGISTIC REGRESSION MODEL')

#fig1.savefig('logre_plot.png')

'''
# visualise the LOGRE model
fig, ax = plt.subplots()

# need to set up a mesh to plot the contour of the model
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# step size in the mesh
h = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
	np.arange(y_min, y_max, h))

# model predicts every point in the mesh and reshapes the array for plotting
Z = logre.predict(np.column_stack([xx.ravel(), 
	yy.ravel()]))
Z = Z.reshape(xx.shape)

# set up the color and symbol encoding
nm = Normalize(vmin = 0, vmax = 1)
cm = ListedColormap(['blue', 'red'])
m = ['o', '^']

# contour plot of the model
ax.contourf(xx, yy, Z, cmap = cm, norm = nm, alpha=0.5)

# plot the data
for i in range(len(X_test)):
	ax.scatter(X_test[i,0], X_test[i,1], 
		marker = m[logre.predict(X_test)[i]], 
		c = y_test[i], cmap = cm, norm = nm, s = 10)

# find the misclassified points
mis_ind = np.where(y_test != logre.predict(X_test))[0]
# print('Misclassified Points:\n', X_test[mis_ind], 
# 	y_test[mis_ind])

# plot the misclassified points
ax.scatter(X_test[mis_ind,0], X_test[mis_ind,1], 
		marker = '*', color = 'white', s = 2)

ax.set_xlabel('Games Played')
ax.set_ylabel('3 Point Percent')
ax.set_title('LOGISTIC REGRESSION MODEL')

#fig.savefig('logre_plot2.png')
'''





########################
## GAUSSIAN NAIVE BAYES MODEL
########################

# construct model and fit to the training data
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("\n, For GAUSSIAN NAIVE")
# output accuracy score
print('Our Accuracy is %.2f:' % gnb.score(X_test, y_test))
# number of mislabeled points
print('Number of mislabeled points out of a total of %d points: %d' 
	% (X_test.shape[0], (y_test != gnb.predict(X_test)).sum()))


# visualise the model in sigmoid
# fig2, ax2 = plt.subplots()

# ax2.scatter(X_test, y_test, color='blue')
# ax2.scatter(X_test, gnb.predict(X_test), color='red', marker='*')
# ax2.scatter(X_test, gnb.predict_proba(X_test)[:,1], color='green', marker='.')

# ax2.set_xlabel('Games Played')
# ax2.set_ylabel('TARGET_5Yrs')
# ax2.set_title('GAUSSIAN NAIVE BAYES MODEL')

#fig2.savefig('gnb_plot.png')


#to predict
# y_pred = gnb.predict([[]])
# print('Predict a value:', y_pred)


'''
# visualise the GNB model
fig, ax = plt.subplots()

# need to set up a mesh to plot the contour of the model
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# step size in the mesh
h = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
	np.arange(y_min, y_max, h))

# model predicts every point in the mesh and reshapes the array for plotting
Z = gnb.predict(np.column_stack([xx.ravel(), 
	yy.ravel()]))
Z = Z.reshape(xx.shape)

# set up the color and symbol encoding
nm = Normalize(vmin = 0, vmax = 1)
cm = ListedColormap(['blue', 'red'])
m = ['o', '^']

# contour plot of the model
ax.contourf(xx, yy, Z, cmap = cm, norm = nm, alpha=0.5)

# plot the data
for i in range(len(X_test)):
	ax.scatter(X_test[i,0], X_test[i,1], 
		marker = m[gnb.predict(X_test)[i]], 
		c = y_test[i], cmap = cm, norm = nm, s = 10)

# find the misclassified points
mis_ind = np.where(y_test != gnb.predict(X_test))[0]
# print('Misclassified Points:\n', X_test[mis_ind], 
# 	y_test[mis_ind])

# plot the misclassified points
ax.scatter(X_test[mis_ind,0], X_test[mis_ind,1], 
		marker = '*', color = 'white', s = 2)

ax.set_xlabel('Games Played')
ax.set_ylabel('3 Point Percent')
ax.set_title('GAUSSIAN NAIVE BAYES MODEL')

#fig.savefig('gnb_plot2.png')
'''





########################
## NEURAL NETWORK MODEL
########################

# change activation function: (tanh, relu, logistic) and layers
mlp = MLPClassifier(hidden_layer_sizes=(40,80,120), 
	activation="logistic" ,random_state=42, max_iter=2000)
mlp.fit(X_train, y_train)

# performance metrics
print("\n, For NEURAL NETWORK")
print('Our Accuracy is %.2f' % mlp.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
		% (X_test.shape[0], (y_test != mlp.predict(X_test)).sum()))


# visualise the model in sigmoid
# fig3, ax3 = plt.subplots()

# ax3.scatter(X_test, y_test, color='blue')
# ax3.scatter(X_test, mlp.predict(X_test), color='red', marker='*')
# ax3.scatter(X_test, mlp.predict_proba(X_test)[:,1], color='green', marker='.')

# ax3.set_xlabel('Games Played')
# ax3.set_ylabel('TARGET_5Yrs')
# ax3.set_title('NEURAL NETWORK MODEL')

#fig3.savefig('mlp_plot.png')



# visualise the NN model
fig, ax = plt.subplots()

# need to set up a mesh to plot the contour of the model
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# step size in the mesh
h = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
	np.arange(y_min, y_max, h))

# model predicts every point in the mesh and reshapes the array for plotting
Z = mlp.predict(np.column_stack([xx.ravel(), 
	yy.ravel()]))
Z = Z.reshape(xx.shape)

# set up the color and symbol encoding
nm = Normalize(vmin = 0, vmax = 1)
cm = ListedColormap(['blue', 'red'])
m = ['o', '^']

# contour plot of the model
ax.contourf(xx, yy, Z, cmap = cm, norm = nm, alpha=0.5)

# plot the data
for i in range(len(X_test)):
	ax.scatter(X_test[i,0], X_test[i,1], 
		marker = m[mlp.predict(X_test)[i]], 
		c = y_test[i], cmap = cm, norm = nm, s = 10)

# find the misclassified points
mis_ind = np.where(y_test != mlp.predict(X_test))[0]
# print('Misclassified Points:\n', X_test[mis_ind], 
# 	y_test[mis_ind])

# plot the misclassified points
ax.scatter(X_test[mis_ind,0], X_test[mis_ind,1], 
		marker = '*', color = 'white', s = 2)

ax.set_xlabel('Games Played')
ax.set_ylabel('Offensive Rebounds')
ax.set_title('NEURAL NETWORK MODEL')

#fig.savefig('NN_plot.png')



