import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


raw_training_data = pd.read_csv("/home/rana/Desktop/train.csv")
raw_testing_data = pd.read_csv("/home/rana/Desktop/test.csv")
print(raw_training_data.size)
train_data = raw_training_data.dropna()
test_data = raw_testing_data.dropna()

# print(train_data.shape)
# print(train_data['x'])

X = train_data.as_matrix(columns= ['x']).reshape(699)
Y = train_data.as_matrix(columns= ['y']).reshape(699)

print("Mean of x values is %f and median is %f\n" % (np.mean(X), np.median(X)))
print("Mean of y values is %f and median is %f\n" % (np.mean(Y), np.median(Y)))

# plotting the box plots
plt.subplot(1, 2, 1)
plt.title('X training set')
plt.boxplot(X)

plt.subplot(1, 2, 2)
plt.title('Y training set')
plt.boxplot(Y)
plt.show()

