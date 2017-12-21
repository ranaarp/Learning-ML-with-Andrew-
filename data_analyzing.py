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

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

print("Mean of x values is %f and median is %f\n" % (np.mean(X), np.median(X)))
print("Mean of y values is %f and median is %f\n" % (np.mean(Y), np.median(Y)))


theta_1_val = np.linspace(-10, 10, num=100)
theta_0_val = np.linspace(-10, 10, num=100)

THETA_0_VAL, THETA_1_VAL = np.meshgrid(theta_0_val, theta_1_val)
print(THETA_0_VAL.shape)
print(THETA_1_VAL.shape)
print(THETA_0_VAL)
print(THETA_1_VAL)


def cost_function(theta_0, theta_1, x_value, y_value, m):
    # This function returns our cost function value at particular theta values
    total_error = 0
    for k in range(m):
        total_error += (theta_0 + theta_1*x_value[k] - y_value[k])**2
    return (total_error/(2*m))

Z = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        Z[i, j] = cost_function(THETA_0_VAL[i, j], THETA_1_VAL[i, j], X, Y, X.size)

print("***Z = ",Z)
Z.reshape((100, 100))
for _ in Z:
    print(_)

# # contour 2D plot
# # Z = np.sqrt(THETA_0_VAL**2 + THETA_1_VAL**2)
# plt.figure()
# cp = plt.contourf(THETA_0_VAL, THETA_1_VAL, Z)
# plt.colorbar(cp)
# # plt.clabel(cp, inline= True, fontsize=5)
# plt.xlabel("Theta_0 values")
# plt.ylabel("Theta_1 values")
# plt.title("Variation of Cost Function with Theta values")
# plt.show()

# #plotting the data
# plt.xlabel("x values")
# plt.ylabel("y values")
# plt.plot(X, Y, "oc")

# x_values_for_line = np.linspace(0, 100, num = 100)
# y_values_for_line = x_values_for_line
# plt.plot(x_values_for_line, y_values_for_line, "k-")
#
# plt.show()

plt.subplot(1, 2, 1)
plt.title('X training set')
plt.boxplot(X)

plt.subplot(1, 2, 2)
plt.title('Y training set')
plt.violinplot(Y)
plt.show()

# xpoint = []
# ypoint = []
# zpoint = []
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(THETA_0_VAL, THETA_1_VAL, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # ax.set_zlim(-1.01, 1.01)
# ax.set_xlabel("theta 0 values")
# ax.set_ylabel("theta 1 values")
# ax.set_zlabel("Cost Function values")
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()
