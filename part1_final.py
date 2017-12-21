import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import style
from matplotlib import rcParams
rcParams.update({'font.size': 16})

# style.use('ggplot')


def hypothesis(theta_array, x):
    # Returns value of hypothesis at the point corresponding to the 'x' entry
    # H(x) = theta_0 + theta_1*x
    return theta_array[0] + theta_array[1]*x


def cost_function(theta_array, x_value, y_value, m):
    # This function returns our cost function value at particular theta values
    total_error = 0
    for i in range(m):
        total_error += (theta_array[0] + theta_array[1]*x_value[i] - y_value[i])**2
    return total_error/(2*m)


def training(x_train, y_train, alpha, iters):
    # This is the function which takes care of the Regression

    # Finding the size of the
    m = x_train.size

    # initializing values of thetas
    theta_0 = 0     # bias
    theta_1 = 0     # weight

    # creating a weight matrix which contains both theta_0 and theta_1
    theta_array = [theta_0, theta_1]

    # creating an array that stores the values of the cost function during each iteration
    cost_function_values = []

    for i in range(iters):

        # plotting our hypothesis across the training data after each iteration until 10 iterations
        if i < 1:
            # finding the range of x values across which we would like to plot our hypothesis
            lower_x_coordinate = min(x_train)
            higher_x_coordinate = max(x_train)

            # setting up the x and y coordinate values for plotting our hypothesis
            x_coordinate_for_line = np.arange(lower_x_coordinate, stop=higher_x_coordinate, step=1)
            y_coordinate_for_line = x_coordinate_for_line * theta_array[1] + theta_array[0]
            # y(x) = m*x +c is the final line
            fig = plt.figure(figsize=(8, 12))
            plt.tight_layout()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2, projection= '3d')
            ax1.set_title("Plotting our hypothesis over training data")
            plt.suptitle("iteration number : %d" % i).set_size(fontsize= 20)
            ax1.set_xlabel("x values")
            ax1.set_ylabel("y values")
            ax1.plot(x_train, y_train, "oc", label="data points")
            ax1.plot(x_coordinate_for_line, y_coordinate_for_line, "-k", label="hypothesis")
            ax1.set_ylim(-9, 115)
            ax1.legend()
            # plt.show()

            theta_1_val = np.linspace(-1, 3, num=100)
            theta_0_val = np.linspace(-10, 10, num=100)

            THETA_0_VAL, THETA_1_VAL = np.meshgrid(theta_0_val, theta_1_val)

            Z = np.zeros((100, 100))

            for i in range(100):
                for j in range(100):
                    Z[i, j] = cost_function([THETA_0_VAL[i, j], THETA_1_VAL[i, j]], x_train, y_train, x_train.size)

            xpoint = theta_array[0]
            ypoint = theta_array[1]
            zpoint = np.zeros((1))
            zpoint = cost_function(theta_array, x_train, y_train, x_train.size)
            # 3D plot of the cost function at every iteration
            # ax = fig.add_subplot(1, 2, 2)
            # ax2 = fig.gca(projection='3d')
            # surf = ax.plot_surface(THETA_0_VAL, THETA_1_VAL, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0,antialiased=False)
            # ax.set_zlim(-1.01, 1.01)
            # ax.plot_wireframe(THETA_0_VAL, THETA_1_VAL, Z, rstride=10, cstride=10)
            # ax.plot_trisurf(THETA_0_VAL, THETA_1_VAL, Z, cmap=cm.jet, linewidth=0.2)
            ax2.plot_surface(THETA_0_VAL, THETA_1_VAL, Z, rstride=8, cstride=8, alpha=0.3)
            # ax2.set_aspect(10)
            # cset = ax.contour(THETA_0_VAL, THETA_1_VAL, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
            # cset = ax.contour(THETA_0_VAL, THETA_1_VAL, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
            # cset = ax.contour(THETA_0_VAL, THETA_1_VAL, Z, zdir='y', offset=40, cmap=cm.coolwarm)
            ax2.set_xlabel("theta 0 values")
            ax2.set_ylabel("theta 1 values")
            ax2.set_zlabel("Cost Function values")
            ax2.set_title("Cost Function ")
            ax2.tick_params(labelsize= 5)
            # print(ax2.ayim)
            ax2.view_init(azim = 10)
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax2.scatter(xpoint, ypoint, zpoint, color='r', s= 40, label="point representing thetas")
            # ax2.legend()
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()

        # changing the values of theta 0 and theta 1 according to the gradient descent method
        theta_array = improvise_thetas(theta_array, x_train, y_train, alpha, m, i)

        # storing values of the cost function after every improvisation step (gradient descent step)
        cost_function_values.append(cost_function(theta_array, x_train, y_train, m))

        # print values every 10 iterations
        if i % 2 == 0:
            print('value of theta_0 at iteration %d is: ' % i, theta_array[0])
            print('value of theta_1 at iteration %d is: ' % i, theta_array[1], '\n')

    # plot predicted values on x-axis and given values of 'y' on the y-axis
    plt.title("Predicted vs Actual results")
    plt.plot(hypothesis(theta_array, x_train), y_train, 'oc')
    plt.xlabel("Predicted value")
    plt.ylabel("Actual value")
    plt.show()

    # Plot our cost function's progress over our training period to check if the model has learnt
    iterations = np.arange(0, len(cost_function_values), step=1)
    plt.plot(iterations, cost_function_values, "-b", label="Cost Function Curve")
    plt.title("Learning Curve")
    plt.xlabel("Number Of Iterations")
    plt.ylabel("Cost Function Value")
    plt.legend()
    plt.show()

    # By returning our theta_array and saving it we are basically saving our trained model
    return theta_array


def improvise_thetas(theta_array, X, Y, alpha, m, iteration_number):
    ''' This function updates the values of theta_0 and theta_1 and returns an array containing
            the updated theta values. This is where gradient descent takes place '''

    # initializing summations to zero
    summation_0 = 0
    summation_1 = 0

    for i in range(m):        # finding the value of summations and finally the value of
        summation_0 += (theta_array[0] + theta_array[1]*X[i]) - Y[i]

        summation_1 += X[i]*((theta_array[0] + theta_array[1]*X[i])-Y[i])

    new_theta_0 = theta_array[0] - alpha * (summation_0) / m

    new_theta_1 = theta_array[1] - alpha * (summation_1) / m

    updated_theta_array = [new_theta_0, new_theta_1]

    return updated_theta_array


def testing(x_test, y_test, theta_array) :
    m = x_test.size

    # print('value of m is : ', m)
    theta_0 = theta_array[0]
    theta_1 = theta_array[1]

    # finding the range of x values across which we would like to plot our hypothesis
    lower_x_coordinate = min(x_test)
    higher_x_coordinate = max(x_test)

    # setting up the x and y coordinate values for plotting our hypothesis
    x_coordinate_for_line = np.arange(lower_x_coordinate, stop=higher_x_coordinate, step=1)
    y_coordinate_for_line = x_coordinate_for_line*theta_1 + theta_0  # y(x) = m*x +c is the final line

    plt.title("Plotting our hypothesis across the testing data")
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.plot(x_test, y_test, "oc", label="data points")
    plt.plot(x_coordinate_for_line, y_coordinate_for_line, "-k", label="hypothesis")
    plt.legend()
    plt.show()

    SSTO = []   # total sum of squares
    SSR = []    # regression sum of squares
    SSE = []    # error sum of squares
    y_mean = np.mean(y_test)

    for i in range(m):
        prediction = hypothesis(theta_array, x_test[i])  # value of prediction (value of hypothesis at point i)
        y_i = y_test[i]                             # value of 'y' at point i
        SSE.append((prediction - y_i)**2)           # sum of values in the array is SSE (error sum of squares)
        SSR.append((prediction - y_mean)**2)        # sum of values in the array is SSR (regression sum of squares)
        SSTO.append((y_i - y_mean)**2)              # sum of values in the array is SSTO (total sum of squares)

    print('\nminimum error is :', min(SSE),'\nmaximum error is : ', max(SSE),'\naverage error is : ', sum(SSE)/len(SSE))
    print('\nsum of squares of error (SSE) : ', sum(SSE))
    print('\nregression sum of squares (SSR) : ', sum(SSR))
    print('\ntotal sum of squares (SSTO) : ', sum(SSTO))
    print('\nThe Coefficient Of Determination R-squared is : ', (sum(SSR)/sum(SSTO))*100,'%')


if __name__ == "__main__" :
    # It's never a bad idea to first load your data into excel and stare at it for a while :)

    # Load the data, both for training and testing. I'm using the 'pandas' library to do this here.
    # If you are only given training data, then set aside a part of it for testing.
    raw_training_data = pd.read_csv("/home/rana/Desktop/train.csv")
    raw_testing_data = pd.read_csv("/home/rana/Desktop/test.csv")

    # Cleaning the data by removing the rows with "NaN" values in them. dropna() does the job
    cleaned_training_data = raw_training_data.dropna()
    cleaned_testing_data = raw_testing_data.dropna()
    # Note : The data cleaning step must be changed according to your data

    # Segregating the testing and training data into 'x' and 'y' and making them ready for training
    x_train_new = cleaned_training_data.as_matrix(columns=['x'])
    y_train_new = cleaned_training_data.as_matrix(columns=['y'])

    # reshaping the data from (699,1) to (699)
    x_train = x_train_new.reshape(x_train_new.size)
    y_train = y_train_new.reshape(y_train_new.size)
    # print(x_train.shape) #use this if you want to print the size of x and y

    # defining a learning rate
    alpha = 0.0001

    # Setting the number of iterations
    iters = 50

    # training time!
    theta_array = training(x_train, y_train, alpha, iters)

    print("\n*** The final value of theta_0 is ",theta_array[0]," and theta_1 is ",theta_array[1]," ***\n")

    x_test = cleaned_testing_data.as_matrix(columns=['x'])
    y_test = cleaned_testing_data.as_matrix(columns=['y'])

    testing(x_test, y_test, theta_array)