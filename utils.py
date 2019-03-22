import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y, model):
    # plot the data points X and y

    # find indices of positive and negative examples
    pos = np.where(y == 1)
    neg = np.where(y == -1)

    # plot examples
    plt.plot(X[pos, 0], X[pos, 1], 'r+', markersize=20)
    plt.plot(X[neg, 0], X[neg, 1], 'm^', markersize=20, mfc='none')

    # add circle around support vectors
    plt.plot(model.support_vectors[:, 0], model.support_vectors[:, 1], 'bo', markersize=40, mfc='none')


def visualizeBoundary(X, y, model):
    # plot data
    plotData(X, y, model)

    # make classification predictions over a grid of values
    n_grid = 60
    x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), n_grid)
    x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), n_grid)

    X1, X2 = np.meshgrid(x1plot, x2plot)
    X1_tmp = np.reshape(X1, [X1.size, 1])
    X2_tmp = np.reshape(X2, [X2.size, 1])
    vals = model.predict(np.concatenate((X1_tmp, X2_tmp), axis=1))
    vals = np.reshape(vals, X1.shape)

    vals[vals == -1] = 0
    # plot the svm boundary
    plt.contour(X1, X2, vals, 1, colors='b')
    plt.show()


def visualizeBoundaryLinear(X, y, model):
    # calculate w and b

    alpha = np.expand_dims(model.lagr_multipliers, axis=1)  # N*1
    support_vectors = model.support_vectors  # N*K
    labels = np.expand_dims(model.support_vector_labels, axis=1)  # N*1
    w = np.sum(alpha * labels * support_vectors, axis=0)

    b = model.intercept

    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = - (w[0] * xp + b) / w[1]

    # plot data
    plotData(X, y, model)

    # plot decision boundary
    plt.plot(xp, yp, '-b')

    # plot margin boundary
    yp = - (w[0] * xp + (b + 1)) / w[1]
    plt.plot(xp, yp, '--b')
    yp = - (w[0] * xp + (b - 1)) / w[1]
    plt.plot(xp, yp, '--b')
    plt.show()
