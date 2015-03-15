import csv
import numpy as np
import random
import math

FILENAME = 'data/data.txt'
NUM_DATA = 100000 # Number of data points
K = 20 # Number of latent factors
g_mean = 0

def RMSE(R, P, Q):

    rmse = 0.0
    count = 0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] > 0:
                rmse += pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                count += 1

    rmse = rmse / count

    return math.sqrt(rmse)

def matrix_factorization(R, P, Q, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    S = []
    l_rate = 0.02
    rmse_prev = 0.0

    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] > 0:
                S.append((i,j))

    print "Running SGD..."
    print "|{:^15s}|{:^15s}|{:^15s}|{:^15s}|".format("Step", "RMSE", "Delta", "L Rate")
    print "%s" % '-'*65

    # Stochastic gradient descent
    for step in xrange(steps):

        random.shuffle(S) # create a random permutation

        for i,j in S:
            pi = P[i,:]
            qj = Q[:,j]
            eij = R[i][j] - np.dot(pi, qj)

            P[i,:] = pi + l_rate * (2*eij*qj - beta*pi)
            Q[:,j] = qj + l_rate * (2*eij*pi - beta*qj)
            
        # Calculate root mean squared error
        rmse = RMSE(R, P, Q)
        delta = abs(rmse - rmse_prev)

        print "|{:>15d}|{:>15.5f}|{:>15.5f}|{:>15.5f}|".format(step, rmse, delta, l_rate)

        # Convergence Test
        if delta < 0.0001:
            break

        rmse_prev = rmse

        # Update learning rate
        if l_rate > alpha:
            l_rate = l_rate * 0.9

    return P, Q.T

def parse_data(filename):

    with open(filename, 'rU') as data_file:
        data = np.array(zip(*[line.strip().split('\t')
                    for line in data_file])).astype(int)

    N = len(set(data[0]))
    M = len(set(data[1]))

    Y = np.zeros((N, M))

    for data_point in data.T:
        user_id = data_point[0]
        movie_id = data_point[1]
        rate = data_point[2]

        Y[user_id-1, movie_id-1] = rate

    return Y


if __name__ == "__main__":

    # Initialize user/movie rating matrix

    print "Loading data..."

    # Row: users
    # Col: movies
    Y = parse_data(FILENAME)

    sum_rate = 0.
    for i in xrange(len(Y)):
        for j in xrange(len(Y[i])):
            if Y[i,j] > 0:
                sum_rate += Y[i,j]

    g_mean = sum_rate / NUM_DATA
    num_users = len(Y)
    num_movies = len(Y.T)

    print "  num_users : %d" % num_users
    print "  num_movies: %d" % num_movies
    print "  global_avg: %f" % g_mean
    print


    # Initialize user matrix and movie matrix randomly

    print "Initializing..."

    U = np.random.rand(num_users, K)
    V = np.random.rand(num_movies, K)

    nU, nV = matrix_factorization(Y, U, V)

    out_matrix = np.concatenate((nU, nV))
    np.savetxt('out.txt', out_matrix, delimiter='\t')
