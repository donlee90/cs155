import csv
import numpy as np

filename = 'data/data.txt'
N = 100000 # Number of data points
K = 20 # Number of latent factors

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    l_rate = 0.02
    mse_prev = 0.0

    print "Running SGD..."
    print "|{:^15s}|{:^15s}|{:^15s}|{:^15s}|".format("Step", "MSE", "Delta", "L Rate")
    print "%s" % '-'*65

    for step in xrange(steps):

        # Stochastic gradient descent
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    pi = P[i,:]
                    qj = Q[:,j]
                    eij = R[i][j] - np.dot(pi, qj)

                    P[i,:] = pi + l_rate * (2*eij*qj - beta*pi)
                    Q[:,j] = qj + l_rate * (2*eij*pi - beta*qj)
                    
        # Calculate mean squared error
        mse = 0.0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    mse = mse + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

        mse = mse / N
        delta = abs(mse - mse_prev)

        print "|{:>15d}|{:>15.5f}|{:>15.5f}|{:>15.5f}|".format(step, mse, delta, l_rate)

        # Convergence Test
        if delta < 0.0001:
            break

        mse_prev = mse

        # Update learning rate
        if l_rate > alpha:
            l_rate = l_rate * 0.9

    return P, Q.T


if __name__ == "__main__":

    print "Loading data..."
    with open(filename, 'rU') as data_file:
        data = np.array(zip(*[line.strip().split('\t')
                    for line in data_file])).astype(int)

    num_users = len(set(data[0]))
    num_movies = len(set(data[1]))

    print "  num_users: %d" % num_users
    print "  num_movies: %d" % num_movies
    print

    print "Initializing..."
    print

    # Initialize user/movie rating matrix
    # Row: users
    # Col: movies
    Y = np.zeros((num_users, num_movies))

    for column in data.T:
        user_id = column[0]
        movie_id = column[1]
        rate = column[2]

        Y[user_id-1, movie_id-1] = rate


    # Initialize user matrix and movie matrix randomly
    U = np.random.rand(num_users, K)
    V = np.random.rand(num_movies, K)

    nU, nV = matrix_factorization(Y, U, V, K)

    out_matrix = np.concatenate((nU, nV))
    np.savetxt('out.txt', out_matrix, delimiter='\t')
