import csv
import numpy as np

filename = 'data.txt'

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        print "step %d: " % step,
        print "e = %d" % e
        if e < 0.001:
            break
    return P, Q.T

if __name__ == "__main__":
    with open(filename, 'rU') as data_file:
        data = np.array(zip(*[line.strip().split('\t')
                    for line in data_file])).astype(int)

    num_users = len(set(data[0]))
    num_movies = len(set(data[1]))

    print "num_users: %d" % num_users
    print "num_movies: %d" % num_movies

    # Initialize user/movie rating matrix
    # Row: N users
    # Col: M movies
    Y = np.zeros((num_users, num_movies))

    for column in data.T:
        user_id = column[0]
        movie_id = column[1]
        rate = column[2]

        Y[user_id-1, movie_id-1] = rate

    print Y

    K = 20 # number of latent factors

    # Initialize user matrix and movie matrix randomly
    U = np.random.rand(num_users, K)
    V = np.random.rand(num_movies, K)

    nU, nV = matrix_factorization(Y, U, V, K)
