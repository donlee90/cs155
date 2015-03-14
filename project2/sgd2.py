import csv
import numpy as np

filename = 'data/data.txt'

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    l_rate = 0.02
    e_prev = 0.0

    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    P[i,:] = P[i,:] + l_rate * (2*eij*Q[:,j] - beta*P[i,:])
                    Q[:,j] = Q[:,j] + l_rate * (2*eij*P[i,:] - beta*Q[:,j])
                    
        e = 0.0
        n = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    n = n + 1
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

        e = e / n
        e_rate = abs(e - e_prev)/e_prev

        print "Step %d: " % step
        print "  l_rate = %f" % l_rate
        print "  e_prev = %f" % e_prev
        print "  e      = %f" % e
        print "  e_rate = %f" % e_rate
        print

        if e_rate < 0.0005:
            break
        e_prev = e

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
    # Initialize user/movie rating matrix
    # Row: N users
    # Col: M movies
    Y = np.zeros((num_users, num_movies))

    for column in data.T:
        user_id = column[0]
        movie_id = column[1]
        rate = column[2]

        Y[user_id-1, movie_id-1] = rate


    K = 20 # number of latent factors

    # Initialize user matrix and movie matrix randomly
    U = np.random.rand(num_users, K)
    V = np.random.rand(num_movies, K)
    a = np.random.rand(num_users)
    b = np.random.rand(num_movies)

    nU, nV = matrix_factorization(Y, U, V, K)

    out_matrix = np.concatenate((nU, nV))
    np.savetxt('out.txt', out_matrix, delimiter='\t')
    np.savetxt('U.txt', nU, delimiter='\t')
    np.savetxt('V.txt', nV, delimiter='\t')
