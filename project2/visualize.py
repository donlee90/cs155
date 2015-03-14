import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt
import sys

NUM_USERS = 943
NUM_MOVIES = 1682

def toNpArray(arr):
    return np.array([[elem] for elem in arr])

def parse(matrix):
    print "Extracting into numpy arrays..."

    with open(matrix, 'r') as f:
        lines = map(lambda x: map(lambda y: float(y), x.strip().split('\t')), 
                filter(lambda x: x, f.readlines()))
        if len(lines) != NUM_USERS + NUM_MOVIES:
            print "Number of lines does not match"
            sys.exit(1)

    U = np.array(lines[:NUM_USERS]).T
    V = np.array(lines[NUM_USERS:]).T
    print U.shape, V.shape
    
    print np.dot(U[0,:].T, V[0,:])

    return U, V


def visualize(U, V, data, movie):
    print "Compute SVD of V..."
    A, s, B = np.linalg.svd(V)
    V_tilde = np.dot(A[:,:2].T, V)
    U_tilde = np.dot(A[:,:2].T, U)

    user_id = []
    movie_id = []
    with open(data, 'r') as f:
        lines = f.readlines()[0].split('\r')
        for line in lines:
            user_id.append(line.split('\t')[0])
            movie_id.append(line.split('\t')[1])

    movies = []
    with open(movie, 'r') as f:
        lines = f.readlines()[0].split('\r')
        for line in lines:
            name = line.split('\t')[1]
            genres = line.split('\t')[2:]
            movies.append({'name': name, 'genres': genres})


    #user_x = U_tilde[0,:10]
    #user_y = U_tilde[1,:10]
    movie_x = V_tilde[0,:10] * (10 ** 10)
    movie_y = V_tilde[1,:10] * (10 ** 10)
    print movie_x
    print movie_y
    plt.scatter(movie_x, movie_y)
    for i in xrange(10):
        plt.annotate(movies[i]['name'], (movie_x[i],movie_y[i]))
    plt.show()

 
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage: python visualize.py [matrix_filename] [data_filename] [movie_filename]"
        print "e.g. python collab_filter.py data.matrix data.txt movie.txt"
        sys.exit(1)

    matrix = sys.argv[1]
    data = sys.argv[2]
    movie = sys.argv[3]
    U, V = parse(matrix)
    #visualize(U, V, data, movie)
