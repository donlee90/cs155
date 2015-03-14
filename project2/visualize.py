import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt
import sys

NUM_USERS = 943
NUM_MOVIES = 1682
K = 20

INTERESTING = ['My Fair Lady', 'Clockwork Orange', 'Free Willy 2: The Adventure Home', 'Free Willy', 'Batman Forever', 'Bad Boys',
        'Birdcage', 'Nutty Professor', 'GoldenEye', 'Apollo 13', 'Twelve Monkeys', '2001: A Space Odyssey',
        'Fargo', 'Jurassic Park', 'Forrest Gump', 'Braveheart', 'Seven (Se7en)', 'Aliens', 'Return of the Jedi', 
        'Star Wars', 'Empire Strikes Back']

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
    print "U is ", U.shape
    print "V is ", V.shape

    return U, V

def get_movies(movie):
    print "Extract movie info..."
    movies = {}
    with open(movie, 'r') as f:
        lines = f.readlines()[0].split('\r')
        for line in lines:
            idx = int(line.split('\t')[0])
            name = line.split('\t')[1].strip('"')[:-7].split(',')[0]
            genres = [int(i) for i in line.split('\t')[2]]
            movies[name] = {'idx':idx, 'genres':genres}
    
    return movies


def visualize(U, V, movies):
    print "Compute SVD of V..."
    V = V - np.mean(V, axis=1)[:, None]
    #print np.mean(V, axis=1)
    A, s, B = np.linalg.svd(V)
    #print A.shape, s.shape, B.shape
    V_tilde = np.dot(A[:,:2].T, V)
    V_tilde = V_tilde / np.std(V_tilde, axis=1)[:, None]
    print np.var(V_tilde)

    interest = np.array([movies[name]['idx'] for name in INTERESTING])
    #print interest
    movie_x = V_tilde[0, interest]
    movie_y = V_tilde[1, interest]
    plt.scatter(movie_x, movie_y)
    for name in INTERESTING:
        i = movies[name]['idx']
        plt.annotate(name, (V_tilde[0, i], V_tilde[1, i]))
    plt.show()

 
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: python visualize.py [matrix_filename] [movie_filename]"
        print "e.g. python collab_filter.py data.matrix movie.txt"
        sys.exit(1)

    matrix = sys.argv[1]
    movie = sys.argv[2]
    U, V = parse(matrix)
    movies = get_movies(movie)
    visualize(U, V, movies)
