import numpy as np
import matplotlib.pyplot as plt
import sys

NUM_USERS = 943
NUM_MOVIES = 1682
K = 20

MOVIE_FILE = 'data/movies.txt'

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

def get_movies():
    print "Extract movie info..."
    movies = {}
    movies_idx = [''] * NUM_MOVIES
    with open(MOVIE_FILE, 'r') as f:
        lines = f.readlines()[0].split('\r')
        for line in lines:
            idx = int(line.split('\t')[0])-1
            name = line.split('\t')[1].strip('"')[:-7].split(',')[0]
            genres = map(int, line.split('\t')[2:])
            movies[name] = {'idx':idx, 'genres':genres}
            movies_idx[idx] = name
    
    return movies, movies_idx

def top_rated():
    movie_cnt = np.zeros(NUM_MOVIES)
    print movie_cnt
    with open('data/data.txt', 'r') as f:
        lines = f.readlines()[0].split('\r')
        for line in lines:
            movie_cnt[int(line.split('\t')[1])-1] += 1
    
    return np.argsort(movie_cnt)



def visualize(U, V, movies, movies_idx):
    print "Compute SVD of V..."
    V = V - np.mean(V, axis=1)[:, None]
    print 'Mean = ', np.mean(V, axis=1)
    A, s, B = np.linalg.svd(V)
    print A.shape, s.shape, B.shape
    V_tilde = np.dot(A[:,np.array([0,1])].T, V)
    V_tilde = V_tilde / np.std(V_tilde, axis=1)[:, None]
    print 'Variance = ', np.var(V_tilde, axis=1)

    #interest = np.array([movies[name]['idx'] for name in INTERESTING])
    interest = top_rated()[-40:]
    #interest = np.arange(40)

    movie_x = V_tilde[0, interest]
    movie_y = V_tilde[1, interest]
    plt.scatter(movie_x, movie_y)
    for i in interest:
        name = movies_idx[i]
        plt.annotate(name, (V_tilde[0, i], V_tilde[1, i]))
    plt.show()

 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python visualize.py [matrix_filename]"
        print "e.g. python collab_filter.py matrix.txt"
        sys.exit(1)

    matrix = sys.argv[1]
    U, V = parse(matrix)
    movies, movies_idx = get_movies()
    visualize(U, V, movies, movies_idx)
