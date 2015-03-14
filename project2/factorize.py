import graphlab as gl
import sys
import os

NUM_USERS = 943
NUM_MOVIES = 1682

def load_data(filename):
    data_file = filename.replace(".txt", ".sframe")
    data = None
    if os.path.exists(data_file):
        print "Loading data..."
        data = gl.load_sframe(data_file)
    else:
        print "Creating data..."
        raw_data = {'user_id': [], 'item_id': [], 'rating': []}
        with open(filename, 'r') as f:
            lines = f.readlines()[0].split('\r')
            for line in lines:
                raw_data['user_id'].append(line.split('\t')[0])
                raw_data['item_id'].append(line.split('\t')[1])
                raw_data['rating'].append(int(line.split('\t')[2]))
        data = gl.SFrame(raw_data)
        data.save(model)
    return data
 
def factorize(data):
    print "Factorizing..."
    m = gl.recommender.factorization_recommender.create(
            data, target='rating', num_factors=20, 
            regularization=10, max_iterations=100000,
            solver='sgd')

    # Print the summary
    m.summary()
    print m.get("coefficients")

    # Save the model
    modelname = filename.replace(".txt", ".model")
    m.save(modelname)

    return m

def export(m, filename):
    matrixname = filename.replace(".txt", ".matrix")
    with open(matrixname, 'w') as f:
        U= m['coefficients']['user_id']['factors']
        V= m['coefficients']['item_id']['factors']
        for row in U:
            for elem in row:
                f.write(str(elem) + '\t')
            f.write('\n')
        for row in V:
            for elem in row:
                f.write(str(elem) + '\t')
            f.write('\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python factorize.py [data_filename]"
        print "e.g. python collab_filter.py data.txt"
        sys.exit(1)

    filename = sys.argv[1]
    data = load_data(filename)
    m = factorize(data)
    #m = gl.load_model(filename.replace(".txt",".model"))
    export(m, filename)

