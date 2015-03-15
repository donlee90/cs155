import graphlab as gl
import sys
import os

NUM_USERS = 943
NUM_MOVIES = 1682
K = 20

DATA_FILE = 'data/data.txt'
DATA_SFRAME = 'data/data.sframe'

def load_data():
    data = None
    if os.path.exists(DATA_SFRAME):
        print "Loading data..."
        data = gl.load_sframe(data_file)
    else:
        print "Creating data..."
        raw_data = {'user_id': [], 'item_id': [], 'rating': []}
        with open(DATA_FILE, 'r') as f:
            lines = f.readlines()[0].split('\r')
            for line in lines:
                raw_data['user_id'].append(line.split('\t')[0])
                raw_data['item_id'].append(line.split('\t')[1])
                raw_data['rating'].append(int(line.split('\t')[2]))
        data = gl.SFrame(raw_data)
        data.save(DATA_SFRAME)
    return data
 
def factorize(data, filename):
    print "Factorizing..."
    m = gl.recommender.factorization_recommender.create(
            data, target='rating', num_factors=K, 
            #regularization=10, linear_regularization=10,
            max_iterations=100000, solver='sgd')

    # Save the model
    modelname = filename.replace(".txt", ".model")
    m.save(modelname)

    return m

def export(m, filename):
    print "Exporting..."
    m.summary()
    print m['coefficients']
    print "Sorting..."
    U_lst = [None] * NUM_USERS
    V_lst = [None] * NUM_MOVIES
    U_idx = list(m['coefficients']['user_id']['user_id'])
    V_idx = list(m['coefficients']['item_id']['item_id'])
    for i, user in enumerate(U_idx):
        U_lst[int(user)-1] = map(str, m['coefficients']['user_id']['factors'][i])
    for i, movie in enumerate(V_idx):
        V_lst[int(movie)-1] = map(str, m['coefficients']['item_id']['factors'][i])

    print "Writing..."
    with open(filename, 'w') as f:
        for row in U_lst:
            f.write('\t'.join(row))
            f.write('\n')
        for row in V_lst:
            f.write('\t'.join(row))
            f.write('\n')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: python factorize.py [data] [output] [optional:model]"
        print "e.g. python collab_filter.py data.txt out.txt model.txt"
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    if len(sys.argv) > 3:
        model_file = sys.argv[3]
        m = gl.load_model(model_file)
    else:
        data = load_data(in_file)
        m = factorize(data, out_file)

    export(m, out_file)

