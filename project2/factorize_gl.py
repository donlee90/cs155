import graphlab as gl
import sys
import os

NUM_USERS = 943
NUM_MOVIES = 1682
K = 20

DATA_FILE = 'data/data.txt'
DATA_SFRAME = 'data/data.sframe'

l1_lst = [1, 1e-2, 1e-4, 1e-8]
l2_lst = [1, 1e-3, 1e-5, 1e-10]

def load_data():
    data = None
    if os.path.exists(DATA_SFRAME):
        print "\n----- Loading SFrame -----"
        print " filename: ", DATA_SFRAME
        print "--------------------------\n"
        data = gl.load_sframe(DATA_SFRAME)
    else:
        print "\n----- Creating SFrame -----"
        print " filename: ", DATA_FILE
        print "---------------------------\n"
        raw_data = {'user_id': [], 'item_id': [], 'rating': []}
        with open(DATA_FILE, 'r') as f:
            lines = f.readlines()[0].split('\r')
            for line in lines:
                raw_data['user_id'].append(int(line.split('\t')[0]))
                raw_data['item_id'].append(int(line.split('\t')[1]))
                raw_data['rating'].append(int(line.split('\t')[2]))
        data = gl.SFrame(raw_data)
        print "\n----- Saving SFrame -----"
        print " filename: ", DATA_SFRAME
        print "------------------------\n-"
        data.save(DATA_SFRAME)

    return data
 
def train(data, l1, l2, filename):
    print "\n----- Training -----"
    m = gl.recommender.factorization_recommender.create(
            data, target='rating', num_factors=K, 
            regularization=l1, linear_regularization=l2,
            max_iterations=100000, solver='sgd')
    print "--------------------\n"

    # Evaluation
    # Save the model
    modelname = filename.split('.')[0] + ".model"
    print "\n----- Saving model -----"
    print " filename: ", modelname
    print "------------------------\n"
    m.save(modelname)

    return m

def cross_validate(data, filename):
    print "\n----- Cross Validating -----"
    result_total = {}
    for l1 in l1_lst:
        for l2 in l2_lst:
            result = []
            for i in xrange(5):
                train, test = data.random_split(0.8, seed=1)
                m = gl.recommender.factorization_recommender.create(
                        train, target='rating', num_factors=K, 
                        regularization=l1, linear_regularization=l2,
                        max_iterations=100000, solver='sgd')
                result.append(m.evaluate_rmse(test, target='rating')['rmse_overall'])

            # Evaluation
            result_total[(l1, l2)] = result
            print "\n----- Result ------"
            print "l1 = %f, l2 = %f rmse = %f\n" % (l1, l2, sum(result_total[(l1,l2)]) / 5)
            print "-------------------\n"
    print "----------------------------\n"

    # Print result
    resultname = filename.split('.')[0] + ".cross"
    print "\n----- Saving result -----"
    print " filename: ", resultname
    with open(resultname, 'w') as f:
        for l1 in l1_lst:
            for l2 in l2_lst:
                f.write("l1 = %f, l2 = %f rmse = %f\n" % (l1, l2, sum(result_total[(l1,l2)]) / 5))
    print "------------------------\n"

def export(m, filename):
    # Print summary
    print "\n----- Summary -----"
    m.summary()
    print m['coefficients']
    print "-------------------\n"

    print "\n----- Sorting U, V -----"
    U_sf = m['coefficients']['user_id'].sort('user_id')
    V_sf = m['coefficients']['item_id'].sort('item_id')
    print U_sf
    print V_sf
    print "------------------------\n"

    print "\n----- Writing -----"
    print " filename: ", filename
    with open(filename, 'w') as f:
        #for row in U_lst:
        for row in U_sf['factors']:
            f.write('\t'.join(map(str, row)))
            f.write('\n')
        #for row in V_lst:
        for row in V_sf['factors']:
            f.write('\t'.join(map(str, row)))
            f.write('\n')
    print "-------------------\n"

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage: python factorize.py [output] [lambda1] [labmda2] [optional:model]"
        print "e.g. python factorize.py out.txt 1e-8 1e-8 [out.model]"
        sys.exit(1)

    out_file = sys.argv[1]
    l1 = float(sys.argv[2])
    l2 = float(sys.argv[3])
    print l1, l2
    if len(sys.argv) > 4:
        model_file = sys.argv[4]
        m = gl.load_model(model_file)
    else:
        data = load_data()
        m = train(data, l1, l2, out_file)
        #cross_validate(data, out_file)

    export(m, out_file)

