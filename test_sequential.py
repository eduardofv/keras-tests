import numpy as np
from scipy import spatial
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation

#Setup
#np.random.seed(1234)
np.set_printoptions(precision=3)
ADD_RANDOM_NOISE = False
RANDOM_NOISE_SCALE = 0.05

def tf_canberra(a,b):
    num = np.abs(np.subtract(a,b)) / np.add(np.abs(a),np.abs(b))
    dist = np.sum(num)
    return dist

#def canberra(a,b):
#    dist=0
#    for i in range(len(a)):
#        num = np.abs(a[i]-b[i])
#        den = np.abs(a[i])+np.abs(b[i])
#        dist += num/den
#    return dist

# Linear combinatios: 
#  dat: data
#  idx: indices of the data to be combined
def linear_combination(dat,idx):
    return sum( np.array([ dat[i] for i in idx ]))

# Creates random permutation of the input_dimension indices for each output_dimension
#  IOW: selects some of the input dimensions to be combined for each output
def shuffle_indexes(input_dim,output_dim):
    idx = []
    for i in range(output_dim):
        indexes = range(input_dim)
        np.random.shuffle(indexes)
        idx.append(indexes[0:3])
    return idx

def create_vectors(data,fun):
    labels = np.zeros_like(data,dtype=np.float64)
    for i,dat in enumerate(data):
        labels[i] = fun(dat)
    if ADD_RANDOM_NOISE:
        labels = np.add(labels,np.random.normal(
            scale = RANDOM_NOISE_SCALE,
            size = np.shape(labels)))
    return labels


def linear(data,input_dim,output_dim):
    return create_vectors(
            data,
            lambda dat:  (dat[0]+dat[1],dat[1]+dat[2],dat[0]+dat[1]+dat[2]))

def quad_linear(data,input_dim,output_dim):
    return create_vectors(
            data,
            lambda dat: (dat[0]*dat[1],dat[1]*dat[2],dat[0]+dat[1]+dat[2]))

def quad(data,input_dim,output_dim):
    return create_vectors(
            data,
            lambda dat: (dat[0]*dat[1],dat[1]*dat[2],dat[0]*dat[1]*dat[2]))

def linearvec(data,input_dim,output_dim):
    indexes = shuffle_indexes(input_dim,output_dim)
    labels = np.zeros((len(data),output_dim))
    for i,d in enumerate(data):
        for x in range(output_dim):
            labels[i][x] = linear_combination(d,indexes[x])
    return labels

#Params
function_to_train = linearvec
input_dim = 3
output_dim = 3
dense_layer_sizes = [30,30,30,30]
optimizer = 'rmsprop'
loss = 'mse'#lambda x,y: tf_canberra(x,y)#'mse'#'cosine_proximity'
training_set_size = 10000
test_set_size = 100
epochs_to_train = 50

#Process
sess = tf.Session()
K.set_session(sess)

model = Sequential()
model.add( Dense(dense_layer_sizes[0], input_shape=(input_dim,)) )
model.add( Activation('relu') )

layers = len(dense_layer_sizes)
for i in range(1,layers):
    print("Adding dense layer %d size %d"%(i,dense_layer_sizes[i]))
    model.add( Dense(dense_layer_sizes[i]) )
    if i < layers:
        print("Adding activation: relu")
        model.add( Activation('relu') )

print("Adding output layer")
model.add( Dense(output_dim) )
    
model.compile(optimizer=optimizer,
              loss=loss)

train_data = np.random.random((training_set_size,input_dim))
train_labels = function_to_train(train_data,input_dim,output_dim)

test_data = np.random.random((test_set_size,input_dim))
test_labels = function_to_train(test_data,input_dim,output_dim)

model.fit(train_data, train_labels, epochs=epochs_to_train)
score = model.evaluate(test_data,test_labels)
print("\nScore: %f"%score)

test_predicted = model.predict(test_data)
dist_sum = 0
for i,dat in enumerate(test_data):
    #dist = spatial.distance.euclidean(test_labels[i],test_predicted[i])
    #dist = spatial.distance.cosine(test_labels[i],test_predicted[i])
    dist = tf_canberra(test_labels[i],test_predicted[i])
    dist_sum += dist
    print("%s\t%s\t%s\t%.04f" % (
        str(dat),
        str(test_labels[i]),
        str(test_predicted[i]),
        dist))

print("Avg. distance: %.04f"%(dist_sum/len(test_data)))

