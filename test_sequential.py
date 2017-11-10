import math
import numpy as np
from scipy import spatial
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation

#Setup
#np.random.seed(1234)
np.set_printoptions(precision=3)
USE_WEIGHTS = False
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

# Creates random permutation of the input_dimension indices for each output_dimension
#  IOW: selects some of the input dimensions to be combined for each output
def shuffle_indexes(input_dim,output_dim):
    idx = []
    max_dims = int(math.floor(input_dim/2)+1) #proporcion de dimensiones a usar en la linearizacion
    for i in range(output_dim):
        indexes = range(input_dim)
        np.random.shuffle(indexes)
        idx.append(indexes[0:max_dims])
    return idx

def create_vectors(data,indexes,weights,fun):
    rows = np.shape(data)[0]
    cols = len(indexes)
    labels = np.zeros((rows,cols))
    for i,dat in enumerate(data):
        labels[i] = fun(dat,indexes,weights)
    if ADD_RANDOM_NOISE:
        labels = np.add(labels,np.random.normal(
            scale = RANDOM_NOISE_SCALE,
            size = np.shape(labels)))
    return labels

# Linear combination: 
#  dat: data
#  idx: indices of the data to be combined
def linear_combination(dat,idx,weights):
    return sum( [ weights[i]*dat[idx[i]] for i in range(len(idx)) ] )

#Functions that can be used to generate the training set
def linear(data,indexes,weights):
    return create_vectors(
            data,
            indexes,
            weights,
            lambda dat,idx,wei: [linear_combination(dat,idx[i],wei[i]) for i in range(len(idx))] )

#  the 3x3 are always 3 inputs by 3 outputs
def linear3x3(data,indexes):
    return create_vectors(
            data,
            indexes,
            weights,
            lambda dat,indexes:  (dat[0]+dat[1],dat[1]+dat[2],dat[0]+dat[1]+dat[2]))

def quad_linear3x3(data,indexes):
    return create_vectors(
            data,
            indexes,
            weights,
            lambda dat,indexes: (dat[0]*dat[1],dat[1]*dat[2],dat[0]+dat[1]+dat[2]))

def quad3x3(data,indexes):
    return create_vectors(
            data,
            indexes,
            weights,
            lambda dat,indexes: (dat[0]*dat[1],dat[1]*dat[2],dat[0]*dat[1]*dat[2]))


#Params
function_to_train = linear
input_dim = 3
output_dim = 3
dense_layer_sizes = [30]
optimizer = 'rmsprop'
loss = 'mse'#lambda x,y: tf_canberra(x,y)#'mse'#'cosine_proximity'
training_set_size = 10000
test_set_size = 100
epochs_to_train = 20

#Process

#  Create the Model
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

#  Generate functions
indexes = shuffle_indexes(input_dim,output_dim)
if USE_WEIGHTS:
    weights = np.random.rand(output_dim,input_dim) #en realidad solo se usa una fraccion de input_dim
else:
    weights = np.ones((output_dim,input_dim))

print("Linear Indexes:")
print(indexes)
print("Linear Weights:")
print(weights)
 
train_data = np.random.random((training_set_size,input_dim))
train_labels = function_to_train(train_data,indexes,weights)

test_data = np.random.random((test_set_size,input_dim))
test_labels = function_to_train(test_data,indexes,weights)

model.fit(train_data, train_labels, epochs=epochs_to_train)
score = model.evaluate(test_data,test_labels)
print("\nScore: %f"%score)

test_predicted = model.predict(test_data)
dist_sum = 0
for i,dat in enumerate(test_data):
    dist = spatial.distance.euclidean(test_labels[i],test_predicted[i])
    #dist = spatial.distance.cosine(test_labels[i],test_predicted[i])
    #dist = tf_canberra(test_labels[i],test_predicted[i])
    dist_sum += dist
    print("%s\t%s\t%s\t%.04f" % (
        str(dat),
        str(test_labels[i]),
        str(test_predicted[i]),
        dist))

print("Avg. distance: %.04f"%(dist_sum/len(test_data)))

