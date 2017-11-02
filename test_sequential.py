import numpy as np
from scipy import spatial
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation

#Setup
np.random.seed(1234)
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

def create_vectors(data,fun):
    labels = np.zeros_like(data,dtype=np.float64)
    for i,dat in enumerate(data):
        labels[i] = fun(dat)
    if ADD_RANDOM_NOISE:
        labels = np.add(labels,np.random.normal(
            scale = RANDOM_NOISE_SCALE,
            size = np.shape(labels)))
    return labels

#def linearvec(data,input_dim,output_dim):


def linear(data):
    return create_vectors(
            data,
            lambda dat:  (dat[0]+dat[1],dat[1]+dat[2],dat[0]+dat[1]+dat[2]))

def quad_linear(data):
    return create_vectors(
            data,
            lambda dat: (dat[0]*dat[1],dat[1]*dat[2],dat[0]+dat[1]+dat[2]))

def quad(data):
    return create_vectors(
            data,
            lambda dat: (dat[0]*dat[1],dat[1]*dat[2],dat[0]*dat[1]*dat[2]))

#Params
input_dim = 3
output_dim = 3
dense_layer_size = 30
optimizer = 'rmsprop'
loss = lambda x,y: tf_canberra(x,y)#'mse'#'cosine_proximity'#'mse'
function_to_train = quad
training_set_size = 10000
test_set_size = 100
epochs_to_train = 150

#Process
sess = tf.Session()
K.set_session(sess)

model = Sequential([
    Dense(dense_layer_size, input_shape=(input_dim,)),
    Activation('relu'),
    Dense(output_dim)])
    
model.compile(optimizer=optimizer,
              loss=loss)

train_data = np.random.random((training_set_size,input_dim))
train_labels = function_to_train(train_data)

test_data = np.random.random((test_set_size,input_dim))
test_labels = function_to_train(test_data)

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

