
##

import os

from tempfile import gettempdir
import urllib
import zipfile
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)

##

# build a naive graph
a_tensor = tf.constant(3., name="const3")
b_tensor = tf.constant(4., name="const4")
out_tensor = tf.add(a_tensor, b_tensor)
print(a_tensor, b_tensor, out_tensor, sep="\n")
##

sess=tf.Session()
a,b,c=sess.run([a_tensor,b_tensor,out_tensor])
print("a = {} \nb = {} \nc = {}".format(a, b, c))

##
# constant of 1d tensor (vector)
a = tf.constant([2, 2], dtype=tf.int32, name="vector")
# constant of 2x2 tensor (matrix)
b = tf.constant([[0, 1], [2, 3]], name="matrix")
print(a, b, sep="\n")

c=tf.constant([[[1,2,3],[4,5,6]]],name='matrix')
print(c)



##

# a matrix with filled zeros
c = tf.zeros([2, 3], tf.int32, name="zeros_matrix") # [[0, 0, 0], [0, 0, 0]]
# a matrix with filled ones
d = tf.ones([2, 3], tf.int32, name="ones_matrix") #  [[1, 1, 1], [1, 1, 1]]

# create a tensor filled zeros/ones, with shape and type as input_tensor
input_tensor = tf.constant([[1,1], [2,2], [3,3]], dtype=tf.float32)
e = tf.zeros_like(input_tensor, name="zeros_like_matrix")  #  [[0, 0], [0, 0], [0, 0]]
f = tf.ones_like(input_tensor, name="ones_like_matrix") # [[1, 1], [1, 1], [1, 1]]

print(c, d, e, f, sep="\n")
##

# create a variable of vector
vec_var = tf.get_variable(name="vector", shape=[3],
                          initializer=tf.ones_initializer)
# create a variable of matrix
mat_var = tf.get_variable(name="matrix", shape=[5, 3],
                          initializer=tf.random_normal_initializer)

print(vec_var, mat_var, sep="\n")

my_vec=tf.get_variable(name='my_vect',shape=[5],initializer=tf.ones_initializer)
print(my_vec)


##

# instance of `tf.Variable`
var = tf.Variable(2, name="scalar")

# we can assign new value to a variable
var_times_two = var.assign(var * 2) # an operation that assigns value var*2 to var
print(var, var_times_two, sep="\n")


##
# constant value is not changable
# the following code will casue error
c = tf.constant(0.)
c.assign(1.)

##

variable_init_op = tf.global_variables_initializer() # an operation
sess.run(variable_init_op)



##
a_placeholder = tf.placeholder(tf.float32, shape=[None, 3], name="a")
print(a_placeholder)

##

sess = tf.Session()
a = sess.run(a_placeholder, feed_dict={a_placeholder: [[1, 2, 3], [4, 5, 6]]})
print(a)
##


# Build a basic graph that demos the basic tensorflow concepts
with tf.Graph().as_default() as g:
    # a constant tensor with rank = 0
    scalar_tensor = tf.constant(5., name="scalar")

    # a vector tensor with rank = 1, and filled with random values
    vector_tensor = tf.random_normal(shape=[5], name="vector")

    # tensorflow supports broadcast
    broadcast_with_scalar = vector_tensor + scalar_tensor

    # use placeholder to get values in runtime
    x_input = tf.placeholder(tf.float32, shape=[None, 5], name="input")
    feature_dims = x_input.shape[1]

    # a matrix variable with rank = 2.
    matrix_variable = tf.get_variable("matrix",
                                      shape=[feature_dims, 2],
                                      initializer=tf.ones_initializer)
    mul_with_matrix = tf.matmul(x_input, matrix_variable, name="output")

    var_init_op = tf.global_variables_initializer()

##
feed_data = np.random.randint(5, size=[5, 5])

# Under the scope of session, we can run the value of tensor in default graph
with tf.Session(graph=g) as sess:
    sess.run(var_init_op)  # must initialize the variables

    scalar, vector, broadcast = sess.run([scalar_tensor,
                                          vector_tensor,
                                          broadcast_with_scalar])
    print("[scalar]\n {} \n[vector]\n {} \n[broadcast]\n {}".format(scalar,
                                                                    vector,
                                                                    broadcast))

    x, m, out = sess.run([x_input, matrix_variable, mul_with_matrix],
                         feed_dict={x_input: feed_data})
    print("[input]\n {} \n[matrix]\n {} \n[output]\n {}".format(x, m, out))

##

graph_dir = "graphs/demo"
os.makedirs(graph_dir)
with tf.Graph().as_default() as g:
    const_a = tf.constant(1., shape=[1, 5], name="const_a")
    const_b = tf.add(const_a, 5, name="const_b")
    var_c = tf.get_variable("var_c", shape=[5, 3])
    const_d = tf.matmul(const_b, var_c, name="const_d")
    # create a writer
    writer = tf.summary.FileWriter(graph_dir, tf.get_default_graph())

##

ckpt_dir = "checkpoints/demo"
os.makedirs(ckpt_dir)
with tf.Graph().as_default() as g:
    const_a = tf.constant(2, tf.int32, [5])
    var_b = tf.get_variable("var_b", dtype=tf.int32, shape=[5],
                            initializer=tf.zeros_initializer)  # variable
    const_c = var_b + const_a
    var_d = tf.get_variable("var_d", shape=[3],
                            initializer=tf.ones_initializer)  # variable

    print("[Graph]", const_a, var_b, const_c, var_d, sep="\n")
    print("\n[Trainable variables]", *tf.trainable_variables(), sep="\n")

    init_op = tf.global_variables_initializer()
    # Declare a saver object to save checkpoints
    saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    # Initialize variables
    sess.run(init_op)

    # Do some works with the model
    a, b, c, d = sess.run([const_a, var_b, const_c, var_d])
    print("\n[Value]", a, b, c, d, sep="\n")

    # Save the variables to disk
    save_path = saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"))
    print("\n[Model saved in path: {}]".format(save_path))
##

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# tensor_name: Name of the tensor in the checkpoint file to print.
# all_tensors: Boolean indicating whether to print all tensors.
# all_tensor_names: Boolean indicating whether to print all tensor names.
print_tensors_in_checkpoint_file(save_path,
                                 tensor_name="",
                                 all_tensors="",
                                 all_tensor_names=False)
##

with tf.Session(graph=g) as sess:
    saver.restore(sess,save_path)
    a, b, c, d = sess.run([const_a, var_b, const_c, var_d])
    print("\n[Value]", a, b, c, d, sep="\n")

##
with tf.Session(graph=g) as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    a, b, c, d = sess.run([const_a, var_b, const_c, var_d])
    print("\n[Value]", a, b, c, d, sep="\n")

##
tf.reset_default_graph()
# Pseudo dataset
dataset_size = 20
# 20 examples, each example has 5 features
data = np.random.rand(dataset_size, 5)
# this dataset has 3 labels
label = np.random.randint(low=0, high=3, size=dataset_size)

##

batch_size = 7

# Create a `dataset` by `from_tensor_slices`
training_dataset = tf.data.Dataset.from_tensor_slices((data, label))
print("[Original dataset] \n", training_dataset)
training_dataset = training_dataset.batch(batch_size)
print("\n[Transformed dataset] \n", training_dataset)

# Create a `iterator` to extract elements from `dataset`
training_iterator = training_dataset.make_initializable_iterator()
x_input, y_label = training_iterator.get_next()
print("\n[Iterator] \n", training_iterator)
print("\n[Elements extracted by iterator] \n", x_input, "\n", y_label)
##
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # avoids occupying full memory of GPU
with tf.Session(config=config) as sess:
    sess.run(training_iterator.initializer) # initialize the iterator
    step = 0 # record the steps
    try:
        while(True):
            x_, y_ = sess.run([x_input, y_label])
            step += 1
            print("{} batch - {} examples".format(step, len(y_)))
            print(x_, y_)
    except tf.errors.OutOfRangeError:
        pass

##



# setting
feature_dims = 784 # example with 784 features
neurons = 1024 # fully connected layer with 1024 neurons
classes = 10 # 10 classes classification problem
##
def fully_connected_layer(x_inputs, out_dim, name='fc'):
    """ Low level method
        x_inputs: a batch examples [batch_size, feature_dims]
        out_dim: neurons in this layer.
    """
    in_dim = x_inputs.shape[-1] # feature_dims
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights", shape=[in_dim, out_dim])
        bias = tf.get_variable("bias", shape=[out_dim])
        out = tf.matmul(x_inputs, weights) + bias
        return out


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, feature_dims])
fc = fully_connected_layer(x, neurons, "fc")
out = fully_connected_layer(fc, classes, "logits")
print("[Output tensor]", fc, out, sep="\n")
print("\n[Variables] ", *tf.trainable_variables(), sep="\n")
##

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, feature_dims])
fc = tf.layers.dense(x, neurons, activation=tf.nn.relu, name="fc")
out = tf.layers.dense(fc, classes,
                      activation=tf.nn.softmax, name="logits")
print("[Output tensor]", fc, out, sep="\n")
print("\n[Variables] ", *tf.trainable_variables(), sep="\n")
##
# Naive idea, this is not runnable.

vocabulary_size = 10000
embedding_size = 128
batch_size = 64

with tf.Graph().as_default() as g:
    center_words = tf.placeholder(tf.int32, [batch_size])
    target_words = tf.placeholder(tf.int32, [batch_size])

    encode_matrix = tf.get_variable("encoder",
                                    shape=[vocabulary_size, embedding_size])
    decode_matrix = tf.get_variable("decoder",
                                    shape=[embedding_size, vocabulary_size])

    embedding = tf.matmul(center_words, encode_matrix)
    logits = tf.matmul(embedding, decode_matrix)

    output = tf.nn.softmax(logits)

##


# Download the data.
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
DATA_FOLDER = "data"
FILE_NAME = "text8.zip"
EXPECTED_BYTES = 31344016


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    local_file_path = os.path.join(DATA_FOLDER, file_name)
    if os.path.exists(local_file_path):
        print("Dataset ready")
        return local_file_path
    file_name, _ = urllib.request.urlretrieve(
        os.path.join(DOWNLOAD_URL, file_name), local_file_path)
    file_stat = os.stat(local_file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception(
            'File ' + file_name +
            ' might be corrupted. You should try downloading it with a browser.')
    return local_file_path


make_dir(DATA_FOLDER)
file_path = download(FILE_NAME, EXPECTED_BYTES)




##

# Read the data into a list of strings.
def read_data(file_path):
    """ Read data into a list of tokens"""
    with zipfile.ZipFile(file_path) as f:
        # tf.compat.as_str() converts the input into the string
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(file_path)
print('Data size', len(vocabulary))
##
import collections
# Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
    """ Create two dictionaries and count of occuring words
        - word_to_id: map of words to their codes
        - id_to_word: maps codes to words (inverse word_to_id)
        - count: map of words to count of occurrences
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    word_to_id = dict() # (word, id)
    # record word id
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys())) # (id, word)
    return word_to_id, id_to_word, count

def convert_words_to_id(words, dictionary, count):
    """ Replace each word in the dataset with its index in the dictionary"""
    data_w2id = []
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data_w2id.append(index)
    count[0][1] = unk_count
    return data_w2id, count

##
"""Filling 4 global variables:
# data_w2id - list of codes (integers from 0 to vocabulary_size-1).
              This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# word_to_id - map of words(strings) to their codes(integers)
# id_to_word - maps codes(integers) to words(strings)
"""

vocabulary_size = 50000
word_to_id, id_to_word, count = build_dataset(vocabulary, vocabulary_size)
data_w2id, count = convert_words_to_id(vocabulary, word_to_id, count)
del vocabulary  # reduce memory.

##
print('Most common words (+UNK)', count[:5])
print('Sample data: {}'.format(data_w2id[:10]))
print([id_to_word[i] for i in data_w2id[:10]])

##
# utility function
def generate_sample(center_words, context_window_size):
    """ Form training pairs according to the skip-gram model."""
    for idx, center in enumerate(center_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in center_words[max(0, idx - context) : idx]:
            yield center, target
        # get a random target after the center word
        for target in center_words[idx + 1 : idx + context + 1]:
            yield center, target

def batch_generator(data, skip_window, batch_size):
    """ Group a numeric stream into batches and yield them as Numpy arrays."""
    single_gen = generate_sample(data, skip_window)
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1], dtype=np.int32)
        for idx in range(batch_size):
            center_batch[idx], target_batch[idx] = next(single_gen)
        yield center_batch, target_batch

##
## some training settings
training_steps = 1000
skip_step = 100
graph_dir = "graphs/word2vec_simple"
ckpt_dir = "checkpoints/word2vec_simple"

## some hyperparameters
batch_size = 128
embed_size = 128
num_sampled = 64
learning_rates = 1.0


## geneartor for `tf.data.Dataset`
def gen():
    """ Return a python generator that generates batches. """
    yield from batch_generator(data_w2id, 2, batch_size)


## model
def word2vec(dataset):
    """ 1. Build the graph"""
    with tf.name_scope("data"):
        # one_shot_iterator doesn't need to be initialized
        iterator = dataset.make_one_shot_iterator()
        # get the input and output
        center_words, target_words = iterator.get_next()

    with tf.name_scope('embed'):
        embedding_matrix = tf.get_variable("embedding_matrix",
                                           shape=[vocabulary_size, embed_size])
        embedding = tf.nn.embedding_lookup(embedding_matrix,
                                           center_words, name='embedding')

    with tf.name_scope('loss'):
        initializer = tf.truncated_normal_initializer(stddev=1.0 / (embed_size ** 0.5))
        nce_weight = tf.get_variable('nce_weight',
                                     shape=[vocabulary_size, embed_size],
                                     initializer=initializer)
        nce_bias = tf.get_variable('nce_bias', shape=[vocabulary_size],
                                   initializer=tf.zeros_initializer)

        # define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias,
                                             labels=target_words,
                                             inputs=embedding,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size), name='loss')
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss)

    # store checkpoints
    saver = tf.train.Saver()

    """ 2. Execute a session """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # avoids occupying full memory of GPU
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # we use this to calculate late average loss in the last SKIP_STEP steps
        total_loss = 0.0
        writer = tf.summary.FileWriter(graph_dir, sess.graph)

        for index in range(1, training_steps + 1):
            try:
                loss_batch, _ = sess.run([loss, optimizer])
                total_loss += loss_batch
                if index % skip_step == 0:
                    print('Average loss at step {}: {:5.1f}'.format(
                        index, total_loss / skip_step))
                    total_loss = 0.0
                    saver.save(sess,
                               os.path.join(ckpt_dir, "model"),
                               index)
            except tf.errors.OutOfRangeError:
                pass
        writer.close()


tf.reset_default_graph()

dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
                                         (tf.TensorShape([batch_size]),
                                          tf.TensorShape([batch_size, 1])))
word2vec(dataset)

##

