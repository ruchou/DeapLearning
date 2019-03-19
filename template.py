import tensorflow as tf

## template
with tf.Graph().as_default() as g:
    """ Define dataset and iterator """
    with tf.name_scope("data"):
        pass

    """ Build the model """
    with tf.name_scope("model"):
        pass

    """ Define the loss """
    with tf.name_scope("loss"):
        pass

    """ Define the optimizer """
    with tf.name_scope("optimizer"):
        pass

    """ Other tensors or operations you need """
    with tf.name_scope("accuracy"):
        pass

with tf.Session(graph=g) as sess:
    """ Initialize the variables """
    """ Run the target tensors and operations """
    pass