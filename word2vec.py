import tensorflow as tf
import numpy as np
from corpus_reader import CorpusReader

batch_size = 20
neg_samples = 10
embedding_size = 50
vocabulary_size = 100

def init_weights(shape):
    init = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init)

def init_biases(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

X_train = tf.placeholder(tf.int32, [batch_size])
y_train = tf.placeholder(tf.int32, [batch_size, 1])

embeddings = init_weights([vocabulary_size, embedding_size])
W = init_weights([vocabulary_size, embedding_size])
b = init_biases([vocabulary_size])

batch_embed = tf.nn.embedding_lookup(embeddings, X_train)

loss = tf.nn.nce_loss(W, b, batch_embed, y_train, neg_samples, vocabulary_size)
train = tf.train.AdamOptimizer(1e-3).minimize(loss)

#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    for _ in range(20):
#        batch = None
#        sess.run(train, feed_dict={X_train: batch[0], y_train: batch[1]})
