import tensorflow as tf
import numpy as np
from corpus_reader import CorpusReader

batch_size = 100
neg_samples = 40
embedding_size = 200
window_size = 1

def init_weights(shape):
    init = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init)

def init_biases(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

corpus = CorpusReader('data', window_size=window_size)
vocabulary_size = corpus.build_dictionary()

X_train = tf.placeholder(tf.int32, [batch_size])
y_train = tf.placeholder(tf.int32, [batch_size])
y = tf.reshape(y_train, [-1, 1])

embeddings = init_weights([vocabulary_size, embedding_size])
W = init_weights([vocabulary_size, embedding_size])
b = init_biases([vocabulary_size])

batch_embed = tf.nn.embedding_lookup(embeddings, X_train)

loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(W, b, batch_embed, y, neg_samples, vocabulary_size))
train = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(2000):
        word, context = zip(*corpus.next_batch(batch_size))
        l, _ = sess.run([loss, train], feed_dict={X_train: word, y_train: context})
        print l
