# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   local_filename = os.path.join(gettempdir(), filename)
#   if not os.path.exists(local_filename):
#     local_filename, _ = urllib.request.urlretrieve(url + filename,
#                                                    local_filename)
#   statinfo = os.stat(local_filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     print(statinfo.st_size)
#     raise Exception('Failed to verify ' + local_filename +
#                     '. Can you get to it with a browser?')
#   return local_filename


# filename = 'text8.zip'


# # Read the data into a list of strings.
# def read_data(filename):
#   """Extract the first file enclosed in a zip file as a list of words."""
#   with zipfile.ZipFile(filename) as f:
#     data = tf.compat.as_str(f.read(f.namelist()[0])).split()
#   return data

# # Vocabulary is a list of all the words in the original text with duplicates
# # Example "The quick brown fox jumps over the lazy dog."
# # -> ["the", "quick", "brown" ..., "the", "lazy", "dog"]
# vocabulary = read_data(filename)
# print('Data size', len(vocabulary))

# # Step 2: Build the dictionary and replace rare words with UNK token.
# # We only care about the 50000 most popular words
# vocabulary_size = 50000


# def build_dataset(words, n_words):
#   """Process raw inputs into a dataset."""
#   # Weight the unknown words (not a part of the 50000 most popular) with -1
#   count = [['UNK', -1]]
#   # Add 50000 -1 (unknown word) to the count along with their occurrences
#   count.extend(collections.Counter(words).most_common(n_words - 1))

#   # Encode the words into a digit
#   # Example:
#   # the:0 sun:1 is:2 ...
#   dictionary = dict()
#   for word, _ in count:
#     dictionary[word] = len(dictionary)

#   # For every word, if it exists in the dictionary get the code or get 0
#   # data is original text with words replaced with their rank of popularity
#   # -> The sun is shining and it the sun and moon are good.
#   # -> [['UNK', -1], ('and', 2), ('sun', 2), ('the', 2), ('good', 1), ('is', 1), ('it', 1), ('moon', 1), ('are', 1), ('shining', 1)]
#   # [3, 2, 5, 9, 1, 6, 3, 2, 1, 7, 8, 4]
#   data = list()
#   unk_count = 0
#   for word in words:
#     index = dictionary.get(word, 0)
#     if index == 0:  # dictionary['UNK']
#       unk_count += 1
#     data.append(index)

#   # Update how many unknown words there are
#   count[0][1] = unk_count

#   # reversed dictionary is popularity rank: original word
#   reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#   return data, count, dictionary, reversed_dictionary

# # Filling 4 global variables:
# # data - list of codes (integers from 0 to vocabulary_size-1).
# #   This is the original text but words are replaced by their codes
# # count - map of words(strings) to count of occurences
# # dictionary - map of words(strings) to their codes(integers)
# # reverse_dictionary - maps codes(integers) to words(strings)
# data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
# # "The sun is shining and it the sun and moon are good.".replace(".", "").lower().split(" ")
# del vocabulary  # Hint to reduce memory.
# print('Most common words (+UNK)', count[:10])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# data_index = 0


# # Step 3: Function to generate a training batch for the skip-gram model.
# def generate_batch(batch_size, num_skips, skip_window):
#   global data_index # Make data_index a global variable

#   # num_skips how many data points per input word
#   # Make sure batch_size is divisible by skips
#   assert batch_size % num_skips == 0
#   # Make sure number of skips is less than twice the skip_window
#   assert num_skips <= 2 * skip_window

#   # Batch is an array of batch_size
#   batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#   # Two dimensional array where internal array is size 1
#   labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#   # span is total size of sliding window of words
#   span = 2 * skip_window + 1  # [ skip_window target skip_window ]

#   # Create a buffer of size span
#   buffer = collections.deque(maxlen=span)

#   # if the data_index + span is greater than data set loop back around to start of data
#   if data_index + span > len(data):
#     data_index = 0
  
#   # Fill the buffer with data
#   buffer.extend(data[data_index:data_index + span])

#   # Increment the data index by the span
#   data_index += span

#   # For i in range of the batch_size divided by number of skips
#   for i in range(batch_size // num_skips):
#     # words that are not the center aka [0,1,2] -> [0,2]
#     context_words = [w for w in range(span) if w != skip_window]
#     # randomly shuffle them (only matters if num_skips >= 2)
#     random.shuffle(context_words)

#     # words_to_use 
#     words_to_use = collections.deque(context_words)

#     # For j in the amount of data points you want per word
#     for j in range(num_skips):
#       # batch index which word, + increment
#       batch[i * num_skips + j] = buffer[skip_window]
#       # context word 
#       context_word = words_to_use.pop()
#       # 
#       labels[i * num_skips + j, 0] = buffer[context_word]
#     # if data index reaches the end of the data restart the buffer
#     # increment data index to the span
#     if data_index == len(data):
#       buffer[:] = data[:span]
#       data_index = span
#     else:
#       # refill buffer and increment data index
#       buffer.append(data[data_index])
#       data_index += 1
#   # Backtrack a little bit to avoid skipping words in the end of a batch
#   data_index = (data_index + len(data) - span) % len(data)
#   return batch, labels

# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#  print(batch[i], reverse_dictionary[batch[i]],
#        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

print "=" * 200
graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[3]) # array of size 128 with batch input aka words
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # array of size 128 with size 1 arrays inside
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  a = tf.random_uniform([5, 2], 0.0, 1.0)
  #embed = tf.nn.embedding_lookup(a, train_inputs)

  nce_weights = tf.Variable(
      tf.truncated_normal([10, 5],
                          stddev=100.0 / math.sqrt(5)))
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # print("This is a random array of values from -1 to 1")
    # b = sess.run(a)
    # print b
    # print b[0]
    # print("\n\n")
    # print("This is an embedding where the train inputs are fed.")
    # embed = tf.nn.embedding_lookup(b, train_inputs)
    # print(sess.run(embed, feed_dict={train_inputs: [1,0,3]}))

    print(sess.run(nce_weights))



  Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


  #Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# # Step 5: Begin training.
# num_steps = 100001

# with tf.Session(graph=graph) as session:
#   # We must initialize all variables before we use them.
#   init.run()
#   print('Initialized')
#   print valid_dataset

#   average_loss = 0
#   for step in xrange(num_steps):
#     batch_inputs, batch_labels = generate_batch(
#         batch_size, num_skips, skip_window)
#     feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

#     # We perform one update step by evaluating the optimizer op (including it
#     # in the list of returned values for session.run()
#     _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
#     average_loss += loss_val

#     if step % 2000 == 0:
#       if step > 0:
#         average_loss /= 2000
#       # The average loss is an estimate of the loss over the last 2000 batches.
#       print('Average loss at step ', step, ': ', average_loss)
#       average_loss = 0

#     # Note that this is expensive (~20% slowdown if computed every 500 steps)
#     if step % 10000 == 0:
#       sim = similarity.eval()
#       for i in xrange(valid_size):
#         valid_word = reverse_dictionary[valid_examples[i]]
#         top_k = 8  # number of nearest neighbors
#         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#         log_str = 'Nearest to %s:' % valid_word
#         for k in xrange(top_k):
#           close_word = reverse_dictionary[nearest[k]]
#           log_str = '%s %s,' % (log_str, close_word)
#         print(log_str)
#   final_embeddings = normalized_embeddings.eval()

# # Step 6: Visualize the embeddings.


# # pylint: disable=missing-docstring
# # Function to draw visualization of distance between embeddings.
# def plot_with_labels(low_dim_embs, labels, filename):
#   assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
#   plt.figure(figsize=(18, 18))  # in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i, :]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')

#   plt.savefig(filename)

# try:
#   # pylint: disable=g-import-not-at-top
#   from sklearn.manifold import TSNE
#   import matplotlib.pyplot as plt

#   tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
#   plot_only = 500
#   low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#   labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#   plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

# except ImportError as ex:
#   print('Please install sklearn, matplotlib, and scipy to show embeddings.')
#   print(ex)
