# CRF

The CRF module implements a linear-chain CRF layer for learning to predict tag sequences. This variant of the CRF is factored into unary potentials for every element in the sequence and binary potentials for every transition between output tags.

### Usage

Below is an example of the API, which learns a CRF for some random data. The linear layer in the example can be replaced by any neural network.


```python
import numpy as np
import tensorflow as tf

# Data settings.
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

# Random features.
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

# Random tag indices representing the gold sequence.
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# All sequences in this example have the same length, but they can be variable in a real model.
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

# Train and evaluate the model.
with tf.Graph().as_default():
  with tf.Session() as session:
    # Add the data to the TensorFlow graph.
    x_t = tf.constant(x)
    y_t = tf.constant(y)
    sequence_lengths_t = tf.constant(sequence_lengths)

    # Compute unary scores from a linear layer.
    weights = tf.get_variable("weights", [num_features, num_tags])
    matricized_x_t = tf.reshape(x_t, [-1, num_features])
    matricized_unary_scores = tf.matmul(matricized_x_t, weights)
    unary_scores = tf.reshape(matricized_unary_scores,
                              [num_examples, num_words, num_tags])

    # Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        unary_scores, y_t, sequence_lengths_t)

    # Compute the viterbi sequence and score.
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
        unary_scores, transition_params, sequence_lengths_t)

    # Add a training op to tune the parameters.
    loss = tf.reduce_mean(-log_likelihood)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    session.run(tf.global_variables_initializer())

    mask = (np.expand_dims(np.arange(num_words), axis=0) <
            np.expand_dims(sequence_lengths, axis=1))
    total_labels = np.sum(sequence_lengths)

    # Train for a fixed number of iterations.
    for i in range(1000):
      tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op])
      if i % 100 == 0:
        correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Accuracy: %.2f%%" % accuracy)
```
