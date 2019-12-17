import tensorflow as tf

from tensorflow.contrib.compiler import xla

# Size of each input image, 28 x 28 pixels
IMAGE_SIZE = 28 * 28
# Number of distinct number labels, [0..9]
NUM_CLASSES = 10
# Number of examples in each training batch (step)
TRAIN_BATCH_SIZE = 100
# Number of training steps to run
TRAIN_STEPS = 1000

# Loads MNIST dataset.
train, test = tf.keras.datasets.mnist.load_data()
train_ds = tf.data.Dataset.from_tensor_slices(train).batch(TRAIN_BATCH_SIZE).repeat()
test_ds = tf.data.Dataset.from_tensor_slices(test).batch(TRAIN_BATCH_SIZE)

iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
images, labels = iterator.get_next()
images = tf.reshape(images, [-1, IMAGE_SIZE])
images, labels = tf.cast(images, tf.float32), tf.cast(labels, tf.int64)

def build_mnist_model(x, y_):
  y = tf.keras.layers.Dense(NUM_CLASSES).apply(x)

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  return y, train_step

# Creates session and initialize all variables.
# xla.compile() doesn't work with Keras model.fit() API or TF eager mode yet.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Feeds training dataset
sess.run(iterator.make_initializer(train_ds))

[y] = xla.compile(build_mnist_model, inputs=[images, labels])

# Runs TRAIN_STEPS steps
for i in range(TRAIN_STEPS):
  sess.run(y)

print("Model trained for %s steps." % TRAIN_STEPS)

# Tests trained model

# Feeds testing dataset
sess.run(iterator.make_initializer(test_ds))

# Calculates accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Prediction accuracy after training: %s" % sess.run(accuracy))

# Cleans up session
sess.close()
