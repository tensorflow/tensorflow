# TensorBoard: Visualizing Learning

The computations you'll use TensorFlow for - like training a massive
deep neural network - can be complex and confusing. To make it easier to
understand, debug, and optimize TensorFlow programs, we've included a suite of
visualization tools called TensorBoard. You can use TensorBoard to visualize
your TensorFlow graph, plot quantitative metrics about the execution of your
graph, and show additional data like images that pass through it. When
TensorBoard is fully configured, it looks like this:

[![MNIST TensorBoard](../../images/mnist_tensorboard.png "MNIST TensorBoard")](http://tensorflow.org/tensorboard)
[*Click try a TensorBoard with data from this tutorial!*](http://tensorflow.org/tensorboard)


## Serializing the data

TensorBoard operates by reading TensorFlow events files, which contain summary
data that you can generate when running TensorFlow. Here's the general
lifecycle for summary data within TensorBoard.

First, create the TensorFlow graph that you'd like to collect summary
data from, and decide which nodes you would like to annotate with
[summary operations]
(../../api_docs/python/train.md#summary-operations).

For example, suppose you are training a convolutional neural network for
recognizing MNIST digits. You'd like to record how the learning rate
varies over time, and how the objective function is changing. Collect these by
attaching [`scalar_summary`](../../api_docs/python/train.md#scalar_summary) ops
to the nodes that output the learning rate and loss respectively. Then, give
each `scalar_summary` a meaningful `tag`, like `'learning rate'` or `'loss
function'`.

Perhaps you'd also like to visualize the distributions of activations coming
off a particular layer, or the distribution of gradients or weights. Collect
this data by attaching
[`histogram_summary`](../../api_docs/python/train.md#histogram_summary) ops to
the gradient outputs and to the variable that holds your weights, respectively.

For details on all of the summary operations available, check out the docs on
[summary operations]
(../../api_docs/python/train.md#summary-operations).

Operations in TensorFlow don't do anything until you run them, or an op that
depends on their output. And the summary nodes that we've just created are
peripheral to your graph: none of the ops you are currently running depend on
them. So, to generate summaries, we need to run all of these summary nodes.
Managing them by hand would be tedious, so use
[`tf.merge_all_summaries`](../../api_docs/python/train.md#merge_all_summaries)
to combine them into a single op that generates all the summary data.

Then, you can just run the merged summary op, which will generate a serialized
`Summary` protobuf object with all of your summary data at a given step.
Finally, to write this summary data to disk, pass the summary protobuf to a
[`tf.train.SummaryWriter`](../../api_docs/python/train.md#SummaryWriter).

The `SummaryWriter` takes a logdir in its constructor - this logdir is quite
important, it's the directory where all of the events will be written out.
Also, the `SummaryWriter` can optionally take a `Graph` in its constructor.
If it receives a `Graph` object, then TensorBoard will visualize your graph
along with tensor shape information. This will give you a much better sense of
what flows through the graph: see
[Tensor shape information](../../how_tos/graph_viz/index.md#tensor-shape-information).

Now that you've modified your graph and have a `SummaryWriter`, you're ready to
start running your network! If you want, you could run the merged summary op
every single step, and record a ton of training data. That's likely to be more
data than you need, though. Instead, consider running the merged summary op
every `n` steps.

The code example below is a modification of the [simple MNIST tutorial]
(http://tensorflow.org/tutorials/mnist/beginners/index.md), in which we have
added some summary ops, and run them every ten steps. If you run this and then
launch `tensorboard --logdir=/tmp/mnist_logs`, you'll be able to visualize
statistics, such as how the weights or accuracy varied during training.
The code below is an excerpt; full source is [here](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py).

```python
def variable_summaries(var, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read, and
  adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope("weights"):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope("biases"):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      activations = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/activations', activations)
    relu = tf.nn.relu(activations, 'relu')
    tf.histogram_summary(layer_name + '/activations_relu', relu)
    return tf.nn.dropout(relu, keep_prob)

layer1 = nn_layer(x, 784, 50, 'layer1')
layer2 = nn_layer(layer1, 50, 10, 'layer2')
y = tf.nn.softmax(layer2, 'predictions')


with tf.name_scope('cross_entropy'):
  diff = y_ * tf.log(y)
  with tf.name_scope('total'):
    cross_entropy = -tf.reduce_sum(diff)
  with tf.name_scope('normalized'):
    normalized_cross_entropy = -tf.reduce_mean(diff)
  tf.scalar_summary('cross entropy', normalized_cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(
      FLAGS.learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
tf.initialize_all_variables().run()

```

You're now all set to visualize this data using TensorBoard.


## Launching TensorBoard

To run TensorBoard, use the command

```bash
tensorboard --logdir=path/to/log-directory
```

where `logdir` points to the directory where the `SummaryWriter` serialized its
data.  If this `logdir` directory contains subdirectories which contain
serialized data from separate runs, then TensorBoard will visualize the data
from all of those runs. Once TensorBoard is running, navigate your web browser
to `localhost:6006` to view the TensorBoard.

When looking at TensorBoard, you will see the navigation tabs in the top right
corner. Each tab represents a set of serialized data that can be visualized.

For in depth information on how to use the *graph* tab to visualize your graph,
see [TensorBoard: Graph Visualization](../../how_tos/graph_viz/index.md).
