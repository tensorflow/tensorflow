# TensorBoard: Visualizing Learning

The computations you'll use TensorFlow for - like training a massive
deep neural network - can be complex and confusing. To make it easier to
understand, debug, and optimize TensorFlow programs, we've included a suite of
visualization tools called TensorBoard. You can use TensorBoard to visualize
your TensorFlow graph, plot quantitative metrics about the execution of your
graph, and show additional data like images that pass through it. When
TensorBoard is fully configured, it looks like this:

![MNIST TensorBoard](../../images/mnist_tensorboard.png "MNIST TensorBoard")


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
Also, the `SummaryWriter` can optionally take a `GraphDef` in its constructor.
If it receives one, then TensorBoard will visualize your graph as well.

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
The code below is an exerpt; full source is [here](../../tutorials/mnist/mnist_with_summaries.py).

```python
# Create the model
x = tf.placeholder("float", [None, 784], name="x-input")
W = tf.Variable(tf.zeros([784,10]), name="weights")
b = tf.Variable(tf.zeros([10], name="bias"))

# use a name scope to organize nodes in the graph visualizer
with tf.name_scope("Wx_b") as scope:
  y = tf.nn.softmax(tf.matmul(x,W) + b)

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

# Define loss and optimizer
y_ = tf.placeholder("float", [None,10], name="y-input")
# More name scopes will clean up the graph representation
with tf.name_scope("xent") as scope:
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("test") as scope:
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)
tf.initialize_all_variables().run()

# Train the model, and feed in test data and record summaries every 10 steps

for i in range(1000):
  if i % 10 == 0:  # Record summary data, and the accuracy
    feed = {x: mnist.test.images, y_: mnist.test.labels}
    result = sess.run([merged, accuracy], feed_dict=feed)
    summary_str = result[0]
    acc = result[1]
    writer.add_summary(summary_str, i)
    print("Accuracy at step %s: %s" % (i, acc))
  else:
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict=feed)

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

```

You're now all set to visualize this data using TensorBoard.


## Launching TensorBoard

To run TensorBoard, use the command

```bash
python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory
```

where `logdir` points to the directory where the `SummaryWriter` serialized its
data.  If this `logdir` directory contains subdirectories which contain
serialized data from separate runs, then TensorBoard will visualize the data
from all of those runs. Once TensorBoard is running, navigate your web browser
to `localhost:6006` to view the TensorBoard.

If you have pip installed TensorFlow, `tensorboard` is installed into
the system path, so you can use the simpler command

```bash
tensorboard --logdir=/path/to/log-directory
```

When looking at TensorBoard, you will see the navigation tabs in the top right
corner. Each tab represents a set of serialized data that can be visualized.
For any tab you are looking at, if the logs being looked at by TensorBoard do
not contain any data relevant to that tab, a message will be displayed
indicating how to serialize data that is applicable to that tab.

For in depth information on how to use the *graph* tab to visualize your graph,
see [TensorBoard: Graph Visualization](../../how_tos/graph_viz/index.md).
