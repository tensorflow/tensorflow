Training a model
----------------

TensorFlow XLA and Poplar provide the opportunity to fuse an entire training
graph into a single operation in the TensorFlow graph.  This accelerates
training by removing the need to make calls to the IPU hardware for each
operation in the graph.

However, if the python code with the training pass on it is called multiple
times, once for each batch in the training data set, then there is still
the overhead of calling the hardware for each batch.

The GraphCore IPU support for TensorFlow provides three mechanisms for improving
the training performance:  training loops, data set feeds, and replicated
graphs.

Training loops, DataSets and feed queues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By placing the training operations inside a loop, they can be executed multiple
times without returning control to the host.  It is possible to use a standard
TensorFlow while_loop operation to wrap the training operation, but the IPU
library provides a convenient and feature rich version.

Normally when TensorFlow runs, operations which are not inside a loop will be
executed once, and those operations will return one or more tensors with fixed
values.  However, when a training operation is placed into a loop, the inputs to
that training operation need to provide a stream of values.  Standard TensorFlow
python feed dictionaries cannot provide data in this form, so when training in
a loop, data must be fed from a TensorFlow DataSet.

More information can be found on the DataSet class and its use in normal
operation at https://www.tensorflow.org/guide/performance/datasets. TensorFlow
provides many pre-configured DataSets for use in training models.  See the site
https://www.tensorflow.org/datasets.

To construct a system that will train in a loop, you will need to understand and
create the following things:

* Wrapping your optimizer training operation in a loop.
* Create an IPUInfeedQueue to feed data to that loop.
* Create an IPUOutfeedQueue to take results out of that loop.
* Create a TensorFlow DataSet to provide data to the input queue.

The following example shows how to construct a trivial DataSet, attach it to
a model using in IPUInfeedQueue, feed results into an IPUOutfeedQueue, and
construct a loop.

.. code-block:: python

  import numpy as np
  import tensorflow as tf

  from tensorflow.contrib.ipu import ipu_compiler
  from tensorflow.contrib.ipu import ipu_infeed_queue
  from tensorflow.contrib.ipu import ipu_outfeed_queue
  from tensorflow.contrib.ipu import loops
  from tensorflow.contrib.ipu import ops
  from tensorflow.contrib.ipu import poprand
  from tensorflow.contrib.ipu import utils

  # The dataset for feeding the graphs
  ds = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[800]))
  ds = ds.map(lambda x: [x, x])
  ds = ds.repeat()

  # The host side queues
  infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds, feed_name="infeed")
  outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")

  # The device side main
  def body(x1, x2):
      d1 = x1 + x2
      d2 = x1 - x2
      outfeed = outfeed_queue.enqueue({'d1': d1, 'd2':d2})
      return outfeed

  def my_net():
      r = loops.repeat(10, body, [], infeed_queue)
      return r

  with ops.ipu_scope('/device:IPU:0'):
      run_loop = ipu_compiler.compile(my_net, inputs=[])

  # The outfeed dequeue has to happen after the outfeed enqueue
  dequeue_outfeed = outfeed_queue.dequeue()


  # Configure the hardware
  config = utils.create_ipu_config()
  config = utils.auto_select_ipus(config, 1)
  utils.configure_ipu_system(config)

  with tf.Session() as sess:
    sess.run(infeed_queue.initializer)

    sess.run(run_loop)
    result = sess.run(dequeue_outfeed)
    print(result)

In this case the DataSet is a trivial one.  It constructs a base DataSet from a
single TensorFlow constant, and then maps the output of that DataSet into a pair
of tensors.  It then arranges for the DataSet to be repeated indefinitely.

After the DataSet is constructed, the two data feed queues are constructed. The
IPUInfeedQueue takes the DataSet as a parameter, along with its name.  Every
queue in the system must have a unique name.

The IPUOutfeedQueue has extra options to control how it collects and outputs
the data sent to it.  None of these are used in this example.

Now that we have the DataSet and the queues for getting data in and out of the
device side code, we can construct the device side part of the model.  In this
example, the `body` function constructs a very simple model, which does not
even have an optimizer.  It takes the two data samples which will be provided by
the DataSet, and performs some simple maths on them, and inserts the results
into the output queue.

Typically, in this function, the full ML model would be constructed, and a
TensorFlow Optimizer would be used to generate a backward pass and variable
update operations.  The returned data would typically be a loss value, or
perhaps nothing at all if all we do is call the training operation.

The `my_net` function is where the `loops.repeat` function is called.  This
wraps the `body` function in a loop.  It takes as the first parameter the number
of times to execute the operation, in this case 10.  It also takes the function
that generated the body of the loop, in this case the function `body`, a list
of extra parameters to pass to the body, in this case none, and finally the
infeed queue which will feed data into the loop.

Next we create an IPU scope at the top level and call `ipu_compiler.compile`
passing the `my_net` function, to create the training loop in the main graph.
The output of the `ipu_compiler.compile` will be an operation that can be called
to execute the training loop.

Finally, we create an operation which can be used to fetch results from the
outfeed queue.  Note that it isn't necessary to use an outfeed queue if you do
not wish to receive any per-sample output from the training loop.  If all you
require is the final value of a tensor, then it can be output normally without
the need for a queue.

If you run this example then you will find that the result is a python
dictionary containing two numpy arrays.  The first is the `d1` array and
will contain `x1 + x2` for each iteration in the loop.  The second is the `d2`
array and will contain `x1 - x2` for each iteration in the loop.

See entries in the :ref:`api-section` for more details.

Replicated graphs
~~~~~~~~~~~~~~~~~

To improve performance, multiple IPUs can be configured to run in a data
parallel mode.  The graph is said to be replicated across multiple IPUs.

Selecting the number of replicas
________________________________

During system configuration, the user specifies the number of IPUs for the
TensorFlow device using the `tensorflow.contrib.ipu.utils.auto_select_ipus()`
function, or the `tensorflow.contrib.ipu.utils.select_ipus()` function.

A graph can be sharded across multiple IPUs (model parallelism), and then
replicated across IPUs (data parallelism).  When specifying the number of IPUs
in the system, the user must specify a multiple of the number of shards used
by the graph.

For instance, if a graph is sharded over 2 IPUs, and the user specifies 8 IPUs
to the `auto_select_ipus` function, then the graph will be replicated four
times.

Supplying data
______________

Data must be fed to a repliciated graph using DataSets and infeeds.  The
`IPUInfeedQueue` and `IPUOutfeedQueue` classes require the number of
replicas to be passed into the constructor in the `replication_factor`
parameter.

Performing parameter updates
____________________________

Each replica maintains its own copy of the graph, but during training it is
important to ensure that the graph parameters are updated so that they are
in sync across replicas.

A wrapper for standard TensorFlow optimizers is used to add extra operations to
the parameter update nodes in the graph to average updates across replicas. It
is called `tensorflow.contrib.ipu.CrossReplicaOptimizer`.  See the
:ref:`api-section` for more details.
