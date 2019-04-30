Targeting the Poplar XLA device
-------------------------------

The name of the Poplar XLA devices are ``/device:IPU:X``.

A python context handler is available for setting up all appropriate scoping
while creating the graph:

::

  from tensorflow.contrib.ipu import ops

  with ops.ipu_scope("/device:IPU:0"):
    ...

To use Tensorflow constructs that contain ``while`` loops, or ``conditional``
operations, the function ``ipu_compiler.compile()`` must be used.

::

  from tensorflow.contrib import ipu
  ...

  def my_net(x):
    # Forward pass
    logits = RNN(X)

    # Loss and training
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    return [loss, train]

  X = tf.placeholder(dataType, [1, timesteps, num_input])

  with ipu.ops.ipu_scope("/device:IPU:0"):
    out = ipu.ipu_compiler.compile(my_net, [X])

  ...

  result = sess.run(out[0], ...)




Device selection
~~~~~~~~~~~~~~~~

Session configuration options allow the number of IPU devices to be
selected.  By default, Tensorflow will create one device.  This device
will be for a single IPU. The first available single IPU will be used.

TODO: Add description of device selection and configuration ...
