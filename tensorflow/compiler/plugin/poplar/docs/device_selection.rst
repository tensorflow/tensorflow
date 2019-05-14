Targeting the Poplar XLA device
-------------------------------

The name of the Poplar XLA devices are ``/device:IPU:X``.

A python context handler is available for setting up all appropriate scoping
while creating the graph:

::

  from tensorflow.contrib.ipu import ops

  with ops.ipu_scope("/device:IPU:0"):
    ...

To use TensorFlow constructs that contain ``while`` loops, or ``conditional``
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
selected.  By default, TensorFlow will create one device.  This device
will be for a single IPU. The first available single IPU will be used.

Two API calls are available for selecting the number and configuration
of the IPU system.

``tensorflow.contrib.ipu.util.auto_select_ipus`` allows the selection
of a number of IPUs.  The process searches for the first set of IPUs
which match the number requested.

``tensorflow.contrib.ipu.util.select_ipus`` allows the selection of
a specific IPU hardware device ordinal, as returned by the ``gc-info``
tool.

Each of these functions takes as a first argument the options structure
returned by the ``create_ipu_config`` function.  The second argument is
either an integer or a list.  When an integer is supplied, then the user
gets a single TensorFlow device (`/device:IPU:0`) configured with the
appropriate number of IPUs.  When a list of integers is provided, then the
system is configured with multiple TensorFlow IPU devices (`/device:IPU:0`,
`/device:IPU:1`, etc), configured as specified.  For examples look at the
documentation in the :ref:`api-section`.

Configuring compilation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO - stuff about the other configuration functions.

Caching of compiled executables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It can take a long time to compile a large fused graph into an executable
suitable for the IPU.  To prevent the need for compiling every time a
TensorFlow process is started, it is possible to enable an executable cache.

The environment variable ``TF_POPLAR_FLAGS`` can have the argument
``--executable_cache_path`` set to a directory where compiled files will
be placed.  Fused XLA/HLO graphs are hashed into a 64 bit hash and stored
in this directory.

::

  TF_POPLAR_FLAGS='--executable_cache_path=.'

A pair of files will be saved for each compiled graph, the TensorFlow
metadata and the Poplar executable.

The cache does not manage the files within the directory. It is the
responsibility of the user to delete files.  No index is kept of the
files, so they can be deleted without risk.
