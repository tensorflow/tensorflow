Outline
-------

General outline: Update as document matures!!

The document will begin with a concise introduction to Tensorflow, reviewing key
concepts of graph definition and suggest some useful references to dive into
graph development. Three preliminary scripts are then presented to introduce
some of the basic aspects of targeting the IPU, beginning with a plain graph
definition, subsequently introducing *XLA*, (the *Accelerated Linear Algebra*),
and finishing up with an example of sharding.

General outline: Update as document matures!!

Supported types
~~~~~~~~~~~~~~~

Poplar and the poplibs libraries support the following data types:

::

  tf.float32
  tf.float16
  tf.int32
  tf.bool

Unsupported operations
~~~~~~~~~~~~~~~~~~~~~~

Tensorflow core operations which use variable buffers or strings are not
 supported. For instance, ``JpegDecode``.

Unsupported operations will cause the compilation to fail. By including
``config=tf.ConfigProto(log_device_placement=True)`` as an argument to the
creation of the session, you can checkwhether the operations in your graph have
been targeted at the Poplar device:

::

  # Creates a session with log_device_placement set to True.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

