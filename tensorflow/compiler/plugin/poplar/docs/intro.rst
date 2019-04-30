Overview
--------

Graphcore provides a version of Tensorflow with an integrated XLA driver for
targetting an IPU system.

Extensions are provided for configuring the IPU system, running training in a
hardware loop to improve performance, and targetting poplibs operations that
are not natively supported by Tensorflow.

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

