Targeting the Poplar XLA device
-------------------------------

The name of the Poplar XLA devices are ``/device:IPU:X``.

A python context handler is available for setting up all appropriate scoping
while creating the graph:

.. literalinclude:: tutorial_sharding.py
  :language: python
  :start-at: # Create the IPU section of the graph
  :end-at: result = ipu.ipu_compiler.compile

For very simple graphs, it is sufficient to use the IPU scope to define the
parts of the graph which will be compiled.  For most graphs, the function
``ipu_compiler.compile()`` must be used.  This must be placed inside an IPU
device scope.

The function ``ipu_compiler.compile()`` will cause all operations created by the
python function passed into its first argument to be placed on the IPU system,
and be compiled together into a single Poplar executable.

Supported types
~~~~~~~~~~~~~~~

Poplar and the poplibs libraries support the following data types:

::

  tf.float32
  tf.float16
  tf.int32
  tf.bool

Device selection
~~~~~~~~~~~~~~~~

Hardware configuration options allow the number of IPU devices to be
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

Once the hardware configuration stucture has been configured, the API call
``ipu.utils.configure_ipu_system`` must be used to attach and to configure the
hardware.

.. literalinclude:: tutorial_sharding.py
  :language: python
  :start-at: # Configure the IPU system
  :end-at: ipu.utils.configure_ipu_system



Configuring compilation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``create_ipu_config`` function has many options for system configuration.
They are divided into roughly three categories.

1) Profiling and report generation.
2) IO control.
3) Graph creation.

In addition to ``auto_select_ipus`` and ``select_ipus``, several other functions
exist for configuring the hardware and compiler. ``set_compilation_options``
sets general options to be passed to the Poplar compiler.
``set_convolution_options`` and ``set_pooling_options`` configure specific
types of operation. ``set_report_options`` allows options to be passed directly
to the Poplar summery report generator. ``set_ipu_model_options`` allows control
of the Poplar IPU_MODEL device type. ``set_recomputation_options`` turns on
recomputation, to reduce the memory requirement at the expense of speed.
``set_floating_point_behaviour_options`` allows control of the IPUs floating
point control register.

See the documentation in :ref:`api-section` for more details.

.. _env-var-section:

TF_POPLAR_FLAGS environment variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The options passed through ``create_ipu_config`` and ``configure_ipu_system``
can be directed at any machine in a TensorFlow cluster.  Some configuration
options are provided by an environment variable called ``TF_POPLAR_FLAGS``.

Setting ``TF_POPLAR_FLAGS=--help`` and executing a TF session will produce some
help for each option.

``--use_synthetic_data`` will prevent the system from downloading or uploading
data to the card when executing code.  This is used for testing performance
without the overhead of data transfer.

``--synthetic_data_initializer`` when used in combination with the
``--use_synthetic_data`` flag, all the inputs to the graph will be initialized
directly on the IPU either randomly (synthetic_data_initializer=random) or to a
constant value X (synthetic_data_initializer=X)

``--force_replicated_mode`` allows graphs without ``AllReduce`` operations in
them to be executed in replicated mode.  This might be required if replicated
graphs are used in inference mode, or where there are no per-replica trainable
parameters.

``--max_compilation_threads`` sets the maximum number of threads which Poplar
is allowed to use for compiling the executable.

``--save_oom_profiler`` specifies a file where the compilation profile will be
stored in the event of an out-of-memory when compiling.

``--save_vertex_graph`` dumps the Poplar vertex graph (DOT file) to the given
directory.

``--save_interval_report`` dumps the Poplar interval report to the given
directory.

``--executable_cache_path`` enables the Poplar executable cache. See below.

``--dump_schedule_as_dot`` creates a file containing the scheduled HLO graph as
a graphviz DOT file.

``--tensor_map_file_path`` will cause a JSON file containing the tile mapping
of all tensors to be written to this directory.

``--fallback_scheduler`` uses the standard TensorFlow scheduler, instead of
the GraphCore specific one.

The options can be used at the same time by treating them as command line
switches, eg. ``--executable_cache_path=/tmp/cache --force_replicated_mode``


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

  TF_POPLAR_FLAGS='--executable_cache_path=/tmp/cachedir'

A pair of files will be saved for each compiled graph, the TensorFlow
metadata and the Poplar executable.

The cache does not manage the files within the directory. It is the
responsibility of the user to delete files.  No index is kept of the
files, so they can be deleted without risk.

Supported operations
~~~~~~~~~~~~~~~~~~~~

The Poplar SDK is distributed with another file containing a list of all
TensorFlow operations which are supported by the IPU.

Unsupported operations
~~~~~~~~~~~~~~~~~~~~~~

TensorFlow core operations which use variable buffers or strings are not
supported. For instance, ``JpegDecode``.

Unsupported operations will cause the compilation to fail. By including
``config=tf.ConfigProto(log_device_placement=True)`` as an argument to the
creation of the session, you can check whether the operations in your graph have
been targeted at the Poplar device:

::

  # Creates a session with log_device_placement set to True.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

