Retrieving information about the Poplar compilation and execution
-----------------------------------------------------------------

Several mechanisms are available to retrieve trace information about the
Poplar IPU compilation and executions.  Firstly, there are environment variables
provided by Poplar itself to dump the compilation and execution reports into a
file.  The Poplar documentation can give more information about these.

Within TensorFlow, the basic steps for this are.

* Include an operation in the graph that can retrieve reports
* Enable tracing in the hardware configuration options
* Execute a graph, including the operation to retrieve the reports
* Extract the reports from the returned events


Adding an operation to the graph to get compilation and execution events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two operations are available to fetch events from the Poplar backend. The first
is an operation which fetches the reporting events into a tensor, and is
typically executed independently of the main graph.  The second is a summary
event which will extract the reports along with any other summary events. These
events will typically be written into a file using the
tensorflow.summary.FileWriter class.

``ipu_event_trace()``
_____________________

This is an op which retrieves all IPU events since the last time it was
executed. The operation must be placed on the CPU, and returns the events as a
one dimensional tensor of strings containing serialised IPU event protobufs,
from ``tensorflow.compiler.plugin.poplar.driver.trace_pb2.IpuTraceEvent``.

::

  import tensorflow as tf
  from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
  from tensorflow.core.protobuf import config_pb2
  from tensorflow.contrib.ipu import utils

  pa = tf.placeholder(tf.float32, [2,2], name="a")
  pb = tf.placeholder(tf.float32, [2,2], name="b")

  with tf.contrib.ipu.ops.ipu_scope("/device:IPU:0"):
    out = pa + pb

  with tf.device('cpu'):
    report = gen_ipu_ops.ipu_event_trace()

  opts = utils.create_ipu_config(profiling=True)
  utils.configure_ipu_system(opts)

  with tf.Session() as sess:
    result = sess.run(out, {pa:[[1,2],[3,4]], pb:[[5,6],[7,8]]})
    logs = sess.run(report)
    print tf.contrib.ipu.utils.extract_all_strings_from_event_trace(logs);


``ipu_compile_summary(name, op)``
_________________________________

This produces a summary, which can be tied into the rest of the summary system
to produce output for Tensorboard. The parameter name is the name of the
summary, and op is one of the ops in the IPU graph. It is best to choose either
the inference output for an inference graph, the loss output for an evaluation
graph, or the train op for a training graph.

::

  import tensorflow as tf
  from tensorflow.contrib import ipu
  from tensorflow.core.protobuf import config_pb2

  ...

  tf.summary.scalar('c_out', c)
  ipu.ops.ipu_compile_summary('report', c)
  all_sum = tf.summary.merge_all()

  ...

  f = tf.summary.FileWriter('logs')
  with tf.Session() as s:
    sum_out, ... = s.run([add_sum, ...])
    f.add_summary(sum_out, 0)

    print "c = " + str(c)


Enabling tracing in the hardware configuration options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main function for producing an IPU system hardware configuration is called
``tensorflow.contrib.ipu.create_ipu_config``.  It provides several options for
controlling the logging and tracing of Poplar compilations.

``profiling``
_____________

This enables compilation and execution graph reports in Poplar, and generates
COMPILE_BEGIN and COMPILE_END events in the trace.

``enable_ipu_events``
_____________________

Setting this to True leaving ``profiling`` as False will generate trace events
without setting the Poplar compilation and execution reports in them.  This is
useful for getting timing information from the event trace without the overhead
of the Poplar reporting.

``use_poplar_text_report``
__________________________

Normally, the Poplar reports are generated in JSON format.  Setting this
parameter to True will generate a text summary report instead of JSON.

``use_poplar_cbor_report``
__________________________

Instead of a JSON format report, a CBOR format report will be generated.

``profile_execution``
_____________________

When this is set to true, then EXECUTE events will be generated in addition to
compilation events.

``report_every_nth_execution``
______________________________

This will restrict the number of execution reports to a subset of all
executions.

``max_report_size``
___________________

Poplar reports can get very large.  This parameter can be used to restrict the
maximum size of report generated.  Reports larger than this value will be
discarded and a warning message sent to the TensorFlow log.

``report_directory``
____________________

Rather than reports being placed directly into the events, they can be written
to a file, and the filename written into the event log.  This behaviour is
enabled by setting this parameter to a directory name.


Extract the reports from the returned events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the summary event generator has been used then the events will be inside
Tensor type events in the Tensorboard logs.  A tool is available for extracting
these all from the log.  This is available in the GraphCore Toolshed repository
on GitHub.

If the individual report gathering event is used then executing it will return
an array of Tensors.  Within each Tensor is a string which is an IpuTraceEvent
of one type.

The IpuTraceEvent is within the Tensorflow namespace at
``tensorflow.compiler.plugin.poplar.driver.trace_pb2.IpuTraceEvent``.  It is
a protobuf that can be decoded from the string into an object with fields
containing trace information.

Several utility functions are available for extracting fields.

::

  rep = sess.run(report)
  compile_reports = ipu.utils.extract_compile_reports(rep)
  execute_reports = ipu.utils.extract_execute_reports(rep)
  events = ipu.utils.extract_all_events(rep)

See the :ref:`api-section` section.


COMPILE_BEGIN
_____________

This event is generated when the Poplar compilation begins.  It contains the
XLA module name, a timestamp and the ordinal of the device that the compilation
was done for.

COMPILE_END
___________

This is generated when the Poplar compilation ends.  It contains the module
name, a timestamp, an ordinal and some compilation trace fields.


* ``compilation_report`` is the Poplar compilation report.
* ``duration`` is the duration of the compilation.
* ``tensor_map`` is a mapping of tensors generated by XLA/HLO instructions to
  the IPU tiles where those tensors are mapped.


The ``tensor_map`` field has the following format. It is JSON, but in order to
keep it dense, it is mostly JSON lists, instead of keyed dictionaries.

At the top level there is a map called 'mappings' which contains an entry for
each XLA computation, keyed by the name of that computation.  The value is a
list of tensors generated by that computation.

::

  { 'mapping' : {'computation_0' : [ ... ], 'computation_1' : [ ... ] } }

Each tensor in that list is also a list, consisting of the following items.

::

  0 - name of the XLA/HLO instruction generating the tensor.
  1 - the ordinal of the tensor produced by that instruction.
  2 - a list of integers indicating the shape of the tensor.
  3 - a string indicating the tensor element type.
  4 - a Boolean indicating if the tensor contains any constant elements.
  5 - a Boolean indicating if the tensor contains any aliases.
  6 - the total number of elements in the tensor.
  7 - a list of information about the elements on each tile.

  [ 'add.0', 0, [32, 32], 'float', 0, 0, 2, 256, [ ... ] ]

The list of elements on each tile has one entry per tile that contains
elements of the tensor. Each entry is itself a list, containing the following
items.

::

  - the tile index number.
  - the total number of elements on that tile.


EXECUTE
_______

This event contains the Poplar execution report in the ``execution_report``
field.

TensorFlow options for reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO add TF_CPP_MIN_VLOG_LEVEL, TF_CPP_VMODULE, and useful XLA_FLAGS options

