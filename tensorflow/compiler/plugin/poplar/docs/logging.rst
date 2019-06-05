Retrieving information about the Poplar compilation and execution
-----------------------------------------------------------------

When developing models for the IPU, it is important to be able to see how
compute tiles are being utilized and what is the balance of the memory across
them. In certain cases, such as when investigating memory over-consumption of a
model or investigating any tile imbalance issues, it is useful to produce a
trace report that will disclose a number of different aspects of graph
deployment to the IPU.

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

This is the example from the tutorial with a few lines of additional code to
create a trace report:

.. code-block:: python
    :linenos:

        import tensorflow as tf
        import numpy as np
        import os
        from tensorflow.contrib import ipu
        from tensorflow.contrib.ipu.python.ops import ipu_scope

        # Needed to generate report
        from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
        from tensorflow.contrib.ipu import utils

        # Configure argument for targeting the IPU
        cfg = ipu.utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
        cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
        cfg = ipu.utils.auto_select_ipus(cfg, 1, sharded=True)
        ipu.utils.configure_ipu_system(cfg)

        with tf.device("cpu"):
            pa = tf.placeholder(np.float32, [2], name="a")
            pb = tf.placeholder(np.float32, [2], name="b")
            pc = tf.placeholder(np.float32, [2], name="c")

            # Create a trace event
            report = gen_ipu_ops.ipu_event_trace()


        def basic_graph(pa, pb, pc):
            # Do basic addition with tensors
            o1 = pa + pb
            o2 = pa + pc
            simple_graph_output = o1 + o2
            return simple_graph_output


        with ipu_scope("/device:IPU:0"):
            result = basic_graph(pa, pb, pc)

        with tf.Session() as sess:
            # Run the graph through the session feeding it an arbitrary dictionary
            result = sess.run(result, feed_dict={pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]})

            # Generate report based on the event run in session
            trace_out = sess.run(report)
            trace_report = utils.extract_all_strings_from_event_trace(trace_out)

            # Write trace report to file
            with open('Trace_Event_Report.rep', "w") as f:
                f.write(trace_report)

            # Print the result
            print(result)

Lines *8* and *9* import two new elements that are IPU-specific APIs. The first
import is *gen_ipu_ops*, which will generate the actual event trace, while the
second import is an assortment of utility functions, a component of which will
be used here to parse the event trace to a readable output.

The event trace is created at line *23*, where *gen_ipu_ops* is called to
instantiate the trace and returns it to *report*. This is then fed to the
TensorFlow session as a *run* argument on line *42*, directly following the
session run call to the feed-forward pass through *basic_graph*. In essence the
report is generated based on the last session graph call. The trace output is
then parsed through *extract_all_strings_from_event_trace*, and a log file is
generated. The final component of writing the trace to an actual file is done on
lines *46* and *47*, where a file is opened, named and written to with the
parsed trace data.

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

Using the IPU_MODEL device
~~~~~~~~~~~~~~~~~~~~~~~~~~

When there is an out of memory situation, it is useful to do the compilation
using the IPU_MODEL device. Consider the situation in which the event trace is
being monitored to investigate a graph that creates a tile memory imbalance. In
those instances, running on the IPU will lead to an out of memory exception
before the actual report is generated, and so it is important to target the
*IPU_MODEL* over actual hardware. *IPU_MODEL* is an emulator that mimics the IPU
computational framework on the host device. It is functionally equivalent to the
IPU, but obviously the compute timings will be completely different. There are a
number of ways to target *IPU_MODEL*, but let's assume the previous code is in
the active current directory, and all the pertinent library variables required
by the IPU are set correctly. At the terminal command line, one could then type:

::


    $ TF_POPLAR_FLAGS="--use_ipu_model" python basic_graph.py


See the :ref:`env-var-section` for details about the *TF_POPLAR_FLAGS*
environment variable.

::


    ...] Device /device:IPU:0 attached to IPU: 0


where the *Device /device:IPU:0 attached to IPU: 0* indicates that the device
known to TensorFlow as */device:IPU:0* is IPU 0.  The numbering of IPUs in your
machine can be found by using the `gc-info -l` command.

TensorFlow options for reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some tracing and reporting options are provided by TensorFlow as standard, and
can be useful when developing graphs for the IPU.

`TF_CPP_MIN_VLOG_LEVEL` is an environment variable that enables the logging of
the main C++ backend.  Setting `TF_CPP_MIN_VLOG_LEVEL=1` will show a lot of
output.  Included in this is the compilation and execution of the IPU code.
The output of `TF_CPP_MIN_VLOG_LEVEL` can be overwhelming. `TF_CPP_VMODULE`
provides a mechanism to reduce the logging to certain translation units (source
files).  This combination is quite useful:

::

  TF_CPP_VMODULE='poplar_compiler=1,poplar_executable=1

Finally, there is an environment variable called `XLA_FLAGS` which provides
options to the general XLA back end.

A useful pair of flags will produce a Graphviz DOT file of the optimized HLO
graph which is passed to the Poplar compiler.

::

  XLA_FLAGS='--xla_dump_to=. --xla_dump_hlo_as_dot --xla_dump_hlo_pass_re=forward-allocation --xla_hlo_graph_sharding_color'

The HLO pass `forward-allocation` is the final pass to run before the HLO
instructions are scheduled for passing to the Poplar graph compiler. A file
called something like
`module_0001.0001.IPU.after_forward-allocation.before_hlo-memory-scheduler.dot`
will be produced.  Graphviz `dot` can be used to turn this into an image.

Reading the Poplar textual summary report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the example code is run, a new file is generated called
*Trace_Event_Report.rep*. This is the Poplar compilation report. The report is
broken into a number of sections, but the three that will be focused on here are
the first three: *Target*, *Graph*, and *Memory Usage*.

*Target* describes the target hardware, where in absence of sharding, will be a
single IPU, for instance:

::


    Target:
      Number of IPUs:         1
      Tiles per IPU:          1,216
      Total Tiles:            1,216
      Memory Per-Tile:        256.0 kB
      Total Memory:           304.0 MB
      Clock Speed (approx):   1,600.0 MHz


It is important to note that this section of the report does not distinguish
between hardware or *IPU_MODEL*, and in essence it is only dependent on the
number of IPUs selected for deployment via the sharding utility.

The next section is *Graph*, which describes the topology of the deployed graph.
For instance:

::


    Graph:
      Number of vertices:            1,219
      Number of edges:               1,223
      Number of variables:          30,562
      Number of compute sets:            4


This is from the report generated by the adder example. The graph map includes
control code, not just compute graph components. Note that the number of
vertices in the graph is suspiciously close to the *1,216* tiles on the IPU.

The *Memory Usage* section gives the memory consumption profile of the graph
from a number of different perspectives:

::


    Memory Usage:
      Total:
        Including Gaps:         23,878,396 B
        Excluding Gaps:
          By Memory Region:
            Non-interleaved:     5,355,604 B
            Interleaved:                 0 B
            Overflowed:                  0 B
          By Data Type:
              Variables:                            39,108 B
              Constants:                                 0 B
              Host Exchange Packet Headers:         10,512 B
              Global Exchange Packet Headers:            0 B
              Stack:                             3,852,288 B
              Vertex Instances:                     14,640 B
              Copy Descriptors:                          0 B
              VectorList Descriptors:                    0 B
              Vertex Field Data:                         0 B
              Control Table:                             0 B
              Control Code:                        851,272 B
              Vertex Code:                         170,788 B
              Internal Exchange Code:               60,792 B
              Host Exchange Code:                  351,328 B
              Global Exchange Code:                      0 B
              Instrumentation Results:               4,876 B
              Shared Code Storage:                       0 B
              Shared Data Storage:                       0 B
            Vertex Data (14,640B):
              By Category:
                Internal vertex state:          9,736 B
                Edge pointers:                  4,904 B
                Copy pointers:                      0 B
                Padding:                            0 B
                Descriptors:                        0 B
              By Type:
                poprand::SetSeedSupervisor                                                  34,048 B
                popops::ScaledAddSupervisor<float,float,true>                                   60 B
                popops::BinaryOp1DSupervisor<popops::expr::BinaryOpType::ADD,float>             16 B

      By Tile (Excluding Gaps):
        Range (KB) Histogram (Excluding Gaps)               Count (tiles)
             4 - 5 ****************************************  1,215
             5 - 6 *                                             1

        Maximum (Including Gaps): 49,184 (48.0 K) on tile 0
        Maximum (Excluding Gaps): 5,780 (5.6 K) on tile 0
        0 tile(s) out of memory


The information is presented in distinct sections, where first is the total
memory usage including gaps. This is followed by a breakdown of the
gap-excluding memory: first in terms of interleaved vs non-interleaved usage,
then by data type, followed by vertex data.

A useful portion of the report is the tile histogram memory consumption profile,
which in this simple case is confined to two categories. When the graph is more
complex, the histogram will most likely have a more distributed profile. In
those instances, where there is in fact a tile imbalance, the histogram produced
may look more like:


::


    By Tile (Excluding Gaps):
        Range (KB) Histogram (Excluding Gaps)               Count (tiles)
           0 -   8 *                                            20
           8 -  16 ****************************************  1,192
          16 -  24 *                                             2
          24 -  32                                               0
          32 -  40                                               0
        .
        .
        .
         488 - 496                                               0
         496 - 504                                               0
         504 - 512 *                                             1
         512 - 520                                               0
         520 - 528                                               0
        .
        .
        .
         784 - 792                                               0
         792 - 800                                               0
         800 - 808                                               0
         808 - 816 *                                             1

        Maximum (Including Gaps): 834,416 (814.9 K) on tile 0
        Maximum (Excluding Gaps): 834,339 (814.8 K) on tile 0
        2 tile(s) out of memory


In this case, two tiles are out of physical memory, while most of the allocation
is well within single tile budget. In those instances where a memory imbalance
occurs, the report will produce a detailed depiction of the operations running
on five of the most memory-subscribed tiles, (regardless if they are over their
physical limit or not), and list them in descending order in terms of memory
consumption. In the above case, Tile *0* is the most overly-subscribed tile, and
the report produces the following:

::


    Tile # 0 memory usage:
    Memory Usage:
      Total:
        Including Gaps:            834,416 B
        Excluding Gaps:
          By Memory Region:
            Non-interleaved:       122,880 B
            Interleaved:           131,072 B
            Overflowed:            580,387 B
          By Data Type:
              Variables:                           807,658 B
              Constants:                                 0 B
              Host Exchange Packet Headers:          1,160 B
              Global Exchange Packet Headers:            0 B
              Stack:                                 3,168 B
              Vertex Instances:                     12,074 B
              Copy Descriptors:                      1,385 B
              VectorList Descriptors:                  960 B
              Vertex Field Data:                     7,934 B
              Control Table:                             0 B
              Control Code:                              0 B
                .
                .
                .

            Vertex Data (22,353B):
              By Category:
                Internal vertex state:          4,152 B
                Edge pointers:                 10,798 B
                .
                .
                .
              By Type:
                poplin::ConvPartial1x1Out<float,float,true,false>                                                             6,648 B
                poplar_rt::DstStridedCopy64BitMultiAccess                                                                     2,669 B
                popops::Reduce<popops::ReduceAdd,float,float,false,0>                                                         2,542 B
                popops::ScaledAddSupervisor<float,float,true>                                                                 1,440 B
                poplar_rt::StridedCopyDA32                                                                                    1,374 B
                poplar_rt::DstStridedCopyDA32                                                                                 1,101 B
                popops::BinaryOp1DSupervisor<popops::expr::BinaryOpType::MULTIPLY,float>                                        752 B
                .
                .
                .


This information can be very useful when tracking down the source of the
over-allocation.

Producing an ELF image of the compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is another method to produce much of the same detailed operational
perspectives given in the trace event report, but using a very different
approach. In this second paradigm, the intent is to target IPU hardware, not an
emulator run on host, and use an ELF binary file created at compile time to
review the memory allocation. This second technique will be reviewed in more
concise fashion here, (only exploring how the actual binary is created and
memory-per-tile information extracted), but is detailed in the out of memory
guide.

When compiling the graph, a Poplar engine option can be used to dump the ELF
file to a specified location.

::


    POPLAR_ENGINE_OPTIONS='{"target.saveArchive":"binaries.a", "debug.allowOutOfMemory": "true"}' python basic_graph.py


The file *binaries.a* is created which is a bit-code file of the deployed graph.
To extract size information from it type the following:

::

    $ size -A binaries.a > tiles_elf.txt

This pipes a tile-by-tile rendition of the memory consumed in bytes to the file
*tiles_elf.txt*. The important part is the *data* section, which contains the
graph-dependent components of memory allocation. This can be extracted from the
file to produce a single column of data where each entry is the *data* entry:

::

    $ size -A binaries.a | grep -e ".data" | awk '{print $2}' > data_usage_per_tile.txt

The file *data_usage_per_tile.txt* will contain this single column of *data*
allocation. Further facets of the deployed graph can be extracted from this
approach, and are well documented in the out of memory guide.

Dumping auxiliary Poplar information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two environment variable flags are available to get to extra Poplar information.

Poplar vertex graph
___________________

The Poplar vertex graph is a DOT file containing a complete description of the
lowered Poplar graph.  Each node in the graph represents one vertex in the
Poplar graph operating on one region of a tensor.

It can be used for generating a GraphCore circular graph image.

Poplar interval report
______________________

The interval report is a CSV file describing the number of tiles executing,
exchanging and syncing on each instruction cycle.

It can be used for generating a GraphCore linear activity diagram.


The :ref:`env-var-section` describes how to set the environment flags correctly.

