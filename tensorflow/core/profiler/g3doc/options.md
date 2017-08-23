##Options

###Overview

For all tfprof views, the profiles are processed with the following procedures

1) An in-memory data structure is built represent the view.

   *  graph view. Graph. Each profiler node corresponds to a
      TensorFlow graph node.
   *  scope view. Tree. Each profiler node corresponds to a
      TensorFlow graph node.
   *  code view. Tree. Each profiler node includes to all TensorFlow
      graph nodes created by the profiler node (python code).
   *  op view. List. Each profiler node includes to all TensorFlow
      graph nodes belonging to an operation type.

2) `-account_type_regexes` is used to first select the nodes that includes
   the specified operation types. An operation has its default type
   (e.g. MatMul, Conv2D). `tfprof` also considers device as operation type.
   User can also define customized operation type. Hence, an operation has
   multiple types. Profiler nodes containing matched
   types are selected for display and their statistics are aggregated by the
   parents of the in-memory data structure.

3) Various `-xxx_name_regexes`,  `-min_xxx`, `-max_depth` etc options are then
   applied to further filter based on profiler node names and statistics.
   It's no limited operation name. In code view,
   it's the code string. In op view, it's the operation type name. Different
   from `-account_type_regexes`, Statistics are used even if a profiler node is not displayed.
   For example, in code view, a callee might be hidden, but its statistics is
   still aggregated by it's caller. `-account_displayed_op_only`, however,
   breaks the rule and only aggregates statistics of displayed names.

4) Finally, the filtered data structure is output in a format depending
   on the `-output` option.

####Option Semantics In Different View
options usually have the same semantics in different views. However, some
can vary. For example `-max_depth` in scope view means the depth of
name scope <b>tree</b>. In op view, it means the length of operation <b>list</b>.
In graph view, in means the number of hops in the <b>graph</b>.

### Times

Most machines have mutli-core CPUs. Some installs one or more accelerators.
Each accelerator usually performs massive parallel processing. The profiler
tracks the accumulated processing times. Hence, the accumulated processing
time is likely larger than the time of each step.

micros: This is the sum of cpu and accelerator times.
accelerator_micros: This is the accelerator times.
cpu_micros: This is the cpu times.

### Memory

Tensor memory are usually ref-counted. The memory is released when there is
no more reference to it. It will be difficult to track the release of memory.
Currently, profiler only tracks the allocation of memory. As a result, the
accumulated memory request is uaually larger than the peak memory of the overall
model.

bytes: The memory allocations requested by the operation.
peak_bytes: The peak requested memory (not de-allocated) by the operation.
residual_bytes: The memory requested by the operation and not de-allocated
                when Compute finishes.
output_bytes: The memory output by the operation. It's not necessarily requested
              by the current operation. For example, it can be a tensor
              forwarded from input to output, with in-place mutation.

###Docs

`-max_depth`: Show nodes that are at most this number of hops from starting node in the data structure.

`-min_bytes`: Show nodes that request at least this number of bytes.

`-min_peak_bytes`: Show nodes that using at least this number of bytes during peak memory usage.

`-min_residual_bytes`: Show nodes that have at least this number of bytes not being de-allocated after Compute.

`-min_output_bytes`: Show nodes that have at least this number of bytes output (no necessarily allocated by the nodes).

`-min_micros`: Show nodes that spend at least this number of microseconds to run. It sums
accelerator_micros and cpu_micros. Note: cpu and accelerator can run in parallel.

`-min_accelerator_micros`: Show nodes that spend at least this number of microseconds to run on accelerator (e.g. GPU).

`-min_cpu_micros`: Show nodes that spend at least this number of microseconds to run on CPU.

`-min_params`: Show nodes that contains at least this number of parameters.

`-min_float_ops`: Show nodes that contain at least this number of float operations. Only available if an node has op.RegisterStatistics() defined and OpLogProto is provided

`-min_occurrence`: Show nodes that appear at least this number of times..

`-step`: Show the stats of the this step when multiple steps of RunMetadata were added. By default, show the average of all steps."

`-order_by`: Order the results by [name|depth|bytes|peak_bytes|residual_bytes|output_bytes|micros|accelerator_micros|cpu_micros|params|float_ops|occurrence]

`-account_type_regexes`: Account and display the nodes whose types match one of the type regexes specified. tfprof allow user to define extra operation types for graph nodes through tensorflow.tfprof.OpLogProto proto. regexes are comma-sperated.

`-start_name_regexes`: Show node starting from the node that matches the regexes, recursively. regexes are comma-separated.

`-trim_name_regexes`: Hide node starting from the node that matches the regexes, recursively, regexes are comma-seprated.

`-show_name_regexes`: Show node that match the regexes. regexes are comma-seprated.

`-hide_name_regexes`: Hide node that match the regexes. regexes are comma-seprated.

`-account_displayed_op_only`: If True, only account the statistics of ops eventually displayed. If False, account all op statistics matching -account_type_regexes recursively.


Notes: See <b>overview</b> sesion on how does above options play with each other to decide the output and counting.

`-select`: Comma-separated list of attributes to show. Supported attributes:
[bytes|peak_bytes|residual_bytes|output_bytes|micros|accelerator_micros|cpu_micros|params|float_ops|occurrence|tensor_value|device|op_types|input_shapes].

`-output`: Output results as stdout, file or timeline.
The format is ```output_type:key=value,key=value```.
For example: ```-output timeline:outfile=<filename>```.

```shell
timeline: key=outfile, value=<filename>.
stdout: none.
file: key=outfile, value=<filename>.
```
