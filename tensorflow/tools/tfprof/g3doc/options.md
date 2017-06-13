##Options

###Overview

For all tfprof views, the statistics are processed with the following procedures

1) An in-memory data structure is used represent the view.

2) `-account_type_regexes` is used to first select the operations that match
   the specified operation types. An operation has its default type
   (e.g. MatMul, Conv2D). `tfprof` also considers device as operation type.
   User can also define customized operation type. Hence, an operation has
   multiple types. Operations with matched
   types are selected for display and their statistics are aggregated
   by the in-memory data structure.

3) Various `-xxx_name_regexes`,  `-min_xxx`, `-max_depth` etc options are then
   applied to further filter based on names and values.
   It's no limited operation name. In code view,
   it's the code trace. In op view, it's the operation type name. Different
   from `-account_type_regexes`, Statistics are used even if a name is not displayed.
   For example, in code view, a callee might be hidden, but its statistics is
   still aggregated by it's caller. `-account_displayed_op_only`, however,
   breaks the rule and only use statistics of displayed names.

4) Finally, the filtered data structure is displayed in a format depending
   on the `-output` option.

####Option Semantics In Different View
options usually have the same semantics in different views. However, some
can vary. For example `-max_depth` in scope view means the depth of
name scope <b>tree</b>. In op view, it means the length of operation <b>list</b>.
In graph view, in means the number of hops in the <b>graph</b>.


###Docs

`-max_depth`: Show ops that are at most this number of hops from starting op in the tree/graph structure.

`-min_bytes`: Show ops that request at least this number of bytes.

`-min_micros`: Show ops that spend at least this number of microseconds to run.

`-min_params`: Show ops that contains at least this number of parameters.

`-min_float_ops`: Show ops that contain at least this number of float operations. Only available if an op has op.RegisterStatistics() defined and OpLog is provided

`-min_occurrence`: Show ops that appear at least this number of times. Only available in "op" view.

`-step`: Show the stats of the this step when multiple steps of RunMetadata were added. By default, show the average of all steps."

`-order_by`: Order the results by [name|depth|bytes|micros|params|float_ops|occurrence]

`-account_type_regexes`: Account and display the ops whose types match one of the type regexes specified. tfprof allow user to define extra op types for ops through tensorflow.tfprof.OpLog proto. regexes are comma-sperated.

`-start_name_regexes`: Show ops starting from the ops that matches the regexes, recursively. regexes are comma-separated.

`-trim_name_regexes`: Hide ops starting from the ops that matches the regexes, recursively, regexes are comma-seprated.

`-show_name_regexes`: Show ops that match the regexes. regexes are comma-seprated.

`-hide_name_regexes`: Hide ops that match the regexes. regexes are comma-seprated.

Notes: For each op, `-account_type_regexes` is first evaluated, only ops with
types matching the specified regexes are accounted and selected for displayed.
`-start/trim/show/hide_name_regexes` are used to further filter ops for display.
`-start_name_regexes` is evaluated first to search the starting ops to display.
Descendants of starting ops are then evaluated against `-show/hide_name_regexes`
to make display decision. If an op matches trim_name_regexes, all its
descendants are hidden. Ops statistics are *accounted even if they are hidden*
as long as they match the `-account_xxx` options.

`-account_displayed_op_only`: If True, only account the statistics of ops eventually displayed. If False, account all op statistics matching -account_type_regexes recursively.

`-select`: Comma-separated list of metrics to show:
[bytes|micros|params|float_ops|occurrence|tensor_value|device|op_types|input_shapes].

`-output`: Output results as stdout, file or timeline.
The format is ```output_type:key=value,key=value```.
For example: ```-output timeline:outfile=<filename>```.

```shell
timeline: key=outfile, value=<filename>.
stdout: none.
file: key=outfile, value=<filename>.
```
