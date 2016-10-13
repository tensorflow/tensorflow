Exact match for the pre-1.0 tf.train.SummaryWriter.
- - -

#### `tf.train.SummaryWriter.__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)` {#SummaryWriter.__init__}




- - -

#### `tf.train.SummaryWriter.add_graph(graph, global_step=None, graph_def=None)` {#SummaryWriter.add_graph}

Adds a `Graph` to the event file.

The graph described by the protocol buffer will be displayed by
TensorBoard. Most users pass a graph in the constructor instead.

##### Args:


*  <b>`graph`</b>: A `Graph` object, such as `sess.graph`.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    graph.
*  <b>`graph_def`</b>: DEPRECATED. Use the `graph` parameter instead.

##### Raises:


*  <b>`ValueError`</b>: If both graph and graph_def are passed to the method.


- - -

#### `tf.train.SummaryWriter.add_meta_graph(meta_graph_def, global_step=None)` {#SummaryWriter.add_meta_graph}

Adds a `MetaGraphDef` to the event file.

The `MetaGraphDef` allows running the given graph via
`saver.import_meta_graph()`.

##### Args:


*  <b>`meta_graph_def`</b>: A `MetaGraphDef` object, often as retured by
    `saver.export_meta_graph()`.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    graph.

##### Raises:


*  <b>`TypeError`</b>: If both `meta_graph_def` is not an instance of `MetaGraphDef`.


- - -

#### `tf.train.SummaryWriter.add_run_metadata(run_metadata, tag, global_step=None)` {#SummaryWriter.add_run_metadata}

Adds a metadata information for a single session.run() call.

##### Args:


*  <b>`run_metadata`</b>: A `RunMetadata` protobuf object.
*  <b>`tag`</b>: The tag name for this metadata.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    StepStats.

##### Raises:


*  <b>`ValueError`</b>: If the provided tag was already used for this type of event.


- - -

#### `tf.train.SummaryWriter.add_session_log(session_log, global_step=None)` {#SummaryWriter.add_session_log}

Adds a `SessionLog` protocol buffer to the event file.

This method wraps the provided session in an `Event` protocol buffer
and adds it to the event file.

##### Args:


*  <b>`session_log`</b>: A `SessionLog` protocol buffer.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


- - -

#### `tf.train.SummaryWriter.add_summary(summary, global_step=None)` {#SummaryWriter.add_summary}

Adds a `Summary` protocol buffer to the event file.

This method wraps the provided summary in an `Event` protocol buffer
and adds it to the event file.

You can pass the result of evaluating any summary op, using
[`Session.run()`](client.md#Session.run) or
[`Tensor.eval()`](framework.md#Tensor.eval), to this
function. Alternatively, you can pass a `tf.Summary` protocol
buffer that you populate with your own data. The latter is
commonly done to report evaluation results in event files.

##### Args:


*  <b>`summary`</b>: A `Summary` protocol buffer, optionally serialized as a string.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


