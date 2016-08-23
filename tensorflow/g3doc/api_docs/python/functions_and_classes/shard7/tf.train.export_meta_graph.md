### `tf.train.export_meta_graph(filename=None, meta_info_def=None, graph_def=None, saver_def=None, collection_list=None, as_text=False)` {#export_meta_graph}

Returns `MetaGraphDef` proto. Optionally writes it to filename.

This function exports the graph, saver, and collection objects into
`MetaGraphDef` protocol buffer with the intention of it being imported
at a later time or location to restart training, run inference, or be
a subgraph.

##### Args:


*  <b>`filename`</b>: Optional filename including the path for writing the
    generated `MetaGraphDef` protocol buffer.
*  <b>`meta_info_def`</b>: `MetaInfoDef` protocol buffer.
*  <b>`graph_def`</b>: `GraphDef` protocol buffer.
*  <b>`saver_def`</b>: `SaverDef` protocol buffer.
*  <b>`collection_list`</b>: List of string keys to collect.
*  <b>`as_text`</b>: If `True`, writes the `MetaGraphDef` as an ASCII proto.

##### Returns:

  A `MetaGraphDef` proto.

