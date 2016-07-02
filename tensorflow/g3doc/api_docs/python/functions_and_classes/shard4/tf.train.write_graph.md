### `tf.train.write_graph(graph_def, logdir, name, as_text=True)` {#write_graph}

Writes a graph proto to a file.

The graph is written as a binary proto unless `as_text` is `True`.

```python
v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
```

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` protocol buffer.
*  <b>`logdir`</b>: Directory where to write the graph. This can refer to remote
    filesystems, such as Google Cloud Storage (GCS).
*  <b>`name`</b>: Filename for the graph.
*  <b>`as_text`</b>: If `True`, writes the graph as an ASCII proto.

