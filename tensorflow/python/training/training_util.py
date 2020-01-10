"""Utility functions for training."""
import os.path

from tensorflow.python.platform import gfile


def global_step(sess, global_step_tensor):
  """Small helper to get the global step.

  ```python
  # Creates a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Creates a session.
  sess = tf.Session()
  # Initializes the variable.
  sess.run(global_step_tensor.initializer)
  print 'global_step:', tf.train.global_step(sess, global_step_tensor)

  global_step: 10
  ```

  Args:
    sess: A brain `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
  return int(sess.run(global_step_tensor))


def write_graph(graph_def, logdir, name, as_text=True):
  """Writes a graph proto on disk.

  The graph is written as a binary proto unless as_text is `True`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_def: A `GraphDef` protocol buffer.
    logdir: Directory where to write the graph.
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.
  """
  path = os.path.join(logdir, name)
  gfile.MakeDirs(os.path.dirname(path))
  f = gfile.FastGFile(path, "w")
  if as_text:
    f.write(str(graph_def))
  else:
    f.write(graph_def.SerializeToString())
  f.close()
