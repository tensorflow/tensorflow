import tensorflow as tf

from tensorflow.python import saved_model
from tensorflow.compiler.aot import tfcompile_pb2 as tfcompile


def _gcs_join(*paths):
  return '/'.join([path.rstrip('/') for path in paths])


def sparse_as_dense(tensor_gen):
  for tensor in tensor_gen:
    if isinstance(tensor, tf.SparseTensor):
      yield tensor.indices
      yield tensor.values
      yield tensor.dense_shape
    else:
      yield tensor


def tid_from_tensor(tensor):
  return tfcompile.TensorId(node_name=tensor.name, output_index=tensor.value_index)


def main(saved_model_dir, out_dir, signature_def_key, tag):

  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    meta_graph_def = saved_model.loader.load(sess, [tag], saved_model_dir)

    saver = tf.train.Saver.from_proto(
        tf.train.import_meta_graph(
            meta_graph_def,
            clear_devices=True
        )
    )
    saver.restore(sess, _gcs_join(saved_model_dir, 'variables'))

    sig = meta_graph_def.signature_def[signature_def_key]

    new_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(),
        [t.name for t in sparse_as_dense(sig.outputs.iteritems())]
    )

  new_graph = tf.Graph()
  with new_graph.as_default():
    tf.import_graph_def(new_graph_def, name='')

    input_tensors = sparse_as_dense((
        tf.saved_model.utils.get_tensor_from_tensor_info(tinfo, new_graph)
        for tinfo in sig.inputs.values()
    ))

    output_tensors = sparse_as_dense((
        tf.saved_model.utils.get_tensor_from_tensor_info(tinfo, new_graph)
        for tinfo in sig.outputs.values()
    ))

    feeds = [
        tfcompile.Feed(
            id=tid_from_tensor(t),
            shape=t.tensor_shape,
            type=t.dtype,
        ) for t in input_tensors
    ]

    fetches = [
        tfcompile.Fetch(id=tid_from_tensor(t))
        for t in output_tensors
    ]

  with tf.gfile.Open(_gcs_join(out_dir, 'graph_def.pb2'), 'wb') as f:
    f.write(new_graph_def.SerializeToString())

  with tf.gfile.Open(_gcs_join(out_dir, 'tfcompile_config.pb2'), 'wb') as f:
    f.write(tfcompile.Config(feed=feeds, fetch=fetches).SerializeToString())
