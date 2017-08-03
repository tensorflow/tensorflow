import argparse

import tensorflow as tf

from tensorflow.python import saved_model
from tensorflow.compiler.aot import tfcompile_pb2 as tfcompile


def sparse_as_dense(tensor_gen):
  for tensor in tensor_gen:
    if isinstance(tensor, tf.SparseTensor):
      yield tensor.indices
      yield tensor.values
      yield tensor.dense_shape
    else:
      yield tensor


def denset_from_tinfo(tinfo_gen, graph):
    for denset in sparse_as_dense((
        tf.saved_model.utils.get_tensor_from_tensor_info(tinfo, graph)
        for tinfo in tinfo_gen)):
      yield denset


def tid_from_tensor(tensor):
  return tfcompile.TensorId(
      node_name=tensor.op.node_def.name,
      output_index=tensor.value_index
  )


def main(saved_model_dir, graph_def_path, tfcompile_config_path, signature_def_key, tag):

  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    meta_graph_def = saved_model.loader.load(
        sess,
        [tag],
        saved_model_dir,
        clear_devices=True
    )

    sig = meta_graph_def.signature_def[signature_def_key]

    with open('test.pb2', 'wb') as f:
      f.write(graph.as_graph_def().SerializeToString())

    new_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(),
        [t.op.node_def.name for t in sparse_as_dense((
            tf.saved_model.utils.get_tensor_from_tensor_info(tinfo, graph)
            for tinfo in sig.outputs.values()
        ))]
    )

  new_graph = tf.Graph()
  with new_graph.as_default():
    tf.import_graph_def(new_graph_def, name='')

    input_tensors = denset_from_tinfo(sig.inputs.values(), new_graph)
    output_tensors = denset_from_tinfo(sig.outputs.values(), new_graph)

    feeds = [
        tfcompile.Feed(
            id=tid_from_tensor(t),
            shape=t.shape.as_proto(),
#            type=t.dtype.as_datatype_enum,
        ) for t in input_tensors
    ]

    fetches = [
        tfcompile.Fetch(id=tid_from_tensor(t))
        for t in output_tensors
    ]


  with tf.gfile.Open(graph_def_path, 'wb') as f:
    f.write(new_graph_def.SerializeToString())

  with tf.gfile.Open(tfcompile_config_path, 'wb') as f:
    f.write(tfcompile.Config(feed=feeds, fetch=fetches).SerializeToString())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--saved_model_dir')
  parser.add_argument('--graph_def_path')
  parser.add_argument('--tfcompile_config_path')
  parser.add_argument('--tag', default=saved_model.tag_constants.SERVING)
  parser.add_argument('--signature_def_key', default=saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
  args = parser.parse_args()
  main(args.saved_model_dir, args.graph_def_path, args.tfcompile_config_path, args.signature_def_key, args.tag)
