import argparse
import functools
import itertools
import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util
from google.protobuf import text_format

parser = argparse.ArgumentParser(description="Script to test LSTM model.")
parser.add_argument("--toco",
                    type=str,
                    help="Path to toco tool",
                    required=True)
parser.add_argument("--output_path",
                    type=str,
                    help="Directory to save files")
parser.add_argument("--use_frozen_graph",
                    help="Use frozen graph",
                    action='store_true')
parser.add_argument("--save_log",
                    help="Save tensorboard log",
                    action='store_true')
parser.add_argument("--save_graph",
                    help="Save tensorflow graph and pbtxt",
                    action='store_true')
parser.add_argument("--save_dot",
                    help="Save graphviz dot file",
                    action='store_true')
parser.add_argument("--save_tflite",
                    help="Save tflite file",
                    action='store_true')

RANDOM_SEED = 342

_TF_TYPE_INFO = {
  tf.float32: (np.float32, "FLOAT"),
  tf.float16: (np.float16, "FLOAT"),
  tf.int32: (np.int32, "INT32"),
  tf.uint8: (np.uint8, "QUANTIZED_UINT8"),
  tf.int64: (np.int64, "INT64"),
}

def create_tensor_data(dtype, shape, min_value=-100, max_value=100):
  """Build tensor data sperading the range [min_value, max_value)."""

  if dtype in _TF_TYPE_INFO:
    dtype = _TF_TYPE_INFO[dtype][0]

  if dtype in (tf.float32, tf.float16):
    value = (max_value-min_value)*np.random.random_sample(shape)+min_value
  elif dtype in (tf.int32, tf.uint8, tf.int64):
    value = np.random.randint(min_value, max_value+1, shape)
  return value.astype(dtype)

def normalize_output_name(output_name):
  """Remove :0 suffix from tensor names."""
  return output_name.split(":")[0] if output_name.endswith(
      ":0") else output_name

test_parameters = [
    {
      "dtype": [tf.float32],
      "num_batchs": [1],
      "time_step_size": [2],
      "input_vec_size": [3],
      "num_cells": [128],
    },
  ]

def build_graph(parameters):
  """Build a simple graph with BasicLSTMCell."""

  num_batchs = parameters["num_batchs"]
  time_step_size = parameters["time_step_size"]
  input_vec_size = parameters["input_vec_size"]
  num_cells = parameters["num_cells"]
  inputs_after_split = []
  for i in range(time_step_size):
    one_timestamp_input = tf.placeholder(
        dtype=parameters["dtype"],
        name="split_{}".format(i),
        shape=[num_batchs, input_vec_size])
    inputs_after_split.append(one_timestamp_input)
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
      num_cells, forget_bias=1.0)
  init_state = lstm_cell.zero_state(num_batchs, tf.float32)
  cell_outputs, _ = tf.nn.static_rnn(
      lstm_cell, inputs_after_split, init_state, dtype=tf.float32)
  out = cell_outputs[-1]
  return inputs_after_split, [out]

def build_training_op(logit, label, learning_rate, decay):
  """Build a training operator"""

  with tf.name_scope("cost", reuse=True):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logit, label=label))
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)
    return train_op

def build_input(parameters, sess, inputs, outputs):
  """Feed inputs, assign variables, and freeze graph."""

  num_batchs = parameters["num_batchs"]
  time_step_size = parameters["time_step_size"]
  input_vec_size = parameters["input_vec_size"]
  input_values = []
  for _ in range(time_step_size):
    tensor_data = create_tensor_data(parameters["dtype"],
                                     [num_batchs, input_vec_size], 0, 1)
    input_values.append(tensor_data)
  out = sess.run(outputs, feed_dict=dict(zip(inputs, input_values)))
  return input_values, out

def freeze_graph(session, outputs):
  """Freeze the current graph"""

  return tf_graph_util.convert_variables_to_constants(
      session, session.graph.as_graph_def(), [x.op.name for x in outputs])

def generate_log(session):
  """Generate logs for tensorboard"""

  train_writer = tf.summary.FileWriter(FLAGS.output_path)
  train_writer.add_graph(session.graph)
  train_writer.flush()
  train_writer.close()

def toco_options(data_types,
                 input_arrays,
                 output_arrays,
                 shapes,
                 output_format):
  """Create TOCO options to process a model"""

  shape_str = ":".join([",".join(str(y) for y in x) for x in shapes if x])
  inference_type = "FLOAT"
  if data_types[0] == "QUANTIZED_UINT8":
    inference_type = "QUANTIZED_UINT8"
  s = (" --input_data_types=%s" % ",".join(data_types) +
       " --inference_type=%s" % inference_type +
       " --input_format=TENSORFLOW_GRAPHDEF" +
       " --output_format=%s" % output_format +
       " --input_arrays=%s" % ",".join(input_arrays) +
       " --output_arrays=%s" % ",".join(output_arrays))
  if shape_str:
    s += (" --input_shapes=%s" % shape_str)
  s += (" --allow_custom_ops")
  return s

def toco_convert(graph_def_str, input_tensors, output_tensors, output_format='GRAPHVIZ_DOT'):
  """Convert a model's graph def into a tflite model."""

  data_types = [_TF_TYPE_INFO[x[2]][1] for x in input_tensors]
  opts = toco_options(
      data_types=data_types,
      input_arrays=[x[0] for x in input_tensors],
      shapes=[x[1] for x in input_tensors],
      output_arrays=output_tensors,
      output_format=output_format)

  with tempfile.NamedTemporaryFile() as graphdef_file, \
       tempfile.NamedTemporaryFile() as output_file, \
       tempfile.NamedTemporaryFile("w+") as stdout_file:
    graphdef_file.write(graph_def_str)
    graphdef_file.flush()

    cmd = ("%s --input_file=%s --output_file=%s %s > %s " %
           (FLAGS.toco, graphdef_file.name, output_file.name, opts,
            stdout_file.name))
    exit_code = os.system(cmd)
    return (None if exit_code != 0 else output_file.read())

def main(unused_args):
  """Main"""

  if not os.path.exists(FLAGS.toco):
    raise RuntimeError("%s: not found" % FLAGS.toco)

  if FLAGS.output_path:
    if not os.path.exists(FLAGS.output_path):
      os.makedirs(FLAGS.output_path)
      if not os.path.isdir(FLAGS.output_path):
        raise RuntimeError("Failed to create dir %r" % FLAGS.output_path)

  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      param_dict = dict(zip(keys, curr))

      def build_example(param_dict_real):
        """Build the model with parameter values set in param_dict_real."""

        np.random.seed(RANDOM_SEED)

        # Build graph
        tf.reset_default_graph()

        with tf.device("/cpu:0"):
          try:
            inputs, outputs = build_graph(param_dict_real)
          except (tf.errors.UnimplementedError, tf.errors.InvalidArgumentError,
                  ValueError):
            return None

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        try:
          baseline_inputs, baseline_outputs = build_input(
              param_dict_real, sess, inputs, outputs)
        except (tf.errors.UnimplementedError, tf.errors.InvalidArgumentError,
                ValueError):
          return None

        # Convert graph to toco
        input_tensors = [(input_tensor.name.split(":")[0],
                          input_tensor.get_shape(), input_tensor.dtype)
                         for input_tensor in inputs]
        output_tensors = [normalize_output_name(out.name) for out in outputs]

        graph_def = freeze_graph(
            sess,
            tf.global_variables() + inputs + outputs) if FLAGS.use_frozen_graph else sess.graph_def
        model_binary = toco_convert(
            graph_def.SerializeToString(), input_tensors, output_tensors)

        # Check if `LstmCell` is contained
        print("========================")
        if 'label=\"LstmCell\"' in model_binary.decode('utf-8'):
          print("Found 'LstmCell'")
        else:
          print("Not found 'LstmCell'")
        print("========================")

        if FLAGS.output_path:
          if FLAGS.save_log:
            generate_log(sess)

          if FLAGS.save_graph:
            filename = os.path.join(FLAGS.output_path, 'LSTM.pb')
            with open(filename, 'w+b') as f:
              f.write(graph_def.SerializeToString())

            filename = os.path.join(FLAGS.output_path, 'LSTM.pbtxt')
            with open(filename, 'w') as f:
              f.write(text_format.MessageToString(graph_def))

          if FLAGS.save_dot:
            filename = os.path.join(FLAGS.output_path, 'LSTM.dot')
            with open(filename, 'w+b') as f:
              f.write(model_binary)

          if FLAGS.save_tflite:
            tflite_model_binary = toco_convert(
                graph_def.SerializeToString(), input_tensors, output_tensors, 'TFLITE')
            filename = os.path.join(FLAGS.output_path, 'LSTM.tflite')
            with open(filename, 'w+b') as f:
              f.write(tflite_model_binary)

        return model_binary

      _ = build_example(param_dict)

if __name__ == "__main__":
  FLAGS, unparsed =parser.parse_known_args()

  if unparsed:
    print("Usage: %s")
  else:
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
