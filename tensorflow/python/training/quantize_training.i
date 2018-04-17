/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/graph/quantize_training.h"
#include "tensorflow/core/lib/core/status.h"

static PyObject* DoQuantizeTrainingOnGraphDefHelper(
    const string& input_graph,
    int num_bits,
    TF_Status* out_status) {
  string result;
  // TODO(suharshs): Make the QuantizeAndDequantizeV2 configurable.
  tensorflow::Status status =
      tensorflow::DoQuantizeTrainingOnSerializedGraphDef(input_graph, num_bits,
      "QuantizeAndDequantizeV2", &result);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
    Py_RETURN_NONE;
  }
  PyObject* py_str = PyBytes_FromStringAndSize(result.data(), result.size());
  if (!py_str) {
    Set_TF_Status_from_Status(out_status,
        tensorflow::Status(tensorflow::error::INTERNAL,
            "Failed to generate serialized string of the rewritten graph."));
    Py_RETURN_NONE;
  }

  return py_str;
}
%}

%ignoreall
%unignore DoQuantizeTrainingOnGraphDefHelper;

// Wrap this function
PyObject* DoQuantizeTrainingOnGraphDefHelper(
    const string& input_graph,
    int num_bits,
    TF_Status* out_status);


%insert("python") %{
def do_quantize_training_on_graphdef(input_graph, num_bits):
  """A general quantization scheme is being developed in @{tf.contrib.quantize}.

  Consider using that instead, though since it is in the tf.contrib namespace,
  it is not subject to backward compatibility guarantees.
  """
  from tensorflow.core.framework.graph_pb2 import GraphDef
  from tensorflow.python.framework import errors
  with errors.raise_exception_on_not_ok_status() as status:
    graph = GraphDef()
    result_graph_string = DoQuantizeTrainingOnGraphDefHelper(
        input_graph.SerializeToString(), num_bits, status)

  graph.ParseFromString(result_graph_string)
  return graph

do_quantize_training_on_graphdef._tf_api_names = [
    'train.do_quantize_training_on_graphdef']
%}

%unignoreall
