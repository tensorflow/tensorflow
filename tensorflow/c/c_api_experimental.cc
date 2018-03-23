/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api_experimental.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/compiler/jit/legacy_flags/mark_for_compilation_pass_flags.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/config.pb.h"

using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::Status;
using tensorflow::Tensor;

// struct TF_Operation { tensorflow::Node node; };
static TF_Operation* ToTF_Operation(Node* node) {
  return static_cast<TF_Operation*>(static_cast<void*>(node));
}

void TF_EnableXLACompilation(TF_SessionOptions* options, unsigned char enable) {
  tensorflow::ConfigProto& config = options->options.config;
  auto* optimizer_options =
      config.mutable_graph_options()->mutable_optimizer_options();
  if (enable) {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);

    // These XLA flags are needed to trigger XLA properly from C (more generally
    // non-Python) clients. If this API is called again with `enable` set to
    // false, it is safe to keep these flag values as is.
    tensorflow::legacy_flags::MarkForCompilationPassFlags* flags =
        tensorflow::legacy_flags::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
  } else {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  }
}

void TF_InitializeTPU(TF_Session* session, TF_Status* status) {
  VLOG(1) << "Initializing TPU";
  TF_Operation* config_op =
      TF_GraphOperationByName(session->graph, "ConfigureDistributedTPU");
  if (config_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find node ConfigureDistributedTPU in the TF graph.");
    return;
  }

  TF_Output config_node{config_op, 0};

  TF_Tensor* dummy_output;
  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ &config_node, /*output_values*/ &dummy_output,
                /*noutputs*/ 1,
                /*targets*/ nullptr, /*ntargets*/ 0,
                /*run_metadata*/ nullptr, status);
  if (status->status.ok()) {
    TF_DeleteTensor(dummy_output);
  }
}

void TF_ShutdownTPU(TF_Session* session, TF_Status* status) {
  {
    tensorflow::mutex_lock c(session->graph->mu);
    VLOG(1) << "Shutting down TPU, with input graph: "
            << session->graph->graph.ToGraphDefDebug().DebugString();
  }

  TF_Operation* shutdown_op =
      TF_GraphOperationByName(session->graph, "ShutdownDistributedTPU");
  if (shutdown_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find node ShutdownDistributedTPU in the TF graph.");
    return;
  }

  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ nullptr, /*output_values*/ nullptr,
                /*noutputs*/ 0,
                /*targets*/ &shutdown_op, /*ntargets*/ 1,
                /*run_metadata*/ nullptr, status);
}

TF_CAPI_EXPORT extern const char* TF_GraphDebugString(TF_Graph* graph,
                                                      size_t* len) {
  tensorflow::mutex_lock c(graph->mu);
  const auto& debug_str = graph->graph.ToGraphDefDebug().DebugString();
  *len = debug_str.size();
  char* ret = static_cast<char*>(malloc(*len + 1));
  memcpy(ret, debug_str.c_str(), *len + 1);
  return ret;
}

// TODO(hongm): Replace this will a real implementation.
static tensorflow::Status BuildDatasetTest(TF_Graph* dataset_graph,
                                           Node** dataset_node) {
  tensorflow::mutex_lock c(dataset_graph->mu);
  Tensor const_t(tensorflow::DT_INT32, tensorflow::TensorShape({}));
  const_t.flat<tensorflow::int32>()(0) = 1;

  Node* const_node;
  TF_RETURN_IF_ERROR(NodeBuilder("Const", "Const")
                         .Attr("dtype", tensorflow::DT_INT32)
                         .Attr("value", const_t)
                         .Finalize(&dataset_graph->graph, &const_node));

  std::vector<NodeBuilder::NodeOut> input_list;
  input_list.push_back(NodeBuilder::NodeOut(const_node, 0));

  return NodeBuilder("TensorDataset", "TensorDataset")
      .Input(input_list)
      .Attr("Toutput_types", {tensorflow::DT_INT32})
      .Attr("output_shapes", {tensorflow::TensorShapeProto()})
      .Finalize(&dataset_graph->graph, dataset_node);
}

//  On success, returns a newly created TF_Function instance from
//  `text_proto`. It must be deleted by calling TF_DeleteFunction.
static TF_Function* CreateFunctionFromTextProto(const char* text_proto,
                                                TF_Status* status) {
  tensorflow::FunctionDef fdef;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(text_proto, &fdef)) {
    status->status = tensorflow::errors::Internal(
        "Invalid text proto for FunctionDef: ", text_proto);
    return nullptr;
  }
  std::vector<char> binary_proto_buf(fdef.ByteSizeLong());
  fdef.SerializeToArray(binary_proto_buf.data(), binary_proto_buf.size());
  return TF_FunctionImportFunctionDef(binary_proto_buf.data(),
                                      binary_proto_buf.size(), status);
}

//  On success, returns a newly created TF_Function instance from `proto_file`,
//  and sets `dataset_name` to the created dataset name. The returned function
//  must be deleted by calling TF_DeleteFunction.
//
// TODO(hongm): Support reading the file given by `proto_file`.
static TF_Function* LoadDatasetFunction(const char* proto_file,
                                        std::string* dataset_name,
                                        TF_Status* status) {
  const char* func_def = R"PREFIX(
signature {
      name: "_make_dataset_d8de2712"
      output_arg {
        name: "TensorSliceDataset"
        type: DT_VARIANT
      }
      is_stateful: true
    }
    node_def {
      name: "TensorSliceDataset/tensors/component_0"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_FLOAT
            tensor_shape {
              dim {
                size: 3
              }
            }
       tensor_content: "\000\000(B\000\000,B\000\0000B"
          }
        }
      }
    }
    node_def {
      name: "TensorSliceDataset"
      op: "TensorSliceDataset"
      input: "TensorSliceDataset/tensors/component_0:output:0"
      attr {
        key: "Toutput_types"
        value {
          list {
            type: DT_FLOAT
          }
        }
      }
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
            }
          }
        }
      }
    }
    ret {
      key: "TensorSliceDataset"
      value: "TensorSliceDataset:handle:0"
    })PREFIX";

  *dataset_name = "_make_dataset_d8de2712";
  return CreateFunctionFromTextProto(func_def, status);
}

// TODO(hongm): Use `file_path` in the implementation.
TF_Operation* TF_MakeIteratorGetNextWithDatasets(TF_Graph* graph,
                                                 const char* file_path,
                                                 TF_Function** dataset_func,
                                                 TF_Status* status) {
  tensorflow::Status s;

  // We can parameterize the function name, if we ever need more than 1
  // iterators in a graph.
  const std::string dataset_name = "UNIQUE_DATASET";

  std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)> dataset_graph(
      TF_NewGraph(), TF_DeleteGraph);
  Node* dataset_node = nullptr;
  s = BuildDatasetTest(dataset_graph.get(), &dataset_node);
  if (!s.ok()) {
    status->status = s;
    return nullptr;
  }

  TF_Output output{ToTF_Operation(dataset_node), 0};
  std::unique_ptr<TF_Function, decltype(&TF_DeleteFunction)> result_func(
      TF_GraphToFunction(dataset_graph.get(), dataset_name.c_str(),
                         /*append_hash_to_fn_name*/ false,
                         /*num_opers*/ -1,
                         /*opers*/ nullptr,
                         /*numinputs*/ 0,
                         /*inputs*/ nullptr,
                         /*noutputs*/ 1,
                         /*outputs*/ &output,
                         /*outputnames*/ nullptr,
                         /*functionoptions*/ nullptr, "", status),
      TF_DeleteFunction);
  if (!status->status.ok()) {
    return nullptr;
  }

  TF_GraphCopyFunction(graph, result_func.get(), /*gradient*/ nullptr, status);

  if (!status->status.ok()) {
    return nullptr;
  }

  tensorflow::mutex_lock c(graph->mu);

  tensorflow::NameAttrList func;
  func.set_name(dataset_name);
  // Run the iterator node on CPU.
  Node* oneshot_iterator_node;
  std::vector<tensorflow::TensorShapeProto> output_shape_list;
  output_shape_list.push_back(tensorflow::TensorShapeProto());
  s = NodeBuilder("OneShotIterator", "OneShotIterator")
          .Device("/device:CPU:0")
          .Attr("container", "")
          .Attr("dataset_factory", func)
          .Attr("output_types", {tensorflow::DT_INT32})
          .Attr("output_shapes", output_shape_list)
          .Attr("shared_name", "")
          .Finalize(&graph->graph, &oneshot_iterator_node);
  if (!s.ok()) {
    status->status = s;
    return nullptr;
  }
  // Run shape inference function for each newly added node, so that more
  // subsequent nodes can be added to the graph via C API (TF_NewOperation()).
  s = graph->refiner.AddNode(oneshot_iterator_node);
  if (!s.ok()) {
    status->status = s;
    return nullptr;
  }

  // Run the iterator node on CPU.
  Node* getnext_node;
  s = NodeBuilder("IteratorGetNext", "IteratorGetNext")
          .Input(oneshot_iterator_node)
          .Device("/device:CPU:0")
          .Attr("output_types", {tensorflow::DT_INT32})
          .Attr("output_shapes", output_shape_list)
          .Finalize(&graph->graph, &getnext_node);
  if (!s.ok()) {
    status->status = s;
    return nullptr;
  }
  // Run shape inference function for each newly added node, so that more
  // subsequent nodes can be added to the graph via C API (TF_NewOperation()).
  s = graph->refiner.AddNode(getnext_node);
  if (!s.ok()) {
    status->status = s;
    return nullptr;
  }

  VLOG(1) << "Output graph: " << graph->graph.ToGraphDefDebug().DebugString();
  *dataset_func = result_func.release();
  return ToTF_Operation(getnext_node);
}

void TF_GetAttrScalarTensorShapeProto(TF_Buffer* value, TF_Status* status) {
  status->status = Status::OK();
  auto shape = tensorflow::TensorShape({});
  tensorflow::TensorShapeProto shape_proto;
  shape.AsProto(&shape_proto);
  status->status = MessageToBuffer(shape_proto, value);
}
