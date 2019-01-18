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

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

using tensorflow::FunctionDef;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::Status;

namespace {
typedef std::unique_ptr<TF_Function, decltype(&TF_DeleteFunction)>
    UniqueFuncPtr;
}

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
    tensorflow::MarkForCompilationPassFlags* flags =
        tensorflow::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
  } else {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  }
}

TF_Buffer* TF_CreateConfig(unsigned char enable_xla_compilation,
                           unsigned char gpu_memory_allow_growth,
                           unsigned int num_cpu_devices) {
  tensorflow::ConfigProto config;
  auto* optimizer_options =
      config.mutable_graph_options()->mutable_optimizer_options();
  if (enable_xla_compilation) {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);

    // These XLA flags are needed to trigger XLA properly from C (more generally
    // non-Python) clients. If this API is called again with `enable` set to
    // false, it is safe to keep these flag values as is.
    tensorflow::MarkForCompilationPassFlags* flags =
        tensorflow::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
  } else {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  }

  auto* gpu_options = config.mutable_gpu_options();
  gpu_options->set_allow_growth(gpu_memory_allow_growth);

  (*config.mutable_device_count())["CPU"] = num_cpu_devices;

  // TODO(b/113217601): This is needed for EagerContext::runner_ to use a
  // threadpool, so that we avoid the possibility of running the runner_ in the
  // threadpool of GPU event mgr, as that can trigger more callbacks to be
  // scheduled on that same threadpool, causing a deadlock in cases where the
  // caller of event_mgr->ThenExecute() blocks on the completion of the callback
  // (as in the case of ConstOp kernel creation on GPU, which involves copying a
  // CPU tensor to GPU).
  // Setting a larger thread pool does not help with the Swift caller, as we use
  // a different TFE context for each thread of execution (for running graph
  // functions, and their send/recvs corountines).
  config.set_inter_op_parallelism_threads(1);

  TF_Buffer* ret = TF_NewBuffer();
  TF_CHECK_OK(MessageToBuffer(config, ret));
  return ret;
}

TF_Buffer* TF_CreateRunOptions(unsigned char enable_full_trace) {
  tensorflow::RunOptions options;
  if (enable_full_trace) {
    options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
  } else {
    options.set_trace_level(tensorflow::RunOptions::NO_TRACE);
  }
  TF_Buffer* ret = TF_NewBuffer();
  TF_CHECK_OK(MessageToBuffer(options, ret));
  return ret;
}

const char* TF_GraphDebugString(TF_Graph* graph, size_t* len) {
  tensorflow::mutex_lock c(graph->mu);
  const auto& debug_str = graph->graph.ToGraphDefDebug().DebugString();
  *len = debug_str.size();
  char* ret = static_cast<char*>(malloc(*len + 1));
  memcpy(ret, debug_str.c_str(), *len + 1);
  return ret;
}

char* TF_FunctionDebugString(TF_Function* func, size_t* len) {
  const auto& debug_str = func->fdef.DebugString();
  *len = debug_str.size();
  char* ret = static_cast<char*>(malloc(*len + 1));
  memcpy(ret, debug_str.c_str(), *len + 1);
  return ret;
}

// On success, returns a set of TF_Function instances from `text_proto` of
// GraphDef type. These functions must be deleted by calling TF_DeleteFunction.
//
// If `mutate_proto_func` is non-NULL, run it over each FunctionDef proto,
// before creating a TF_Function out of the possibly mutated proto.
static std::vector<UniqueFuncPtr> CreateFunctionsFromTextProto(
    const char* text_proto,
    std::function<void(FunctionDef*)>* mutate_proto_func, TF_Status* status) {
  tensorflow::GraphDef gdef;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(text_proto, &gdef)) {
    status->status = tensorflow::errors::Internal(
        "Invalid text proto for GraphDef: ", text_proto);
    return {};
  }
  const auto& fdef_lib = gdef.library();
  if (fdef_lib.gradient_size() > 0) {
    status->status = tensorflow::errors::Internal(
        "GradientDef is not supported in reading Dataset related functions: ",
        text_proto);
    return {};
  }
  std::vector<UniqueFuncPtr> ret;
  for (const FunctionDef& fdef : fdef_lib.function()) {
    // Make a copy so that we can mutate it.
    FunctionDef fdef_to_load = fdef;
    if (mutate_proto_func) {
      (*mutate_proto_func)(&fdef_to_load);
    }
    VLOG(1) << "Adding func to graph: " << fdef_to_load.DebugString();
    std::vector<char> binary_proto_buf(fdef_to_load.ByteSizeLong());
    fdef_to_load.SerializeToArray(binary_proto_buf.data(),
                                  binary_proto_buf.size());
    TF_Function* func = TF_FunctionImportFunctionDef(
        binary_proto_buf.data(), binary_proto_buf.size(), status);
    if (!status->status.ok()) return {};
    ret.push_back(UniqueFuncPtr(func, TF_DeleteFunction));
  }
  return ret;
}

//  On success, returns a newly created TF_Function instance encoding a dataset
//  node stack that returns a sequence of 3 floats, and sets `dataset_name` to
//  the created dataset name. The returned function must be deleted by calling
//  TF_DeleteFunction.
static UniqueFuncPtr CreateFakeDatasetFunction(std::string* dataset_name,
                                               TF_Status* status) {
  const char* func_def = R"PREFIX(
library {
  function {
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
    }
  }
}
)PREFIX";

  *dataset_name = "_make_dataset_d8de2712";
  auto functions = CreateFunctionsFromTextProto(
      func_def, /*mutate_proto_func*/ nullptr, status);
  DCHECK_EQ(functions.size(), 1);
  return std::move(functions[0]);
}

#if not defined(PLATFORM_WINDOWS)
//  On success, returns a set of TF_Function instances encoding a dataset
//  node stack that reads a Imagenet TFRecordFile dataset from `file_path`, and
//  sets `dataset_name` to the created dataset name. The returned functions must
//  be deleted by calling TF_DeleteFunction.
static std::vector<UniqueFuncPtr> CreateImagenetDatasetFunctions(
    const char* file_path, std::string* dataset_name, TF_Status* status) {
#if defined(PLATFORM_WINDOWS)
  status->status = tensorflow::errors::Unimplemented(
      "TF_MakeFileBasedIteratorGetNextWithDatasets in the experimental C API "
      "is not implemented for Windows");
  return std::vector<UniqueFuncPtr>();
#else
  const char* func_def = R"PREFIX(
library {
  function {
    signature {
      name: "tf_map_func_91295dea"
      input_arg {
        name: "arg0"
        type: DT_STRING
      }
      output_arg {
        name: "FlatMapDataset"
        type: DT_VARIANT
      }
      description: "A wrapper for Defun that facilitates shape inference."
      is_stateful: true
    }
    node_def {
      name: "flat_filenames/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: -1
          }
        }
      }
    }
    node_def {
      name: "flat_filenames"
      op: "Reshape"
      input: "arg0"
      input: "flat_filenames/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "TensorSliceDataset"
      op: "TensorSliceDataset"
      input: "flat_filenames:output:0"
      attr {
        key: "Toutput_types"
        value {
          list {
            type: DT_STRING
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
    node_def {
      name: "FlatMapDataset"
      op: "FlatMapDataset"
      input: "TensorSliceDataset:handle:0"
      attr {
        key: "Targuments"
        value {
          list {
          }
        }
      }
      attr {
        key: "f"
        value {
          func {
            name: "tf_map_func_0cc8c35b"
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
      attr {
        key: "output_types"
        value {
          list {
            type: DT_STRING
          }
        }
      }
    }
    ret {
      key: "FlatMapDataset"
      value: "FlatMapDataset:handle:0"
    }
  }
  function {
    signature {
      name: "tf_map_func_0cc8c35b"
      input_arg {
        name: "arg0"
        type: DT_STRING
      }
      output_arg {
        name: "TFRecordDataset"
        type: DT_VARIANT
      }
      description: "A wrapper for Defun that facilitates shape inference."
      is_stateful: true
    }
    node_def {
      name: "compression_type"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: ""
          }
        }
      }
    }
    node_def {
      name: "buffer_size"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 8388608
          }
        }
      }
    }
    node_def {
      name: "TFRecordDataset"
      op: "TFRecordDataset"
      input: "arg0"
      input: "compression_type:output:0"
      input: "buffer_size:output:0"
    }
    ret {
      key: "TFRecordDataset"
      value: "TFRecordDataset:handle:0"
    }
  }
  function {
    signature {
      name: "tf_map_func_74b6b15c"
      input_arg {
        name: "arg0"
        type: DT_STRING
      }
      output_arg {
        name: "Reshape_1"
        type: DT_FLOAT
      }
      output_arg {
        name: "sub_1"
        type: DT_INT32
      }
      description: "A wrapper for Defun that facilitates shape inference."
      is_stateful: true
    }
    node_def {
      name: "ParseSingleExample/key_image/class/label"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: -1
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape"
      op: "Reshape"
      input: "ParseSingleExample/key_image/class/label:output:0"
      input: "ParseSingleExample/Reshape/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "ParseSingleExample/key_image/class/text"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: ""
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape_1/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape_1"
      op: "Reshape"
      input: "ParseSingleExample/key_image/class/text:output:0"
      input: "ParseSingleExample/Reshape_1/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "ParseSingleExample/key_image/encoded"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: ""
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape_2/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape_2"
      op: "Reshape"
      input: "ParseSingleExample/key_image/encoded:output:0"
      input: "ParseSingleExample/Reshape_2/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "ParseSingleExample/key_image/format"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "jpeg"
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape_3/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "ParseSingleExample/Reshape_3"
      op: "Reshape"
      input: "ParseSingleExample/key_image/format:output:0"
      input: "ParseSingleExample/Reshape_3/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "ParseSingleExample/ParseSingleExample"
      op: "ParseSingleExample"
      input: "arg0"
      input: "ParseSingleExample/Reshape:output:0"
      input: "ParseSingleExample/Reshape_1:output:0"
      input: "ParseSingleExample/Reshape_2:output:0"
      input: "ParseSingleExample/Reshape_3:output:0"
      attr {
        key: "Tdense"
        value {
          list {
            type: DT_INT64
            type: DT_STRING
            type: DT_STRING
            type: DT_STRING
          }
        }
      }
      attr {
        key: "dense_keys"
        value {
          list {
            s: "image/class/label"
            s: "image/class/text"
            s: "image/encoded"
            s: "image/format"
          }
        }
      }
      attr {
        key: "dense_shapes"
        value {
          list {
            shape {
            }
            shape {
            }
            shape {
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "num_sparse"
        value {
          i: 5
        }
      }
      attr {
        key: "sparse_keys"
        value {
          list {
            s: "image/object/bbox/xmax"
            s: "image/object/bbox/xmin"
            s: "image/object/bbox/ymax"
            s: "image/object/bbox/ymin"
            s: "image/object/class/label"
          }
        }
      }
      attr {
        key: "sparse_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_FLOAT
            type: DT_FLOAT
            type: DT_FLOAT
            type: DT_INT64
          }
        }
      }
    }
    node_def {
      name: "Reshape/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "Reshape"
      op: "Reshape"
      input: "ParseSingleExample/ParseSingleExample:dense_values:2"
      input: "Reshape/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/Substr/pos"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "decode_image/Substr/len"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "decode_image/Substr"
      op: "Substr"
      input: "Reshape:output:0"
      input: "decode_image/Substr/pos:output:0"
      input: "decode_image/Substr/len:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/is_jpeg/Substr/pos"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "decode_image/is_jpeg/Substr/len"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "decode_image/is_jpeg/Substr"
      op: "Substr"
      input: "Reshape:output:0"
      input: "decode_image/is_jpeg/Substr/pos:output:0"
      input: "decode_image/is_jpeg/Substr/len:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/is_jpeg/Equal/y"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "\377\330\377"
          }
        }
      }
    }
    node_def {
      name: "decode_image/is_jpeg/Equal"
      op: "Equal"
      input: "decode_image/is_jpeg/Substr:output:0"
      input: "decode_image/is_jpeg/Equal/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/Switch"
      op: "Switch"
      input: "decode_image/is_jpeg/Equal:z:0"
      input: "decode_image/is_jpeg/Equal:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/switch_t"
      op: "Identity"
      input: "decode_image/cond_jpeg/Switch:output_true:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/switch_f"
      op: "Identity"
      input: "decode_image/cond_jpeg/Switch:output_false:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/pred_id"
      op: "Identity"
      input: "decode_image/is_jpeg/Equal:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/check_jpeg_channels/x"
      op: "Const"
      input: "^decode_image/cond_jpeg/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/check_jpeg_channels/y"
      op: "Const"
      input: "^decode_image/cond_jpeg/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 4
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/check_jpeg_channels"
      op: "NotEqual"
      input: "decode_image/cond_jpeg/check_jpeg_channels/x:output:0"
      input: "decode_image/cond_jpeg/check_jpeg_channels/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/Assert/Const"
      op: "Const"
      input: "^decode_image/cond_jpeg/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Channels must be in (None, 0, 1, 3) when decoding JPEG images"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/Assert/Assert/data_0"
      op: "Const"
      input: "^decode_image/cond_jpeg/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Channels must be in (None, 0, 1, 3) when decoding JPEG images"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/Assert/Assert"
      op: "Assert"
      input: "decode_image/cond_jpeg/check_jpeg_channels:z:0"
      input: "decode_image/cond_jpeg/Assert/Assert/data_0:output:0"
      attr {
        key: "T"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "summarize"
        value {
          i: 3
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/DecodeJpeg"
      op: "DecodeJpeg"
      input: "decode_image/cond_jpeg/DecodeJpeg/Switch:output_true:0"
      input: "^decode_image/cond_jpeg/Assert/Assert"
      attr {
        key: "acceptable_fraction"
        value {
          f: 1.0
        }
      }
      attr {
        key: "channels"
        value {
          i: 3
        }
      }
      attr {
        key: "dct_method"
        value {
          s: ""
        }
      }
      attr {
        key: "fancy_upscaling"
        value {
          b: true
        }
      }
      attr {
        key: "ratio"
        value {
          i: 1
        }
      }
      attr {
        key: "try_recover_truncated"
        value {
          b: false
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/DecodeJpeg/Switch"
      op: "Switch"
      input: "Reshape:output:0"
      input: "decode_image/cond_jpeg/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@Reshape"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/is_png/y"
      op: "Const"
      input: "^decode_image/cond_jpeg/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "\211PN"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/is_png"
      op: "Equal"
      input: "decode_image/cond_jpeg/is_png/Switch:output_false:0"
      input: "decode_image/cond_jpeg/is_png/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/is_png/Switch"
      op: "Switch"
      input: "decode_image/Substr:output:0"
      input: "decode_image/cond_jpeg/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@decode_image/Substr"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/Switch"
      op: "Switch"
      input: "decode_image/cond_jpeg/is_png:z:0"
      input: "decode_image/cond_jpeg/is_png:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/switch_t"
      op: "Identity"
      input: "decode_image/cond_jpeg/cond_png/Switch:output_true:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/switch_f"
      op: "Identity"
      input: "decode_image/cond_jpeg/cond_png/Switch:output_false:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/pred_id"
      op: "Identity"
      input: "decode_image/cond_jpeg/is_png:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/DecodePng"
      op: "DecodePng"
      input: "decode_image/cond_jpeg/cond_png/DecodePng/Switch_1:output_true:0"
      attr {
        key: "channels"
        value {
          i: 3
        }
      }
      attr {
        key: "dtype"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/DecodePng/Switch"
      op: "Switch"
      input: "Reshape:output:0"
      input: "decode_image/cond_jpeg/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@Reshape"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/DecodePng/Switch_1"
      op: "Switch"
      input: "decode_image/cond_jpeg/cond_png/DecodePng/Switch:output_false:0"
      input: "decode_image/cond_jpeg/cond_png/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@Reshape"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/is_gif/y"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "GIF"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/is_gif"
      op: "Equal"
      input: "decode_image/cond_jpeg/cond_png/is_gif/Switch:output_false:0"
      input: "decode_image/cond_jpeg/cond_png/is_gif/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/is_gif/Switch"
      op: "Switch"
      input: "decode_image/cond_jpeg/is_png/Switch:output_false:0"
      input: "decode_image/cond_jpeg/cond_png/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@decode_image/Substr"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Switch"
      op: "Switch"
      input: "decode_image/cond_jpeg/cond_png/is_gif:z:0"
      input: "decode_image/cond_jpeg/cond_png/is_gif:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/switch_t"
      op: "Identity"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Switch:output_true:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      op: "Identity"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Switch:output_false:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/pred_id"
      op: "Identity"
      input: "decode_image/cond_jpeg/cond_png/is_gif:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/x"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/y"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels"
      op: "NotEqual"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/x:output:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/x"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/y"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 4
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1"
      op: "NotEqual"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/x:output:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/LogicalAnd"
      op: "LogicalAnd"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels:z:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1:z:0"
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert/Const"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Channels must be in (None, 0, 3) when decoding GIF images"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert/data_0"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Channels must be in (None, 0, 3) when decoding GIF images"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert"
      op: "Assert"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/LogicalAnd:z:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert/data_0:output:0"
      attr {
        key: "T"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "summarize"
        value {
          i: 3
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif"
      op: "DecodeGif"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1:output_true:0"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert"
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch"
      op: "Switch"
      input: "decode_image/cond_jpeg/cond_png/DecodePng/Switch:output_false:0"
      input: "decode_image/cond_jpeg/cond_png/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@Reshape"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1"
      op: "Switch"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:output_false:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@Reshape"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Substr/pos"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Substr/len"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Substr"
      op: "Substr"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch:output_false:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Substr/pos:output:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Substr/len:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch"
      op: "Switch"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:output_false:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@Reshape"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/is_bmp/y"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "BM"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/is_bmp"
      op: "Equal"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Substr:output:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/is_bmp/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Const"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Unable to decode bytes as JPEG, PNG, GIF, or BMP"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert/data_0"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Unable to decode bytes as JPEG, PNG, GIF, or BMP"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert"
      op: "Assert"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/is_bmp:z:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert/data_0:output:0"
      attr {
        key: "T"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "summarize"
        value {
          i: 3
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_channels/x"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_channels/y"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/check_channels"
      op: "NotEqual"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_channels/x:output:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_channels/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Const"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Channels must be in (None, 0, 3) when decoding BMP images"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert/data_0"
      op: "Const"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Channels must be in (None, 0, 3) when decoding BMP images"
          }
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert"
      op: "Assert"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/check_channels:z:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert/data_0:output:0"
      attr {
        key: "T"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "summarize"
        value {
          i: 3
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeBmp"
      op: "DecodeBmp"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch:output_false:0"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert"
      input: "^decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert"
      attr {
        key: "channels"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/cond_gif/Merge"
      op: "Merge"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeBmp:image:0"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif:image:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/cond_png/Merge"
      op: "Merge"
      input: "decode_image/cond_jpeg/cond_png/cond_gif/Merge:output:0"
      input: "decode_image/cond_jpeg/cond_png/DecodePng:image:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "decode_image/cond_jpeg/Merge"
      op: "Merge"
      input: "decode_image/cond_jpeg/cond_png/Merge:output:0"
      input: "decode_image/cond_jpeg/DecodeJpeg:image:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "convert_image/Cast"
      op: "Cast"
      input: "decode_image/cond_jpeg/Merge:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "convert_image/y"
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
            }
            float_val: 0.00392156885937
          }
        }
      }
    }
    node_def {
      name: "convert_image"
      op: "Mul"
      input: "convert_image/Cast:y:0"
      input: "convert_image/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "Const"
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
                size: 1
              }
              dim {
                size: 1
              }
              dim {
                size: 4
              }
            }
            tensor_content: "\000\000\000\000\000\000\000\000\000\000\200?\000\000\200?"
          }
        }
      }
    }
    node_def {
      name: "distorted_bounding_box_crop/Shape"
      op: "Shape"
      input: "convert_image:z:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "distorted_bounding_box_crop/sample_distorted_bounding_box/SampleDistortedBoundingBoxV2/min_object_covered"
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
            }
            float_val: 0.10000000149
          }
        }
      }
    }
    node_def {
      name: "distorted_bounding_box_crop/sample_distorted_bounding_box/SampleDistortedBoundingBoxV2"
      op: "SampleDistortedBoundingBoxV2"
      input: "distorted_bounding_box_crop/Shape:output:0"
      input: "Const:output:0"
      input: "distorted_bounding_box_crop/sample_distorted_bounding_box/SampleDistortedBoundingBoxV2/min_object_covered:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "area_range"
        value {
          list {
            f: 0.0799999982119
            f: 1.0
          }
        }
      }
      attr {
        key: "aspect_ratio_range"
        value {
          list {
            f: 0.75
            f: 1.33333337307
          }
        }
      }
      attr {
        key: "max_attempts"
        value {
          i: 1
        }
      }
      attr {
        key: "seed"
        value {
          i: 0
        }
      }
      attr {
        key: "seed2"
        value {
          i: 0
        }
      }
      attr {
        key: "use_image_if_no_bounding_boxes"
        value {
          b: true
        }
      }
    }
    node_def {
      name: "distorted_bounding_box_crop/Slice"
      op: "Slice"
      input: "convert_image:z:0"
      input: "distorted_bounding_box_crop/sample_distorted_bounding_box/SampleDistortedBoundingBoxV2:begin:0"
      input: "distorted_bounding_box_crop/sample_distorted_bounding_box/SampleDistortedBoundingBoxV2:size:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "Shape"
      op: "Shape"
      input: "convert_image:z:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "Shape_1"
      op: "Shape"
      input: "distorted_bounding_box_crop/Slice:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "Equal"
      op: "Equal"
      input: "Shape:output:0"
      input: "Shape_1:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "Cast"
      op: "Cast"
      input: "Equal:z:0"
      attr {
        key: "DstT"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "Const_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "Sum"
      op: "Sum"
      input: "Cast:y:0"
      input: "Const_1:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "Tidx"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "keep_dims"
        value {
          b: false
        }
      }
    }
    node_def {
      name: "GreaterEqual/y"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "GreaterEqual"
      op: "GreaterEqual"
      input: "Sum:output:0"
      input: "GreaterEqual/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/Switch"
      op: "Switch"
      input: "GreaterEqual:z:0"
      input: "GreaterEqual:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/switch_t"
      op: "Identity"
      input: "cond/Switch:output_true:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/switch_f"
      op: "Identity"
      input: "cond/Switch:output_false:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/pred_id"
      op: "Identity"
      input: "GreaterEqual:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/Shape"
      op: "Shape"
      input: "cond/Shape/Switch:output_true:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/Shape/Switch"
      op: "Switch"
      input: "convert_image:z:0"
      input: "cond/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@convert_image"
          }
        }
      }
    }
    node_def {
      name: "cond/Cast"
      op: "Cast"
      input: "cond/Shape:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/strided_slice/stack"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice/stack_1"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice/stack_2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice"
      op: "StridedSlice"
      input: "cond/Cast:y:0"
      input: "cond/strided_slice/stack:output:0"
      input: "cond/strided_slice/stack_1:output:0"
      input: "cond/strided_slice/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/strided_slice_1/stack"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_1/stack_1"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_1/stack_2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_1"
      op: "StridedSlice"
      input: "cond/Cast:y:0"
      input: "cond/strided_slice_1/stack:output:0"
      input: "cond/strided_slice_1/stack_1:output:0"
      input: "cond/strided_slice_1/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/Greater"
      op: "Greater"
      input: "cond/strided_slice:output:0"
      input: "cond/strided_slice_1:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/cond/Switch"
      op: "Switch"
      input: "cond/Greater:z:0"
      input: "cond/Greater:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/cond/switch_t"
      op: "Identity"
      input: "cond/cond/Switch:output_true:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/cond/switch_f"
      op: "Identity"
      input: "cond/cond/Switch:output_false:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/cond/pred_id"
      op: "Identity"
      input: "cond/Greater:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice/stack"
      op: "Const"
      input: "^cond/cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice/stack_1"
      op: "Const"
      input: "^cond/cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice/stack_2"
      op: "Const"
      input: "^cond/cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice"
      op: "StridedSlice"
      input: "cond/cond/strided_slice/Switch:output_true:0"
      input: "cond/cond/strided_slice/stack:output:0"
      input: "cond/cond/strided_slice/stack_1:output:0"
      input: "cond/cond/strided_slice/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice/Switch"
      op: "Switch"
      input: "cond/Cast:y:0"
      input: "cond/cond/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@cond/Cast"
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_1/stack"
      op: "Const"
      input: "^cond/cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_1/stack_1"
      op: "Const"
      input: "^cond/cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_1/stack_2"
      op: "Const"
      input: "^cond/cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_1"
      op: "StridedSlice"
      input: "cond/cond/strided_slice/Switch:output_true:0"
      input: "cond/cond/strided_slice_1/stack:output:0"
      input: "cond/cond/strided_slice_1/stack_1:output:0"
      input: "cond/cond/strided_slice_1/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/cond/truediv"
      op: "RealDiv"
      input: "cond/cond/strided_slice:output:0"
      input: "cond/cond/strided_slice_1:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/cond/mul/y"
      op: "Const"
      input: "^cond/cond/switch_t"
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
            }
            float_val: 224.0
          }
        }
      }
    }
    node_def {
      name: "cond/cond/mul"
      op: "Mul"
      input: "cond/cond/truediv:z:0"
      input: "cond/cond/mul/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/cond/Cast/x/1"
      op: "Const"
      input: "^cond/cond/switch_t"
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
            }
            float_val: 224.0
          }
        }
      }
    }
    node_def {
      name: "cond/cond/Cast/x"
      op: "Pack"
      input: "cond/cond/mul:z:0"
      input: "cond/cond/Cast/x/1:output:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "axis"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "cond/cond/Cast"
      op: "Cast"
      input: "cond/cond/Cast/x:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_2/stack"
      op: "Const"
      input: "^cond/cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_2/stack_1"
      op: "Const"
      input: "^cond/cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_2/stack_2"
      op: "Const"
      input: "^cond/cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_2"
      op: "StridedSlice"
      input: "cond/cond/strided_slice_2/Switch:output_false:0"
      input: "cond/cond/strided_slice_2/stack:output:0"
      input: "cond/cond/strided_slice_2/stack_1:output:0"
      input: "cond/cond/strided_slice_2/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_2/Switch"
      op: "Switch"
      input: "cond/Cast:y:0"
      input: "cond/cond/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@cond/Cast"
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_3/stack"
      op: "Const"
      input: "^cond/cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_3/stack_1"
      op: "Const"
      input: "^cond/cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_3/stack_2"
      op: "Const"
      input: "^cond/cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/cond/strided_slice_3"
      op: "StridedSlice"
      input: "cond/cond/strided_slice_2/Switch:output_false:0"
      input: "cond/cond/strided_slice_3/stack:output:0"
      input: "cond/cond/strided_slice_3/stack_1:output:0"
      input: "cond/cond/strided_slice_3/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/cond/truediv_1"
      op: "RealDiv"
      input: "cond/cond/strided_slice_2:output:0"
      input: "cond/cond/strided_slice_3:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/cond/mul_1/y"
      op: "Const"
      input: "^cond/cond/switch_f"
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
            }
            float_val: 224.0
          }
        }
      }
    }
    node_def {
      name: "cond/cond/mul_1"
      op: "Mul"
      input: "cond/cond/truediv_1:z:0"
      input: "cond/cond/mul_1/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/cond/Cast_1/x/0"
      op: "Const"
      input: "^cond/cond/switch_f"
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
            }
            float_val: 224.0
          }
        }
      }
    }
    node_def {
      name: "cond/cond/Cast_1/x"
      op: "Pack"
      input: "cond/cond/Cast_1/x/0:output:0"
      input: "cond/cond/mul_1:z:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "axis"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "cond/cond/Cast_1"
      op: "Cast"
      input: "cond/cond/Cast_1/x:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/cond/Merge"
      op: "Merge"
      input: "cond/cond/Cast_1:y:0"
      input: "cond/cond/Cast:y:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/ResizeBicubic/images"
      op: "Pack"
      input: "cond/Shape/Switch:output_true:0"
      attr {
        key: "N"
        value {
          i: 1
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "axis"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "cond/ResizeBicubic"
      op: "ResizeBicubic"
      input: "cond/ResizeBicubic/images:output:0"
      input: "cond/cond/Merge:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "align_corners"
        value {
          b: false
        }
      }
    }
    node_def {
      name: "cond/strided_slice_2/stack"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_2/stack_1"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_2/stack_2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_2"
      op: "StridedSlice"
      input: "cond/ResizeBicubic:resized_images:0"
      input: "cond/strided_slice_2/stack:output:0"
      input: "cond/strided_slice_2/stack_1:output:0"
      input: "cond/strided_slice_2/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/Shape_1"
      op: "Shape"
      input: "cond/strided_slice_2:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/strided_slice_3/stack"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_3/stack_1"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_3/stack_2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_3"
      op: "StridedSlice"
      input: "cond/Shape_1:output:0"
      input: "cond/strided_slice_3/stack:output:0"
      input: "cond/strided_slice_3/stack_1:output:0"
      input: "cond/strided_slice_3/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/Shape_2"
      op: "Shape"
      input: "cond/strided_slice_2:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/strided_slice_4/stack"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_4/stack_1"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_4/stack_2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_4"
      op: "StridedSlice"
      input: "cond/Shape_2:output:0"
      input: "cond/strided_slice_4/stack:output:0"
      input: "cond/strided_slice_4/stack_1:output:0"
      input: "cond/strided_slice_4/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/sub/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 224
          }
        }
      }
    }
    node_def {
      name: "cond/sub"
      op: "Sub"
      input: "cond/strided_slice_3:output:0"
      input: "cond/sub/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/add/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/add"
      op: "Add"
      input: "cond/sub:z:0"
      input: "cond/add/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/truediv/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/truediv/Cast"
      op: "Cast"
      input: "cond/add:z:0"
      attr {
        key: "DstT"
        value {
          type: DT_DOUBLE
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/truediv/Cast_1"
      op: "Cast"
      input: "cond/truediv/y:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_DOUBLE
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/truediv"
      op: "RealDiv"
      input: "cond/truediv/Cast:y:0"
      input: "cond/truediv/Cast_1:y:0"
      attr {
        key: "T"
        value {
          type: DT_DOUBLE
        }
      }
    }
    node_def {
      name: "cond/sub_1/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 224
          }
        }
      }
    }
    node_def {
      name: "cond/sub_1"
      op: "Sub"
      input: "cond/strided_slice_4:output:0"
      input: "cond/sub_1/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/add_1/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/add_1"
      op: "Add"
      input: "cond/sub_1:z:0"
      input: "cond/add_1/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/truediv_1/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/truediv_1/Cast"
      op: "Cast"
      input: "cond/add_1:z:0"
      attr {
        key: "DstT"
        value {
          type: DT_DOUBLE
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/truediv_1/Cast_1"
      op: "Cast"
      input: "cond/truediv_1/y:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_DOUBLE
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/truediv_1"
      op: "RealDiv"
      input: "cond/truediv_1/Cast:y:0"
      input: "cond/truediv_1/Cast_1:y:0"
      attr {
        key: "T"
        value {
          type: DT_DOUBLE
        }
      }
    }
    node_def {
      name: "cond/Shape_3"
      op: "Shape"
      input: "cond/strided_slice_2:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/Rank"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "cond/Equal/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "cond/Equal"
      op: "Equal"
      input: "cond/Rank:output:0"
      input: "cond/Equal/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/Assert/Const"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Rank of image must be equal to 3."
          }
        }
      }
    }
    node_def {
      name: "cond/Assert/Assert/data_0"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Rank of image must be equal to 3."
          }
        }
      }
    }
    node_def {
      name: "cond/Assert/Assert"
      op: "Assert"
      input: "cond/Equal:z:0"
      input: "cond/Assert/Assert/data_0:output:0"
      attr {
        key: "T"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "summarize"
        value {
          i: 3
        }
      }
    }
    node_def {
      name: "cond/strided_slice_5/stack"
      op: "Const"
      input: "^cond/Assert/Assert"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_5/stack_1"
      op: "Const"
      input: "^cond/Assert/Assert"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 3
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_5/stack_2"
      op: "Const"
      input: "^cond/Assert/Assert"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_5"
      op: "StridedSlice"
      input: "cond/Shape_3:output:0"
      input: "cond/strided_slice_5/stack:output:0"
      input: "cond/strided_slice_5/stack_1:output:0"
      input: "cond/strided_slice_5/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/stack/0"
      op: "Const"
      input: "^cond/Assert/Assert"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 224
          }
        }
      }
    }
    node_def {
      name: "cond/stack/1"
      op: "Const"
      input: "^cond/Assert/Assert"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 224
          }
        }
      }
    }
    node_def {
      name: "cond/stack"
      op: "Pack"
      input: "cond/stack/0:output:0"
      input: "cond/stack/1:output:0"
      input: "cond/strided_slice_5:output:0"
      attr {
        key: "N"
        value {
          i: 3
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "axis"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "cond/strided_slice_6/stack"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_6/stack_1"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_6/stack_2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_6"
      op: "StridedSlice"
      input: "cond/Shape_3:output:0"
      input: "cond/strided_slice_6/stack:output:0"
      input: "cond/strided_slice_6/stack_1:output:0"
      input: "cond/strided_slice_6/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/GreaterEqual/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 224
          }
        }
      }
    }
    node_def {
      name: "cond/GreaterEqual"
      op: "GreaterEqual"
      input: "cond/strided_slice_6:output:0"
      input: "cond/GreaterEqual/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/strided_slice_7/stack"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_7/stack_1"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 2
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_7/stack_2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_7"
      op: "StridedSlice"
      input: "cond/Shape_3:output:0"
      input: "cond/strided_slice_7/stack:output:0"
      input: "cond/strided_slice_7/stack_1:output:0"
      input: "cond/strided_slice_7/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/GreaterEqual_1/y"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 224
          }
        }
      }
    }
    node_def {
      name: "cond/GreaterEqual_1"
      op: "GreaterEqual"
      input: "cond/strided_slice_7:output:0"
      input: "cond/GreaterEqual_1/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/LogicalAnd"
      op: "LogicalAnd"
      input: "cond/GreaterEqual:z:0"
      input: "cond/GreaterEqual_1:z:0"
    }
    node_def {
      name: "cond/Assert_1/Const"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Crop size greater than the image size."
          }
        }
      }
    }
    node_def {
      name: "cond/Assert_1/Assert/data_0"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "Crop size greater than the image size."
          }
        }
      }
    }
    node_def {
      name: "cond/Assert_1/Assert"
      op: "Assert"
      input: "cond/LogicalAnd:z:0"
      input: "cond/Assert_1/Assert/data_0:output:0"
      attr {
        key: "T"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "summarize"
        value {
          i: 3
        }
      }
    }
    node_def {
      name: "cond/stack_1/2"
      op: "Const"
      input: "^cond/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_DOUBLE
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_DOUBLE
            tensor_shape {
            }
            double_val: 0.0
          }
        }
      }
    }
    node_def {
      name: "cond/stack_1"
      op: "Pack"
      input: "cond/truediv:z:0"
      input: "cond/truediv_1:z:0"
      input: "cond/stack_1/2:output:0"
      attr {
        key: "N"
        value {
          i: 3
        }
      }
      attr {
        key: "T"
        value {
          type: DT_DOUBLE
        }
      }
      attr {
        key: "axis"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "cond/ToInt32"
      op: "Cast"
      input: "cond/stack_1:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_DOUBLE
        }
      }
    }
    node_def {
      name: "cond/Slice"
      op: "Slice"
      input: "cond/strided_slice_2:output:0"
      input: "cond/ToInt32:y:0"
      input: "cond/stack:output:0"
      input: "^cond/Assert_1/Assert"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "cond/Reshape"
      op: "Reshape"
      input: "cond/Slice:output:0"
      input: "cond/stack:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "cond/ResizeBicubic_1/images"
      op: "Pack"
      input: "cond/ResizeBicubic_1/images/Switch:output_false:0"
      attr {
        key: "N"
        value {
          i: 1
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "axis"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "cond/ResizeBicubic_1/images/Switch"
      op: "Switch"
      input: "distorted_bounding_box_crop/Slice:output:0"
      input: "cond/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@distorted_bounding_box_crop/Slice"
          }
        }
      }
    }
    node_def {
      name: "cond/ResizeBicubic_1/size"
      op: "Const"
      input: "^cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 2
              }
            }
            tensor_content: "\340\000\000\000\340\000\000\000"
          }
        }
      }
    }
    node_def {
      name: "cond/ResizeBicubic_1"
      op: "ResizeBicubic"
      input: "cond/ResizeBicubic_1/images:output:0"
      input: "cond/ResizeBicubic_1/size:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "align_corners"
        value {
          b: false
        }
      }
    }
    node_def {
      name: "cond/strided_slice_8/stack"
      op: "Const"
      input: "^cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_8/stack_1"
      op: "Const"
      input: "^cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_8/stack_2"
      op: "Const"
      input: "^cond/switch_f"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "cond/strided_slice_8"
      op: "StridedSlice"
      input: "cond/ResizeBicubic_1:resized_images:0"
      input: "cond/strided_slice_8/stack:output:0"
      input: "cond/strided_slice_8/stack_1:output:0"
      input: "cond/strided_slice_8/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "cond/Merge"
      op: "Merge"
      input: "cond/strided_slice_8:output:0"
      input: "cond/Reshape:output:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "Const_2"
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
                size: 1
              }
              dim {
                size: 1
              }
              dim {
                size: 3
              }
            }
            tensor_content: "\354Q\370>\325x\351>;\337\317>"
          }
        }
      }
    }
    node_def {
      name: "sub"
      op: "Sub"
      input: "cond/Merge:output:0"
      input: "Const_2:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "Const_3"
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
                size: 1
              }
              dim {
                size: 1
              }
              dim {
                size: 3
              }
            }
            tensor_content: "\372~j>B`e>fff>"
          }
        }
      }
    }
    node_def {
      name: "truediv"
      op: "RealDiv"
      input: "sub:z:0"
      input: "Const_3:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "random_flip_left_right/control_dependency"
      op: "Identity"
      input: "truediv:z:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@truediv"
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/random_uniform/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/random_uniform/min"
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
            }
            float_val: 0.0
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/random_uniform/max"
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
            }
            float_val: 1.0
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/random_uniform/RandomUniform"
      op: "RandomUniform"
      input: "random_flip_left_right/random_uniform/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "seed"
        value {
          i: 0
        }
      }
      attr {
        key: "seed2"
        value {
          i: 0
        }
      }
    }
    node_def {
      name: "random_flip_left_right/random_uniform/sub"
      op: "Sub"
      input: "random_flip_left_right/random_uniform/max:output:0"
      input: "random_flip_left_right/random_uniform/min:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "random_flip_left_right/random_uniform/mul"
      op: "Mul"
      input: "random_flip_left_right/random_uniform/RandomUniform:output:0"
      input: "random_flip_left_right/random_uniform/sub:z:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "random_flip_left_right/random_uniform"
      op: "Add"
      input: "random_flip_left_right/random_uniform/mul:z:0"
      input: "random_flip_left_right/random_uniform/min:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "random_flip_left_right/Less/y"
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
            }
            float_val: 0.5
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/Less"
      op: "Less"
      input: "random_flip_left_right/random_uniform:z:0"
      input: "random_flip_left_right/Less/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "random_flip_left_right/Switch"
      op: "Switch"
      input: "random_flip_left_right/Less:z:0"
      input: "random_flip_left_right/Less:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "random_flip_left_right/switch_t"
      op: "Identity"
      input: "random_flip_left_right/Switch:output_true:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "random_flip_left_right/switch_f"
      op: "Identity"
      input: "random_flip_left_right/Switch:output_false:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "random_flip_left_right/pred_id"
      op: "Identity"
      input: "random_flip_left_right/Less:z:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "random_flip_left_right/ReverseV2/axis"
      op: "Const"
      input: "^random_flip_left_right/switch_t"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/ReverseV2"
      op: "ReverseV2"
      input: "random_flip_left_right/ReverseV2/Switch:output_true:0"
      input: "random_flip_left_right/ReverseV2/axis:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "Tidx"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "random_flip_left_right/ReverseV2/Switch"
      op: "Switch"
      input: "random_flip_left_right/control_dependency:output:0"
      input: "random_flip_left_right/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@truediv"
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/Switch_1"
      op: "Switch"
      input: "random_flip_left_right/control_dependency:output:0"
      input: "random_flip_left_right/pred_id:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@truediv"
          }
        }
      }
    }
    node_def {
      name: "random_flip_left_right/Merge"
      op: "Merge"
      input: "random_flip_left_right/Switch_1:output_false:0"
      input: "random_flip_left_right/ReverseV2:output:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    node_def {
      name: "Reshape_1/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 3
              }
            }
            tensor_content: "\340\000\000\000\340\000\000\000\003\000\000\000"
          }
        }
      }
    }
    node_def {
      name: "Reshape_1"
      op: "Reshape"
      input: "random_flip_left_right/Merge:output:0"
      input: "Reshape_1/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "Reshape_2/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "Reshape_2"
      op: "Reshape"
      input: "ParseSingleExample/ParseSingleExample:dense_values:0"
      input: "Reshape_2/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "Cast_1"
      op: "Cast"
      input: "Reshape_2:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_INT64
        }
      }
    }
    node_def {
      name: "sub_1/y"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "sub_1"
      op: "Sub"
      input: "Cast_1:y:0"
      input: "sub_1/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    ret {
      key: "Reshape_1"
      value: "Reshape_1:output:0"
    }
    ret {
      key: "sub_1"
      value: "sub_1:z:0"
    }
  }
  function {
    signature {
      name: "tf_predicate_7089b845"
      input_arg {
        name: "arg0"
        type: DT_FLOAT
      }
      input_arg {
        name: "arg1"
        type: DT_INT32
      }
      input_arg {
        name: "Equal/Placeholder"
        type: DT_INT64
      }
      output_arg {
        name: "Equal"
        type: DT_BOOL
      }
      description: "A wrapper for Defun that facilitates shape inference."
    }
    node_def {
      name: "Shape"
      op: "Shape"
      input: "arg0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT64
        }
      }
    }
    node_def {
      name: "strided_slice/stack"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "strided_slice/stack_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "strided_slice/stack_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "strided_slice"
      op: "StridedSlice"
      input: "Shape:output:0"
      input: "strided_slice/stack:output:0"
      input: "strided_slice/stack_1:output:0"
      input: "strided_slice/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "Equal"
      op: "Equal"
      input: "strided_slice:output:0"
      input: "Equal/Placeholder"
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
    }
    ret {
      key: "Equal"
      value: "Equal:z:0"
    }
  }
  function {
    signature {
      name: "_make_dataset_5fa5e1f4"
      output_arg {
        name: "PrefetchDataset_1"
        type: DT_VARIANT
      }
      is_stateful: true
    }
    node_def {
      name: "TensorSliceDataset/MatchingFiles/pattern"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "$(DATA_DIR)"
          }
        }
      }
    }
    node_def {
      name: "TensorSliceDataset/MatchingFiles"
      op: "MatchingFiles"
      input: "TensorSliceDataset/MatchingFiles/pattern:output:0"
    }
    node_def {
      name: "TensorSliceDataset"
      op: "TensorSliceDataset"
      input: "TensorSliceDataset/MatchingFiles:filenames:0"
      attr {
        key: "Toutput_types"
        value {
          list {
            type: DT_STRING
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
    node_def {
      name: "ShuffleDataset/MatchingFiles/pattern"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "$(DATA_DIR)"
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/MatchingFiles"
      op: "MatchingFiles"
      input: "ShuffleDataset/MatchingFiles/pattern:output:0"
    }
    node_def {
      name: "ShuffleDataset/Shape"
      op: "Shape"
      input: "ShuffleDataset/MatchingFiles:filenames:0"
      attr {
        key: "T"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT64
        }
      }
    }
    node_def {
      name: "ShuffleDataset/strided_slice/stack"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/strided_slice/stack_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/strided_slice/stack_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/strided_slice"
      op: "StridedSlice"
      input: "ShuffleDataset/Shape:output:0"
      input: "ShuffleDataset/strided_slice/stack:output:0"
      input: "ShuffleDataset/strided_slice/stack_1:output:0"
      input: "ShuffleDataset/strided_slice/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "ShuffleDataset/Maximum/y"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 1
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/Maximum"
      op: "Maximum"
      input: "ShuffleDataset/strided_slice:output:0"
      input: "ShuffleDataset/Maximum/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
    }
    node_def {
      name: "ShuffleDataset/seed"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/seed2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset"
      op: "ShuffleDataset"
      input: "TensorSliceDataset:handle:0"
      input: "ShuffleDataset/Maximum:z:0"
      input: "ShuffleDataset/seed:output:0"
      input: "ShuffleDataset/seed2:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "reshuffle_each_iteration"
        value {
          b: true
        }
      }
    }
    node_def {
      name: "ShuffleDataset_1/buffer_size"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 1024
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset_1/seed_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset_1/seed2_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset_1"
      op: "ShuffleDataset"
      input: "ShuffleDataset:handle:0"
      input: "ShuffleDataset_1/buffer_size:output:0"
      input: "ShuffleDataset_1/seed_1:output:0"
      input: "ShuffleDataset_1/seed2_1:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "reshuffle_each_iteration"
        value {
          b: true
        }
      }
    }
    node_def {
      name: "RepeatDataset/count"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: -1
          }
        }
      }
    }
    node_def {
      name: "RepeatDataset"
      op: "RepeatDataset"
      input: "ShuffleDataset_1:handle:0"
      input: "RepeatDataset/count:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_STRING
          }
        }
      }
    }
    node_def {
      name: "ExperimentalParallelInterleaveDataset/cycle_length"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 8
          }
        }
      }
    }
    node_def {
      name: "ExperimentalParallelInterleaveDataset/block_length"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 1
          }
        }
      }
    }
    node_def {
      name: "ExperimentalParallelInterleaveDataset/sloppy"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_BOOL
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_BOOL
            tensor_shape {
            }
            bool_val: true
          }
        }
      }
    }
    node_def {
      name: "ExperimentalParallelInterleaveDataset/buffer_output_elements"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 2
          }
        }
      }
    }
    node_def {
      name: "ExperimentalParallelInterleaveDataset/prefetch_input_elements"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 16
          }
        }
      }
    }
    node_def {
      name: "ExperimentalParallelInterleaveDataset"
      op: "ExperimentalParallelInterleaveDataset"
      input: "RepeatDataset:handle:0"
      input: "ExperimentalParallelInterleaveDataset/cycle_length:output:0"
      input: "ExperimentalParallelInterleaveDataset/block_length:output:0"
      input: "ExperimentalParallelInterleaveDataset/sloppy:output:0"
      input: "ExperimentalParallelInterleaveDataset/buffer_output_elements:output:0"
      input: "ExperimentalParallelInterleaveDataset/prefetch_input_elements:output:0"
      attr {
        key: "Targuments"
        value {
          list {
          }
        }
      }
      attr {
        key: "f"
        value {
          func {
            name: "tf_map_func_91295dea"
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
      attr {
        key: "output_types"
        value {
          list {
            type: DT_STRING
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset_2/buffer_size_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 1024
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset_2/seed_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset_2/seed2_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset_2"
      op: "ShuffleDataset"
      input: "ExperimentalParallelInterleaveDataset:handle:0"
      input: "ShuffleDataset_2/buffer_size_1:output:0"
      input: "ShuffleDataset_2/seed_2:output:0"
      input: "ShuffleDataset_2/seed2_2:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_STRING
          }
        }
      }
      attr {
        key: "reshuffle_each_iteration"
        value {
          b: true
        }
      }
    }
    node_def {
      name: "ParallelMapDataset/num_parallel_calls"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 64
          }
        }
      }
    }
    node_def {
      name: "ParallelMapDataset"
      op: "ParallelMapDataset"
      input: "ShuffleDataset_2:handle:0"
      input: "ParallelMapDataset/num_parallel_calls:output:0"
      attr {
        key: "Targuments"
        value {
          list {
          }
        }
      }
      attr {
        key: "f"
        value {
          func {
            name: "tf_map_func_74b6b15c"
          }
        }
      }
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 224
              }
              dim {
                size: 224
              }
              dim {
                size: 3
              }
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "PrefetchDataset/buffer_size_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 64
          }
        }
      }
    }
    node_def {
      name: "PrefetchDataset"
      op: "PrefetchDataset"
      input: "ParallelMapDataset:handle:0"
      input: "PrefetchDataset/buffer_size_2:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 224
              }
              dim {
                size: 224
              }
              dim {
                size: 3
              }
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "BatchDataset/batch_size"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 64
          }
        }
      }
    }
    node_def {
      name: "BatchDataset"
      op: "BatchDataset"
      input: "PrefetchDataset:handle:0"
      input: "BatchDataset/batch_size:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: -1
              }
              dim {
                size: 224
              }
              dim {
                size: 224
              }
              dim {
                size: 3
              }
            }
            shape {
              dim {
                size: -1
              }
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "FilterDataset/batch_size_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 64
          }
        }
      }
    }
    node_def {
      name: "FilterDataset"
      op: "FilterDataset"
      input: "BatchDataset:handle:0"
      input: "FilterDataset/batch_size_1:output:0"
      attr {
        key: "Targuments"
        value {
          list {
            type: DT_INT64
          }
        }
      }
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: -1
              }
              dim {
                size: 224
              }
              dim {
                size: 224
              }
              dim {
                size: 3
              }
            }
            shape {
              dim {
                size: -1
              }
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
      attr {
        key: "predicate"
        value {
          func {
            name: "tf_predicate_7089b845"
          }
        }
      }
    }
    node_def {
      name: "PrefetchDataset_1/buffer_size_3"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 2
          }
        }
      }
    }
    node_def {
      name: "PrefetchDataset_1"
      op: "PrefetchDataset"
      input: "FilterDataset:handle:0"
      input: "PrefetchDataset_1/buffer_size_3:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 64
              }
              dim {
                size: 224
              }
              dim {
                size: 224
              }
              dim {
                size: 3
              }
            }
            shape {
              dim {
                size: 64
              }
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    ret {
      key: "PrefetchDataset_1"
      value: "PrefetchDataset_1:handle:0"
    }
  }
}
)PREFIX";

  *dataset_name = "_make_dataset_5fa5e1f4";
  std::function<void(FunctionDef*)> mutate_proto_func =
      [dataset_name, file_path](FunctionDef* fdef) {
        VLOG(1) << "Processsing function " << fdef->DebugString();
        if (std::string(fdef->signature().name()) != *dataset_name) return;
        // Change the input file pattern to `file_path`.
        bool found = false;
        for (auto& node_def : *fdef->mutable_node_def()) {
          if (node_def.name() != "TensorSliceDataset/MatchingFiles/pattern" &&
              node_def.name() != "ShuffleDataset/MatchingFiles/pattern")
            continue;
          DCHECK_EQ(node_def.op(), "Const");
          DCHECK_GT(node_def.attr().count("value"), 0);
          found = true;
          DCHECK_EQ(node_def.attr().at("value").tensor().string_val(0),
                    "$(DATA_DIR)");
          VLOG(1) << "Setting the value of node_def "
                     "TensorSliceDataset/MatchingFiles/pattern to "
                  << file_path;
          auto* tensor = (*node_def.mutable_attr())["value"].mutable_tensor();
          tensor->clear_string_val();
          tensor->add_string_val(file_path);
        }
        VLOG(1) << "Rewrote function to " << fdef->DebugString();
        DCHECK(found);
      };
  return CreateFunctionsFromTextProto(func_def, &mutate_proto_func, status);
#endif
}
#endif

#if not defined(PLATFORM_WINDOWS)
//  On success, returns a set of TF_Function instances encoding a dataset
//  node stack that reads an MNIST file dataset from `file_path`, and
//  sets `dataset_name` to the created dataset name. The returned functions must
//  be deleted by calling TF_DeleteFunction.
static std::vector<UniqueFuncPtr> CreateMNISTDatasetFunctions(
    const char* file_path, int batch_size, std::string* dataset_name,
    TF_Status* status) {
#if defined(PLATFORM_WINDOWS)
  status->status = tensorflow::errors::Unimplemented(
      "TF_MakeFileBasedIteratorGetNextWithDatasets in the experimental C API "
      "is not implemented for Windows");
  return nullptr;
#else
  const char* func_def = R"PREFIX(
library {
  function {
    signature {
      name: "tf_map_func_521bfd08"
      input_arg {
        name: "arg0"
        type: DT_STRING
      }
      output_arg {
        name: "truediv"
        type: DT_FLOAT
      }
      description: "A wrapper for Defun that facilitates shape inference."
    }
    node_def {
      name: "DecodeRaw"
      op: "DecodeRaw"
      input: "arg0"
      attr {
        key: "little_endian"
        value {
          b: true
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "Cast"
      op: "Cast"
      input: "DecodeRaw:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "Reshape/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 784
          }
        }
      }
    }
    node_def {
      name: "Reshape"
      op: "Reshape"
      input: "Cast:y:0"
      input: "Reshape/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "truediv/y"
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
            }
            float_val: 255.0
          }
        }
      }
    }
    node_def {
      name: "truediv"
      op: "RealDiv"
      input: "Reshape:output:0"
      input: "truediv/y:output:0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
    }
    ret {
      key: "truediv"
      value: "truediv:z:0"
    }
  }
  function {
    signature {
      name: "tf_map_func_9a08860d"
      input_arg {
        name: "arg0"
        type: DT_STRING
      }
      output_arg {
        name: "ToInt32"
        type: DT_INT32
      }
      description: "A wrapper for Defun that facilitates shape inference."
    }
    node_def {
      name: "DecodeRaw"
      op: "DecodeRaw"
      input: "arg0"
      attr {
        key: "little_endian"
        value {
          b: true
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_UINT8
        }
      }
    }
    node_def {
      name: "Reshape/shape"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
              }
            }
          }
        }
      }
    }
    node_def {
      name: "Reshape"
      op: "Reshape"
      input: "DecodeRaw:output:0"
      input: "Reshape/shape:output:0"
      attr {
        key: "T"
        value {
          type: DT_UINT8
        }
      }
      attr {
        key: "Tshape"
        value {
          type: DT_INT32
        }
      }
    }
    node_def {
      name: "ToInt32"
      op: "Cast"
      input: "Reshape:output:0"
      attr {
        key: "DstT"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "SrcT"
        value {
          type: DT_UINT8
        }
      }
    }
    ret {
      key: "ToInt32"
      value: "ToInt32:y:0"
    }
  }
  function {
    signature {
      name: "tf_predicate_7089b845"
      input_arg {
        name: "arg0"
        type: DT_FLOAT
      }
      input_arg {
        name: "arg1"
        type: DT_INT32
      }
      input_arg {
        name: "Equal/Placeholder"
        type: DT_INT64
      }
      output_arg {
        name: "Equal"
        type: DT_BOOL
      }
      description: "A wrapper for Defun that facilitates shape inference."
    }
    node_def {
      name: "Shape"
      op: "Shape"
      input: "arg0"
      attr {
        key: "T"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "out_type"
        value {
          type: DT_INT64
        }
      }
    }
    node_def {
      name: "strided_slice/stack"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "strided_slice/stack_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "strided_slice/stack_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
              dim {
                size: 1
              }
            }
            int_val: 1
          }
        }
      }
    }
    node_def {
      name: "strided_slice"
      op: "StridedSlice"
      input: "Shape:output:0"
      input: "strided_slice/stack:output:0"
      input: "strided_slice/stack_1:output:0"
      input: "strided_slice/stack_2:output:0"
      attr {
        key: "Index"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "begin_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "ellipsis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "end_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "new_axis_mask"
        value {
          i: 0
        }
      }
      attr {
        key: "shrink_axis_mask"
        value {
          i: 1
        }
      }
    }
    node_def {
      name: "Equal"
      op: "Equal"
      input: "strided_slice:output:0"
      input: "Equal/Placeholder"
      attr {
        key: "T"
        value {
          type: DT_INT64
        }
      }
    }
    ret {
      key: "Equal"
      value: "Equal:z:0"
    }
  }
  function {
    signature {
      name: "_make_dataset_2451e43a"
      output_arg {
        name: "FilterDataset"
        type: DT_VARIANT
      }
      is_stateful: true
    }
    node_def {
      name: "FixedLengthRecordDataset/filenames"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "$(DATA_DIR)/train-images-idx3-ubyte"
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset/header_bytes"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 16
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset/record_bytes"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 784
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset/footer_bytes"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset/buffer_size"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 262144
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset"
      op: "FixedLengthRecordDataset"
      input: "FixedLengthRecordDataset/filenames:output:0"
      input: "FixedLengthRecordDataset/header_bytes:output:0"
      input: "FixedLengthRecordDataset/record_bytes:output:0"
      input: "FixedLengthRecordDataset/footer_bytes:output:0"
      input: "FixedLengthRecordDataset/buffer_size:output:0"
    }
    node_def {
      name: "MapDataset"
      op: "MapDataset"
      input: "FixedLengthRecordDataset:handle:0"
      attr {
        key: "Targuments"
        value {
          list {
          }
        }
      }
      attr {
        key: "f"
        value {
          func {
            name: "tf_map_func_521bfd08"
          }
        }
      }
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 784
              }
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset_1/filenames_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: "$(DATA_DIR)/train-labels-idx1-ubyte"
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset_1/header_bytes_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 8
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset_1/record_bytes_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 1
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset_1/footer_bytes_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset_1/buffer_size_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 262144
          }
        }
      }
    }
    node_def {
      name: "FixedLengthRecordDataset_1"
      op: "FixedLengthRecordDataset"
      input: "FixedLengthRecordDataset_1/filenames_1:output:0"
      input: "FixedLengthRecordDataset_1/header_bytes_1:output:0"
      input: "FixedLengthRecordDataset_1/record_bytes_1:output:0"
      input: "FixedLengthRecordDataset_1/footer_bytes_1:output:0"
      input: "FixedLengthRecordDataset_1/buffer_size_1:output:0"
    }
    node_def {
      name: "MapDataset_1"
      op: "MapDataset"
      input: "FixedLengthRecordDataset_1:handle:0"
      attr {
        key: "Targuments"
        value {
          list {
          }
        }
      }
      attr {
        key: "f"
        value {
          func {
            name: "tf_map_func_9a08860d"
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
      attr {
        key: "output_types"
        value {
          list {
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "ZipDataset"
      op: "ZipDataset"
      input: "MapDataset:handle:0"
      input: "MapDataset_1:handle:0"
      attr {
        key: "N"
        value {
          i: 2
        }
      }
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 784
              }
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "CacheDataset/filename"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_STRING
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_STRING
            tensor_shape {
            }
            string_val: ""
          }
        }
      }
    }
    node_def {
      name: "CacheDataset"
      op: "CacheDataset"
      input: "ZipDataset:handle:0"
      input: "CacheDataset/filename:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 784
              }
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "RepeatDataset/count"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: -1
          }
        }
      }
    }
    node_def {
      name: "RepeatDataset"
      op: "RepeatDataset"
      input: "CacheDataset:handle:0"
      input: "RepeatDataset/count:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 784
              }
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/buffer_size_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 50000
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/seed"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset/seed2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: 0
          }
        }
      }
    }
    node_def {
      name: "ShuffleDataset"
      op: "ShuffleDataset"
      input: "RepeatDataset:handle:0"
      input: "ShuffleDataset/buffer_size_2:output:0"
      input: "ShuffleDataset/seed:output:0"
      input: "ShuffleDataset/seed2:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: 784
              }
            }
            shape {
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
      attr {
        key: "reshuffle_each_iteration"
        value {
          b: true
        }
      }
    }
    node_def {
      name: "BatchDataset/batch_size"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: -123
          }
        }
      }
    }
    node_def {
      name: "BatchDataset"
      op: "BatchDataset"
      input: "ShuffleDataset:handle:0"
      input: "BatchDataset/batch_size:output:0"
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: -1
              }
              dim {
                size: 784
              }
            }
            shape {
              dim {
                size: -1
              }
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
    }
    node_def {
      name: "FilterDataset/batch_size_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT64
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT64
            tensor_shape {
            }
            int64_val: -123
          }
        }
      }
    }
    node_def {
      name: "FilterDataset"
      op: "FilterDataset"
      input: "BatchDataset:handle:0"
      input: "FilterDataset/batch_size_1:output:0"
      attr {
        key: "Targuments"
        value {
          list {
            type: DT_INT64
          }
        }
      }
      attr {
        key: "output_shapes"
        value {
          list {
            shape {
              dim {
                size: -1
              }
              dim {
                size: 784
              }
            }
            shape {
              dim {
                size: -1
              }
            }
          }
        }
      }
      attr {
        key: "output_types"
        value {
          list {
            type: DT_FLOAT
            type: DT_INT32
          }
        }
      }
      attr {
        key: "predicate"
        value {
          func {
            name: "tf_predicate_7089b845"
          }
        }
      }
    }
    ret {
      key: "FilterDataset"
      value: "FilterDataset:handle:0"
    }
  }
}
)PREFIX";

  *dataset_name = "_make_dataset_2451e43a";
  std::function<void(FunctionDef*)> mutate_proto_func =
      [dataset_name, file_path, batch_size](FunctionDef* fdef) {
        VLOG(1) << "Processsing function " << fdef->DebugString();
        if (std::string(fdef->signature().name()) != *dataset_name) return;
        // Change the input file pattern to `file_path`.
        bool found_file_path = false, found_batch_size = false;
        // `node_def` may be mutated.
        for (auto& node_def : *fdef->mutable_node_def()) {
          if (node_def.name() == "FixedLengthRecordDataset/filenames" ||
              node_def.name() == "FixedLengthRecordDataset_1/filenames_1") {
            DCHECK_EQ(node_def.op(), "Const");
            DCHECK_GT(node_def.attr().count("value"), 0);
            found_file_path = true;
            // Replace $(DATA_DIR)/foo with <file_path>/foo
            // TODO(hongm): Use StringPiece manipulation for better efficiency.
            const std::string cur_value =
                node_def.attr().at("value").tensor().string_val(0);
            const std::string pattern = "$(DATA_DIR)";
            DCHECK_EQ(cur_value.compare(0, pattern.length(), pattern), 0);
            const std::string new_value =
                file_path + cur_value.substr(pattern.length());
            VLOG(1) << "Setting the value of node_def " << node_def.name()
                    << " to " << new_value;
            auto* tensor = (*node_def.mutable_attr())["value"].mutable_tensor();
            tensor->clear_string_val();
            tensor->add_string_val(new_value);
          } else if (node_def.name() == "BatchDataset/batch_size" ||
                     node_def.name() == "FilterDataset/batch_size_1") {
            DCHECK_EQ(node_def.op(), "Const");
            DCHECK_GT(node_def.attr().count("value"), 0);
            found_batch_size = true;
            // Replace $(BATCH_SIZE) with `batch_size`
            DCHECK_EQ(node_def.attr().at("value").tensor().int64_val(0), -123);
            VLOG(1) << "Setting the batch size attr value of node_def "
                    << node_def.name() << " to " << batch_size;
            auto* tensor = (*node_def.mutable_attr())["value"].mutable_tensor();
            tensor->clear_int64_val();
            tensor->add_int64_val(batch_size);
          }
        }
        VLOG(1) << "Rewrote function to " << fdef->DebugString();
        DCHECK(found_file_path);
        DCHECK(found_batch_size);
      };
  return CreateFunctionsFromTextProto(func_def, &mutate_proto_func, status);
#endif
}
#endif

// Adds the input functions to `graph`.  On success, returns the created
// IteratorGetNext node.
static TF_Operation* AddDatasetFunctionAndIteratorNodesToGraph(
    const std::vector<UniqueFuncPtr>& funcs, const std::string& dataset_name,
    const std::vector<tensorflow::DataType>& output_types,
    const std::vector<tensorflow::TensorShapeProto>& output_shapes,
    TF_Graph* graph, TF_Status* status) {
  DCHECK(!dataset_name.empty());
  for (auto& func : funcs) {
    TF_GraphCopyFunction(graph, func.get(), /*gradient*/ nullptr, status);
    if (!status->status.ok()) {
      return nullptr;
    }
  }

  tensorflow::mutex_lock c(graph->mu);

  tensorflow::NameAttrList func;
  func.set_name(dataset_name);
  // Run the iterator node on CPU.
  Node* oneshot_iterator_node;
  tensorflow::Status s = NodeBuilder("OneShotIterator", "OneShotIterator")
                             .Device("/device:CPU:0")
                             .Attr("container", "")
                             .Attr("dataset_factory", func)
                             .Attr("output_types", output_types)
                             .Attr("output_shapes", output_shapes)
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
          .Attr("output_types", output_types)
          .Attr("output_shapes", output_shapes)
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
  return ToTF_Operation(getnext_node);
}

TF_Operation* TF_MakeFakeIteratorGetNextWithDatasets(TF_Graph* graph,
                                                     TF_Status* status) {
  tensorflow::Status s;

  std::string dataset_name;
  UniqueFuncPtr result_func = CreateFakeDatasetFunction(&dataset_name, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  std::vector<UniqueFuncPtr> funcs;
  funcs.push_back(std::move(result_func));
  std::vector<tensorflow::TensorShapeProto> output_shape_list;
  output_shape_list.push_back(tensorflow::TensorShapeProto());
  auto* getnext_node = AddDatasetFunctionAndIteratorNodesToGraph(
      funcs, dataset_name, {tensorflow::DT_FLOAT}, output_shape_list, graph,
      status);
  if (!status->status.ok()) {
    return nullptr;
  }

  return getnext_node;
}

TF_Operation* TF_MakeFileBasedIteratorGetNextWithDatasets(
    TF_Graph* graph, const char* file_path, int batch_size,
    unsigned char is_mnist, TF_Status* status) {
#if defined(PLATFORM_WINDOWS)
  // TODO(ashankar): get these functions working on Windows.
  status->status = tensorflow::errors::Unimplemented(
      "TF_MakeFileBasedIteratorGetNextWithDatasets in the experimental C API "
      "is not implemented for Windows");
  return nullptr;
#else
  tensorflow::Status s;

  std::string dataset_name;
  const auto& funcs =
      is_mnist
          ? CreateMNISTDatasetFunctions(file_path, batch_size, &dataset_name,
                                        status)
          : CreateImagenetDatasetFunctions(file_path, &dataset_name, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  std::vector<tensorflow::TensorShapeProto> output_shape_list;
  // batch_size X 224 X 224 X 3
  auto image_shape = tensorflow::TensorShapeProto();
  image_shape.add_dim()->set_size(batch_size);
  if (is_mnist) {
    image_shape.add_dim()->set_size(784);
  } else {
    image_shape.add_dim()->set_size(224);
    image_shape.add_dim()->set_size(224);
    image_shape.add_dim()->set_size(3);
  }
  output_shape_list.push_back(image_shape);

  // batch_size
  auto label_shape = tensorflow::TensorShapeProto();
  label_shape.add_dim()->set_size(batch_size);
  output_shape_list.push_back(label_shape);
  auto* getnext_node = AddDatasetFunctionAndIteratorNodesToGraph(
      funcs, dataset_name, {tensorflow::DT_FLOAT, tensorflow::DT_INT32},
      output_shape_list, graph, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  tensorflow::mutex_lock c(graph->mu);
  VLOG(1) << "The extended graph: "
          << graph->graph.ToGraphDefDebug().DebugString();

  return getnext_node;
#endif
}

TF_Tensor* TF_DequeueNamedTensor(TF_Session* session, int tensor_id,
                                 TF_Status* status) {
  assert(session);
  {
    tensorflow::mutex_lock c(session->graph->mu);
    VLOG(1) << "Dequeuing named tensor with id " << tensor_id
            << ", with input graph: "
            << session->graph->graph.ToGraphDefDebug().DebugString();
  }

  TF_Operation* dequeue_op = TF_GraphOperationByName(
      session->graph,
      tensorflow::strings::StrCat("fifo_queue_dequeue_", tensor_id).c_str());
  if (dequeue_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find the dequeue node in the TF graph.");
    return nullptr;
  }

  VLOG(1) << "Running the dequeue op";
  TF_Output output{dequeue_op, 0};
  TF_Tensor* ret;
  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ &output, /*output_values*/ &ret,
                /*noutputs*/ 1,
                /*targets*/ nullptr, /*ntargets*/ 0,
                /*run_metadata*/ nullptr, status);
  if (VLOG_IS_ON(1) && status->status.ok()) {
    tensorflow::Tensor tensor;
    if (tensorflow::TF_TensorToTensor(ret, &tensor).ok()) {
      VLOG(1) << "Dequeued tensor content: " << tensor.DebugString();
    }
  }
  return ret;
}

void TF_EnqueueNamedTensor(TF_Session* session, int tensor_id,
                           TF_Tensor* tensor, TF_Status* status) {
  assert(session);
  {
    tensorflow::mutex_lock c(session->graph->mu);
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Enqueuing named tensor with id " << tensor_id
              << ", with input graph: "
              << session->graph->graph.ToGraphDefDebug().DebugString();
      tensorflow::Tensor internal_tensor;
      if (tensorflow::TF_TensorToTensor(tensor, &internal_tensor).ok()) {
        VLOG(1) << "Enqueu'ing tensor content: "
                << internal_tensor.DebugString();
      }
    }
  }

  TF_Operation* enqueue_op = TF_GraphOperationByName(
      session->graph,
      tensorflow::strings::StrCat("fifo_queue_enqueue_", tensor_id).c_str());
  if (enqueue_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find the enqueue node in the TF graph.");
    return;
  }

  TF_Operation* placeholder_op = TF_GraphOperationByName(
      session->graph,
      tensorflow::strings::StrCat("arg_tensor_enqueue_", tensor_id).c_str());
  if (placeholder_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find the placeholder node as input to enqueue in the TF "
        "graph.");
    return;
  }

  VLOG(1) << "Running the enqueue op";
  TF_Output input{placeholder_op, 0};
  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ &input, /*input_values*/ &tensor, /*ninputs*/ 1,
                // output related parameters
                /*outputs*/ nullptr, /*output_values*/ nullptr, /*noutputs*/ 0,
                /*targets*/ &enqueue_op, /*ntargets*/ 1,
                /*run_metadata*/ nullptr, status);
  VLOG(1) << "Enqueuing is done.";
}

TF_Buffer* TFE_GetServerDef(const char* text_proto, TF_Status* status) {
  tensorflow::ServerDef server_def;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(text_proto,
                                                         &server_def)) {
    status->status = tensorflow::errors::Internal(
        "Invalid text proto for ServerDef: ", text_proto);
    return nullptr;
  }
  status->status = tensorflow::Status();
  TF_Buffer* ret = TF_NewBuffer();
  TF_CHECK_OK(MessageToBuffer(server_def, ret));
  return ret;
}

TFE_Context* TFE_CreateContextFromSession(TF_Session* session,
                                          TF_Status* status) {
  auto* opts = TFE_NewContextOptions();

  // Reduce GPU memory allocation, and set appropriate config options for TFE
  // context.
  auto* config = TF_CreateConfig(
      /*xla*/ false, /* gpu_memory_allow_growth */ true, /* num_cpu_devices */
      10);
  TFE_ContextOptionsSetConfig(opts, config->data, config->length, status);
  if (!status->status.ok()) {
    CHECK(!config);
    TFE_DeleteContextOptions(opts);
    return nullptr;
  }

  auto* ctx = TFE_NewContextFromSession(opts, session, status);
  TF_DeleteBuffer(config);
  TFE_DeleteContextOptions(opts);
  return ctx;
}

// TODO: retrieve the device string via TFE_ContextListDevices()
static const char DEFAULT_CPU_DEVICE[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

static TFE_TensorHandle* createTFEQueue(TFE_Context* ctx, TF_DataType inputType,
                                        int tensor_id, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> queueOp(
      TFE_NewOp(ctx, "FIFOQueueV2", status), TFE_DeleteOp);
  TFE_OpSetDevice(queueOp.get(), DEFAULT_CPU_DEVICE, status);
  if (!status->status.ok()) return nullptr;
  // TODO: use NAMED_TENSOR_QUEUE_CAPACITY in S4TF compiler.
  TFE_OpSetAttrInt(queueOp.get(), "capacity", 1);
  TFE_OpSetAttrTypeList(queueOp.get(), "component_types", &inputType, 1);
  auto shared_name = tensorflow::strings::StrCat("fifo_queue_", tensor_id);
  TFE_OpSetAttrString(queueOp.get(), "shared_name", shared_name.data(),
                      shared_name.size());
  TFE_OpSetAttrString(queueOp.get(), "container", "", 0);

  // TODO: consider making this an unknown shape.
  const int64_t* dims_ptr = nullptr;
  int num_dims = 0;
  TFE_OpSetAttrShapeList(queueOp.get(), "shapes", &dims_ptr, &num_dims,
                         /*num_values*/ 0, status);
  if (!status->status.ok()) return nullptr;

  int num_retvals = 1;
  TFE_TensorHandle* queue = nullptr;
  TFE_Execute(queueOp.get(), &queue, &num_retvals, status);
  if (!status->status.ok()) return nullptr;
  CHECK_EQ(num_retvals, 1);

  return queue;
}

static void createTFEEnqueue(TFE_Context* ctx, TF_DataType inputType,
                             TFE_TensorHandle* queue, TFE_TensorHandle* tensor,
                             TF_Status* status) {
  TFE_Op* op = TFE_NewOp(ctx, "QueueEnqueueV2", status);
  if (!status->status.ok()) return;
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op_deleter(op, TFE_DeleteOp);
  TFE_OpSetDevice(op, DEFAULT_CPU_DEVICE, status);
  if (!status->status.ok()) return;
  TFE_OpAddInput(op, queue, status);
  if (!status->status.ok()) return;
  TFE_OpAddInput(op, tensor, status);
  if (!status->status.ok()) return;
  TFE_OpSetAttrTypeList(op, "Tcomponents", &inputType, 1);
  TFE_OpSetAttrInt(op, "timeout_ms", -1);

  int num_retvals = 0;
  TFE_Execute(op, nullptr /*retvals*/, &num_retvals, status);
  if (!status->status.ok()) return;
  CHECK_EQ(num_retvals, 0);
}

static TFE_TensorHandle* createTFEDequeue(TFE_Context* ctx,
                                          TF_DataType inputType,
                                          TFE_TensorHandle* queue,
                                          TF_Status* status) {
  TFE_Op* op = TFE_NewOp(ctx, "QueueDequeueV2", status);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op_deleter(op, TFE_DeleteOp);
  TFE_OpSetDevice(op, DEFAULT_CPU_DEVICE, status);
  if (!status->status.ok()) return nullptr;

  TFE_OpAddInput(op, queue, status);
  if (!status->status.ok()) return nullptr;
  TFE_OpSetAttrTypeList(op, "component_types", &inputType, 1);
  TFE_OpSetAttrInt(op, "timeout_ms", -1);
  TFE_TensorHandle* ret;
  int num_retvals = 1;
  TFE_Execute(op, &ret, &num_retvals, status);
  if (!status->status.ok()) return nullptr;
  CHECK_EQ(num_retvals, 1);
  return ret;
}

TFE_TensorHandle* TFE_DequeueNamedTensor(TF_Session* session, int tensor_id,
                                         TF_DataType inputType,
                                         TF_Status* status) {
  assert(session);
  VLOG(1) << "Dequeuing data tensor with id " << tensor_id;

  auto ctx = TFE_CreateContextFromSession(session, status);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> ctx_deleter(
      ctx, TFE_DeleteContext);

  TFE_TensorHandle* queue = createTFEQueue(ctx, inputType, tensor_id, status);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      queue_deleter(queue, TFE_DeleteTensorHandle);

  auto* ret = createTFEDequeue(ctx, inputType, queue, status);
  return ret;
}

TFE_TensorHandle* TFE_DequeueNamedTensorFromCtx(TFE_Context* ctx, int tensor_id,
                                                TF_DataType inputType,
                                                TF_Status* status) {
  TFE_TensorHandle* queue = createTFEQueue(ctx, inputType, tensor_id, status);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      queue_deleter(queue, TFE_DeleteTensorHandle);

  auto* ret = createTFEDequeue(ctx, inputType, queue, status);

  return ret;
}

void TFE_EnqueueNamedTensor(TF_Session* session, int tensor_id,
                            TFE_TensorHandle* tensor, TF_Status* status) {
  assert(session);
  VLOG(1) << "Enqueuing data tensor with id " << tensor_id;

  auto ctx = TFE_CreateContextFromSession(session, status);
  if (!status->status.ok()) return;
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> ctx_deleter(
      ctx, TFE_DeleteContext);

  TF_DataType inputType = TFE_TensorHandleDataType(tensor);
  TFE_TensorHandle* queue = createTFEQueue(ctx, inputType, tensor_id, status);
  if (!status->status.ok()) return;
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      queue_deleter(queue, TFE_DeleteTensorHandle);

  createTFEEnqueue(ctx, inputType, queue, tensor, status);
}

void TFE_EnqueueNamedTensorFromCtx(TFE_Context* ctx, int tensor_id,
                                   TFE_TensorHandle* tensor,
                                   TF_Status* status) {
  VLOG(1) << "Enqueuing data tensor with id " << tensor_id;

  TF_DataType inputType = TFE_TensorHandleDataType(tensor);
  TFE_TensorHandle* queue = createTFEQueue(ctx, inputType, tensor_id, status);
  if (!status->status.ok()) return;
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      queue_deleter(queue, TFE_DeleteTensorHandle);

  createTFEEnqueue(ctx, inputType, queue, tensor, status);
}

void TFE_EnqueueVariantTensor(TF_Session* session, int tensor_id,
                              TFE_TensorHandle* tensor, TF_Status* status) {
  VLOG(1) << "Enqueuing variant tensor with id " << tensor_id;

  auto ctx = TFE_CreateContextFromSession(session, status);
  if (!status->status.ok()) return;
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> ctx_deleter(
      ctx, TFE_DeleteContext);

  TFE_TensorHandle* queue = createTFEQueue(ctx, TF_VARIANT, tensor_id, status);
  if (!status->status.ok()) return;
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      queue_deleter(queue, TFE_DeleteTensorHandle);

  createTFEEnqueue(ctx, TF_VARIANT, queue, tensor, status);
}

TFE_TensorHandle* TFE_DequeueVariantTensor(TF_Session* session, int tensor_id,
                                           TF_Status* status) {
  VLOG(1) << "Dequeuing variant tensor with id " << tensor_id;

  auto ctx = TFE_CreateContextFromSession(session, status);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> ctx_deleter(
      ctx, TFE_DeleteContext);

  TFE_TensorHandle* queue = createTFEQueue(ctx, TF_VARIANT, tensor_id, status);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      queue_deleter(queue, TFE_DeleteTensorHandle);

  return createTFEDequeue(ctx, TF_VARIANT, queue, status);
}

static void CheckOk(TF_Status* status) {
  CHECK_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
}

void TFE_TensorHandlePrintDebugString(TFE_TensorHandle* handle) {
  auto* status = TF_NewStatus();
  TF_Tensor* t = TFE_TensorHandleResolve(handle, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  tensorflow::Tensor dst;
  TF_CHECK_OK(TF_TensorToTensor(t, &dst));
  LOG(INFO) << dst.DebugString();

  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
}

void TFE_OpPrintDebugString(TFE_Op* op) {
  VLOG(1) << "TFE_OpPrintDebugString() over " << op;
  LOG(INFO) << op->operation.DebugString();
}

struct TFE_ExecuteOpNotification {
  TFE_ExecuteOpNotification() : status(TF_NewStatus(), TF_DeleteStatus) {}
  tensorflow::Notification n;
  std::unique_ptr<tensorflow::Thread> thread;
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status;
};

TFE_ExecuteOpNotification* TFE_ExecuteOpInNewThread(TFE_Op* op,
                                                    TFE_TensorHandle** retvals,
                                                    int* num_retvals,
                                                    TF_Status* status) {
  TFE_ExecuteOpNotification* n = new TFE_ExecuteOpNotification;

  n->thread.reset(op->operation.EagerContext()->TFEnv()->StartThread(
      tensorflow::ThreadOptions(), "ExecuteOpThread",
      [op, retvals, num_retvals, n]() {
        TFE_Execute(op, retvals, num_retvals, n->status.get());
        n->n.Notify();
      }));

  return n;
}

void TFE_ExecuteOpNotificationWaitAndDelete(
    TFE_ExecuteOpNotification* notification, TF_Status* status) {
  if (notification == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "Passed in notification is a nullptr.");

    return;
  }
  if (notification->thread == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "Passed in notification didn't start a thread correctly. Cleaning up "
        "this notification. Please re-execute the operation to get a new "
        "notification.");

    delete notification;
    return;
  }

  notification->n.WaitForNotification();

  status->status = notification->status->status;

  delete notification;
}

void TF_MakeInternalErrorStatus(TF_Status* status, const char* errMsg) {
  status->status = tensorflow::errors::Internal(errMsg);
}

// This builder is used in the eager API to build a NodeDef.
struct TF_AttrBuilder : public tensorflow::AttrBuilder {
  using tensorflow::AttrBuilder::AttrBuilder;
  // The string buffers to make sure that any `attr_name` we pass into
  // `builder->Set()` will outlive the subsequent
  // `TF_AttrBuilderCheckCanRunOnDevice()` call(s) on the same `builder`.
  std::set<std::string> attr_names;
};

TF_AttrBuilder* TF_NewAttrBuilder(const char* op_name) {
  return new TF_AttrBuilder(op_name);
}

void TF_DeleteAttrBuilder(TF_AttrBuilder* builder) { delete builder; }

void TF_AttrBuilderSetType(TF_AttrBuilder* builder, const char* attr_name,
                           TF_DataType value) {
  auto iter = builder->attr_names.insert(attr_name).first;
  builder->Set((*iter).c_str(), static_cast<tensorflow::DataType>(value));
}

void TF_AttrBuilderSetTypeList(TF_AttrBuilder* builder, const char* attr_name,
                               const TF_DataType* values, int num_values) {
  auto iter = builder->attr_names.insert(attr_name).first;
  builder->Set(
      (*iter).c_str(),
      tensorflow::gtl::ArraySlice<const tensorflow::DataType>(
          reinterpret_cast<const tensorflow::DataType*>(values), num_values));
}

void TF_AttrBuilderCheckCanRunOnDevice(TF_AttrBuilder* builder,
                                       const char* device_type,
                                       TF_Status* status) {
  status->status = tensorflow::FindKernelDef(
      tensorflow::DeviceType(device_type), builder->BuildNodeDef(),
      /* def = */ nullptr, /* kernel_class_name = */ nullptr);
}

const char* TF_GetNumberAttrForOpListInput(const char* op_name, int input_index,
                                           TF_Status* status) {
  const tensorflow::OpDef* op_def = nullptr;
  status->status =
      tensorflow::OpRegistry::Global()->LookUpOpDef(op_name, &op_def);
  if (!status->status.ok()) return nullptr;

  if (input_index >= op_def->input_arg_size() || input_index < 0) {
    status->status = tensorflow::errors::InvalidArgument(
        input_index, " out of range for ", op_name);
    return nullptr;
  }

  const tensorflow::OpDef_ArgDef& input_arg = op_def->input_arg()[input_index];

  if (input_arg.number_attr().empty()) {
    status->status = tensorflow::errors::NotFound(
        op_name, " does not have number_attr() defined.");
    return nullptr;
  }

  // The returned string is owned by OpRegistry, so liveness is not a concern.
  return input_arg.number_attr().c_str();
}

int TF_OpIsStateful(const char* op_type, TF_Status* status) {
  const tensorflow::OpRegistrationData* op_reg_data;
  status->status =
      tensorflow::OpRegistry::Global()->LookUp(op_type, &op_reg_data);
  if (!status->status.ok()) {
    return 0;
  }
  return op_reg_data->op_def.is_stateful();
}

void TF_InitMain(const char* usage, int* argc, char*** argv) {
  tensorflow::port::InitMain(usage, argc, argv);
}

int TF_PickUnusedPortOrDie() {
  return tensorflow::internal::PickUnusedPortOrDie();
}

TFE_TensorHandle* TFE_NewTensorHandleFromScalar(TF_DataType dtype_arg,
                                                void* data, size_t len) {
  auto dtype = static_cast<tensorflow::DataType>(dtype_arg);
  DCHECK(tensorflow::DataTypeCanUseMemcpy(dtype));

  tensorflow::Tensor tensor(dtype, tensorflow::TensorShape({}));
  std::memcpy(tensorflow::TensorCApi::Buffer(tensor)->data(), data, len);
  return new TFE_TensorHandle(tensor, nullptr, nullptr);
}

namespace {
tensorflow::Status EnableCollectiveOps(const tensorflow::ServerDef& server_def,
                                       TFE_Context* ctx) {
  // We don't use the TF_RETURN_IF_ERROR macro directly since that destroys the
  // server object (which currently CHECK-fails) and we miss the error, instead,
  // we log the error, and then return to allow the user to see the error
  // message.
#define LOG_AND_RETURN_IF_ERROR(...)                    \
  do {                                                  \
    const ::tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {              \
      LOG(ERROR) << _status.error_message();            \
      return _status;                                   \
    }                                                   \
  } while (0);

  std::unique_ptr<tensorflow::ServerInterface> server;
  LOG_AND_RETURN_IF_ERROR(tensorflow::NewServer(server_def, &server));

  tensorflow::GrpcServer* grpc_server =
      dynamic_cast<tensorflow::GrpcServer*>(server.get());
  if (grpc_server == nullptr) {
    LOG_AND_RETURN_IF_ERROR(tensorflow::errors::Internal(
        "Currently, TFE_NewContext only supports tensorflow::GrpcServer."));
  }

  LOG_AND_RETURN_IF_ERROR(grpc_server->Start());

  LOG_AND_RETURN_IF_ERROR(ctx->context.StoreCollectiveOpsServer(
      std::move(server), grpc_server->worker_env()->device_mgr,
      grpc_server->worker_env()->collective_executor_mgr));

  return tensorflow::Status::OK();
#undef LOG_AND_RETURN_IF_ERROR
}
}  // namespace

// Set server_def on the context, possibly updating it.
TF_CAPI_EXPORT extern void TFE_EnableCollectiveOps(TFE_Context* ctx,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status) {
  tensorflow::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  }
  status->status = EnableCollectiveOps(server_def, ctx);
}
