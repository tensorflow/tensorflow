/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"  // NOLINT

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/c/experimental/filesystem/modular_filesystem.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_buffer_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eval_const_tensor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/cpp_shape_inference.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/release_version.h"
#include "tensorflow/core/public/session.h"

// The implementation below is at the top level instead of the
// brain namespace because we are defining 'extern "C"' functions.
using tensorflow::AttrValueMap;
using tensorflow::DataType;
using tensorflow::ExtendSessionGraphHelper;
using tensorflow::FullTypeDef;
using tensorflow::Graph;
using tensorflow::GraphDef;
using tensorflow::mutex_lock;
using tensorflow::NameRangeMap;
using tensorflow::NameRangesForNode;
using tensorflow::NewSession;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::OpDef;
using tensorflow::PartialTensorShape;
using tensorflow::RunMetadata;
using tensorflow::RunOptions;
using tensorflow::Session;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorId;
using tensorflow::TensorShapeProto;
using tensorflow::VersionDef;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::OutOfRange;
using tensorflow::gtl::ArraySlice;
using tensorflow::strings::StrCat;

extern "C" {

// --------------------------------------------------------------------------
const char* TF_Version() { return TF_VERSION_STRING; }

// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
TF_SessionOptions* TF_NewSessionOptions() {
  TF_SessionOptions* out = new TF_SessionOptions;
  // Disable optimizations for static graph to allow calls to Session::Extend.
  out->options.config.mutable_experimental()
      ->set_disable_optimize_for_static_graph(true);
  return out;
}
void TF_DeleteSessionOptions(TF_SessionOptions* opt) { delete opt; }

void TF_SetTarget(TF_SessionOptions* options, const char* target) {
  options->options.target = target;
}

void TF_SetConfig(TF_SessionOptions* options, const void* proto,
                  size_t proto_len, TF_Status* status) {
  if (!options->options.config.ParseFromArray(proto, proto_len)) {
    status->status = InvalidArgument("Unparseable ConfigProto");
  }
  // Disable optimizations for static graph to allow calls to Session::Extend.
  options->options.config.mutable_experimental()
      ->set_disable_optimize_for_static_graph(true);
}

void TF_TensorFromProto(const TF_Buffer* from, TF_Tensor* to,
                        TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  tensorflow::TensorProto from_tensor_proto;
  status->status = BufferToMessage(from, &from_tensor_proto);
  if (!status->status.ok()) {
    return;
  }
  status->status =
      tensorflow::down_cast<tensorflow::TensorInterface*>(to->tensor)
          ->FromProto(from_tensor_proto);
}
// --------------------------------------------------------------------------

TF_DeprecatedSession* TF_NewDeprecatedSession(const TF_SessionOptions* opt,
                                              TF_Status* status) {
  Session* session;
  status->status = NewSession(opt->options, &session);
  if (status->status.ok()) {
    return new TF_DeprecatedSession({session});
  } else {
    DCHECK_EQ(nullptr, session);
    return nullptr;
  }
}

void TF_CloseDeprecatedSession(TF_DeprecatedSession* s, TF_Status* status) {
  status->status = s->session->Close();
}

void TF_DeleteDeprecatedSession(TF_DeprecatedSession* s, TF_Status* status) {
  status->status = absl::OkStatus();
  if (s == nullptr) return;
  delete s->session;
  delete s;
}

void TF_ExtendGraph(TF_DeprecatedSession* s, const void* proto,
                    size_t proto_len, TF_Status* status) {
  GraphDef g;
  if (!tensorflow::ParseProtoUnlimited(&g, proto, proto_len)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  status->status = s->session->Extend(g);
}

}  // end extern "C"

// Reset helper for converting character arrays to string vectors.
static void TF_Reset_Helper(const TF_SessionOptions* opt,
                            const char** containers, int ncontainers,
                            TF_Status* status) {
  std::vector<string> container_names(ncontainers);
  for (int i = 0; i < ncontainers; ++i) {
    container_names[i] = containers[i];
  }

  status->status = Reset(opt->options, container_names);
}

extern "C" {

void TF_Reset(const TF_SessionOptions* opt, const char** containers,
              int ncontainers, TF_Status* status) {
  TF_Reset_Helper(opt, containers, ncontainers, status);
}

}  // end extern "C"

namespace tensorflow {

void RecordMutation(TF_Graph* graph, const TF_Operation& op,
                    const char* mutation_type) {
  // If any session has already run this node_id, mark this session as
  // unrunnable.
  for (auto it : graph->sessions) {
    mutex_lock session_lock(it.first->mu);
    if (it.first->last_num_graph_nodes > op.node.id()) {
      it.second = strings::StrCat(
          "Operation '", op.node.DebugString(), "' was changed by ",
          mutation_type,
          " after it was run by a session. This mutation will have no effect, "
          "and will trigger an error in the future. Either don't modify "
          "nodes after running them or create a new session.");
    }
  }
}

namespace {

// Helper method that creates a shape handle for a shape described by dims.
tensorflow::shape_inference::ShapeHandle ShapeHandleFromDims(
    tensorflow::shape_inference::InferenceContext* ic, int num_dims,
    const int64_t* dims) {
  if (num_dims != -1) {
    std::vector<tensorflow::shape_inference::DimensionHandle> dim_vec;
    dim_vec.reserve(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      dim_vec.push_back(ic->MakeDim(dims[i]));
    }
    return ic->MakeShape(dim_vec);
  } else {
    return ic->UnknownShape();
  }
}

}  // namespace

void TF_GraphSetOutputHandleShapesAndTypes(TF_Graph* graph, TF_Output output,
                                           int num_shapes_and_types,
                                           const int64_t** shapes,
                                           const int* ranks,
                                           const TF_DataType* types,
                                           TF_Status* status) {
  Node* node = &output.oper->node;

  mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(node);
  if (ic == nullptr) {
    status->status =
        InvalidArgument("Node ", node->name(), " was not found in the graph");
    return;
  }

  auto shape_and_type_vec =
      std::vector<tensorflow::shape_inference::ShapeAndType>(
          num_shapes_and_types);
  for (int i = 0; i < num_shapes_and_types; ++i) {
    tensorflow::shape_inference::ShapeHandle shape_handle =
        ShapeHandleFromDims(ic, ranks[i], shapes[i]);
    shape_and_type_vec[i] = tensorflow::shape_inference::ShapeAndType(
        shape_handle, static_cast<DataType>(types[i]));
  }

  ic->set_output_handle_shapes_and_types(output.index, shape_and_type_vec);
}

// Helpers for loading a TensorFlow plugin (a .so file).
absl::Status LoadDynamicLibrary(const char* library_filename, void** result,
                                const void** buf, size_t* len);

// TODO(josh11b,mrry): Change Session to be able to use a Graph*
// directly, instead of requiring us to serialize to a GraphDef and
// call Session::Extend().
bool ExtendSessionGraphHelper(TF_Session* session, TF_Status* status) {
  if (session->graph != nullptr) {
    // Take the graph lock before the session lock to avoid deadlock. This is
    // safe since session->graph does not change.
    session->graph->mu.lock();
    mutex_lock session_lock(session->mu);
    const Graph& graph = session->graph->graph;

    const string& mutation_warning = session->graph->sessions[session];
    if (!mutation_warning.empty()) {
      // TODO(b/74949947): turn this back into an error status
      LOG(WARNING) << mutation_warning;
      session->graph->sessions[session].clear();
    }

    const auto num_nodes = graph.num_node_ids();
    if (session->last_num_graph_nodes < num_nodes) {
      // TODO(nolivia): check this on a subset of the graph instead of all of
      // it.
      status->status = graph::ValidateGraphHasNoCycle(session->graph->graph);
      if (!status->status.ok()) {
        session->graph->mu.unlock();
        return false;
      }

      GraphDef graph_def;
      *graph_def.mutable_versions() = graph.versions();
      // Fill graph_def with nodes with ids in the range
      // [session->last_num_graph_nodes, num_nodes), that is the nodes
      // added since the last TF_SessionRun() call.
      for (auto id = session->last_num_graph_nodes; id < num_nodes; ++id) {
        Node* const node = graph.FindNodeId(id);
        if (node != nullptr && node->IsOp()) {
          NodeDef* const node_def = graph_def.add_node();
          *node_def = node->def();
        }
      }
      *graph_def.mutable_library() = graph.flib_def().ToProto();
      if (flags::Global().more_stack_traces.value()) {
        *graph_def.mutable_debug_info() = graph.BuildDebugInfo();
      }
      session->graph->mu.unlock();
      status->status = session->session->Extend(std::move(graph_def));
      if (!status->status.ok()) {
        // Contract is we always delete input_values[i].
        return false;
      }
      // Note: session->session is not modified if Extend() fails, so
      // we only set last_num_graph_nodes if it succeeds.
      session->last_num_graph_nodes = num_nodes;
    } else {
      session->graph->mu.unlock();
    }
  }
  return true;
}

}  // namespace tensorflow

static void TF_Run_Setup(int noutputs, TF_Tensor** c_outputs,
                         TF_Status* status) {
  status->status = absl::OkStatus();
  for (int i = 0; i < noutputs; ++i) {
    c_outputs[i] = nullptr;
  }
}

// TF_TensorToTensorV1 decodes a string serialization to DT_RESOURCE.
// In the TFv1 convention, TF_Tensor can hold a string serialization of
// DT_RESOURCE. The string serialization is converted back to a
// ResourceHandle during Session run where the TF_Tensor is converted to a
// Tensor.
// TFv2 does not depend on this conversion. There is no matching
// TF_TensorFromTensorV1 because the conversion to string is performed by the
// python side of Session.
static Status TF_TensorToTensorV1(const TF_Tensor* src, Tensor* dst) {
  Status status = TF_TensorToTensor(src, dst);
  if (!status.ok()) {
    return status;
  }
  if (dst->dtype() == tensorflow::DT_RESOURCE) {
    const auto tensor_interface =
        tensorflow::down_cast<const tensorflow::TensorInterface*>(src->tensor);

    if (dst->dims() != 0) {
      return InvalidArgument(
          "Malformed TF_RESOURCE tensor: expected a scalar, got a tensor with "
          "shape ",
          dst->shape().DebugString());
    }
    *dst = tensorflow::Tensor(tensorflow::DT_RESOURCE, dst->shape());
    if (!dst->scalar<tensorflow::ResourceHandle>()().ParseFromString(
            string(static_cast<const char*>(tensor_interface->Data()),
                   tensor_interface->ByteSize()))) {
      return InvalidArgument(
          "Malformed TF_RESOURCE tensor: unable to parse resource handle");
    }
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

static bool TF_Run_Inputs(TF_Tensor* const* c_inputs,
                          std::vector<std::pair<string, Tensor>>* input_pairs,
                          TF_Status* status) {
  const int ninputs = input_pairs->size();
  for (int i = 0; i < ninputs; ++i) {
    status->status =
        TF_TensorToTensorV1(c_inputs[i], &(*input_pairs)[i].second);
    if (!status->status.ok()) return false;
  }
  return true;
}

// Create an empty tensor of type 'dtype'. 'shape' can be arbitrary, but has to
// result in a zero-sized tensor.
static TF_Tensor* EmptyTensor(TF_DataType dtype,
                              const tensorflow::TensorShape& shape) {
  static char empty;
  int64_t nelems = 1;
  std::vector<int64_t> dims;
  dims.reserve(shape.dims());
  for (int i = 0; i < shape.dims(); ++i) {
    dims.push_back(shape.dim_size(i));
    nelems *= shape.dim_size(i);
  }
  CHECK_EQ(nelems, 0);
  return TF_NewTensor(
      dtype, reinterpret_cast<const int64_t*>(dims.data()), shape.dims(),
      reinterpret_cast<void*>(&empty), 0, [](void*, size_t, void*) {}, nullptr);
}

static void TF_Run_Helper(
    Session* session, const char* handle, const TF_Buffer* run_options,
    // Input tensors
    const std::vector<std::pair<string, Tensor>>& input_pairs,
    // Output tensors
    const std::vector<string>& output_tensor_names, TF_Tensor** c_outputs,
    // Target nodes
    const std::vector<string>& target_oper_names, TF_Buffer* run_metadata,
    TF_Status* status) {
  const int noutputs = output_tensor_names.size();
  std::vector<Tensor> outputs(noutputs);
  Status result;

  if (handle == nullptr) {
    RunOptions run_options_proto;
    if (run_options != nullptr && !run_options_proto.ParseFromArray(
                                      run_options->data, run_options->length)) {
      status->status = InvalidArgument("Unparseable RunOptions proto");
      return;
    }
    if (run_metadata != nullptr && run_metadata->data != nullptr) {
      status->status =
          InvalidArgument("Passing non-empty run_metadata is invalid.");
      return;
    }

    RunMetadata run_metadata_proto;
    result = session->Run(run_options_proto, input_pairs, output_tensor_names,
                          target_oper_names, &outputs, &run_metadata_proto);

    // Serialize back to upstream client, who now owns the new buffer
    if (run_metadata != nullptr) {
      status->status = MessageToBuffer(run_metadata_proto, run_metadata);
      if (!status->status.ok()) return;
    }
  } else {
    // NOTE(zongheng): PRun does not support RunOptions yet.
    result = session->PRun(handle, input_pairs, output_tensor_names, &outputs);
  }
  if (!result.ok()) {
    status->status = result;
    return;
  }

  // Store results in c_outputs[]
  for (int i = 0; i < noutputs; ++i) {
    const Tensor& src = outputs[i];
    if (!src.IsInitialized() || src.NumElements() == 0) {
      c_outputs[i] =
          EmptyTensor(static_cast<TF_DataType>(src.dtype()), src.shape());
      continue;
    }
    c_outputs[i] = TF_TensorFromTensor(src, &status->status);
    if (!status->status.ok()) return;
  }
}

extern "C" {

void TF_Run(TF_DeprecatedSession* s, const TF_Buffer* run_options,
            // Input tensors
            const char** c_input_names, TF_Tensor** c_inputs, int ninputs,
            // Output tensors
            const char** c_output_names, TF_Tensor** c_outputs, int noutputs,
            // Target nodes
            const char** c_target_oper_names, int ntargets,
            TF_Buffer* run_metadata, TF_Status* status) {
  TF_Run_Setup(noutputs, c_outputs, status);
  std::vector<std::pair<string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(c_inputs, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = c_input_names[i];
  }
  std::vector<string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = c_output_names[i];
  }
  std::vector<string> target_oper_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_oper_names[i] = c_target_oper_names[i];
  }
  TF_Run_Helper(s->session, nullptr, run_options, input_pairs, output_names,
                c_outputs, target_oper_names, run_metadata, status);
}

void TF_PRunSetup(TF_DeprecatedSession* s,
                  // Input names
                  const char** c_input_names, int ninputs,
                  // Output names
                  const char** c_output_names, int noutputs,
                  // Target nodes
                  const char** c_target_oper_names, int ntargets,
                  const char** handle, TF_Status* status) {
  *handle = nullptr;

  std::vector<string> input_names(ninputs);
  std::vector<string> output_names(noutputs);
  std::vector<string> target_oper_names(ntargets);
  for (int i = 0; i < ninputs; ++i) {
    input_names[i] = c_input_names[i];
  }
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = c_output_names[i];
  }
  for (int i = 0; i < ntargets; ++i) {
    target_oper_names[i] = c_target_oper_names[i];
  }
  string new_handle;
  status->status = s->session->PRunSetup(input_names, output_names,
                                         target_oper_names, &new_handle);
  if (status->status.ok()) {
    char* buf = new char[new_handle.size() + 1];
    memcpy(buf, new_handle.c_str(), new_handle.size() + 1);
    *handle = buf;
  }
}

void TF_PRun(TF_DeprecatedSession* s, const char* handle,
             // Input tensors
             const char** c_input_names, TF_Tensor** c_inputs, int ninputs,
             // Output tensors
             const char** c_output_names, TF_Tensor** c_outputs, int noutputs,
             // Target nodes
             const char** c_target_oper_names, int ntargets,
             TF_Status* status) {
  TF_Run_Setup(noutputs, c_outputs, status);
  std::vector<std::pair<string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(c_inputs, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = c_input_names[i];
  }

  std::vector<string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = c_output_names[i];
  }
  std::vector<string> target_oper_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_oper_names[i] = c_target_oper_names[i];
  }
  TF_Run_Helper(s->session, handle, nullptr, input_pairs, output_names,
                c_outputs, target_oper_names, nullptr, status);
}

TF_Library* TF_LoadLibrary(const char* library_filename, TF_Status* status) {
  TF_Library* lib_handle = new TF_Library;
  status->status = tensorflow::LoadDynamicLibrary(
      library_filename, &lib_handle->lib_handle, &lib_handle->op_list.data,
      &lib_handle->op_list.length);
  if (!status->status.ok()) {
    delete lib_handle;
    return nullptr;
  }
  return lib_handle;
}

TF_Buffer TF_GetOpList(TF_Library* lib_handle) { return lib_handle->op_list; }

void TF_DeleteLibraryHandle(TF_Library* lib_handle) {
  if (lib_handle == nullptr) return;
  tensorflow::port::Free(const_cast<void*>(lib_handle->op_list.data));
  delete lib_handle;
}

TF_Buffer* TF_GetAllOpList() {
  std::vector<tensorflow::OpDef> op_defs;
  tensorflow::OpRegistry::Global()->GetRegisteredOps(&op_defs);
  tensorflow::OpList op_list;
  for (const auto& op : op_defs) {
    *(op_list.add_op()) = op;
  }
  TF_Buffer* ret = TF_NewBuffer();
  TF_CHECK_OK(MessageToBuffer(op_list, ret));
  return ret;
}

// --------------------------------------------------------------------------
// ListDevices & SessionListDevices API

void TF_DeleteDeviceList(TF_DeviceList* list) { delete list; }

TF_DeviceList* TF_SessionListDevices(TF_Session* session, TF_Status* status) {
  TF_DeviceList* response = new TF_DeviceList;
  if (session && session->session)
    status->status = session->session->ListDevices(&response->response);
  return response;
}

TF_DeviceList* TF_DeprecatedSessionListDevices(TF_DeprecatedSession* session,
                                               TF_Status* status) {
  TF_DeviceList* response = new TF_DeviceList;
  if (session && session->session)
    status->status = session->session->ListDevices(&response->response);
  return response;
}

int TF_DeviceListCount(const TF_DeviceList* list) {
  return list->response.size();
}

#define TF_DEVICELIST_METHOD(return_type, method_name, accessor, err_val) \
  return_type method_name(const TF_DeviceList* list, const int index,     \
                          TF_Status* status) {                            \
    if (list == nullptr) {                                                \
      status->status = InvalidArgument("list is null!");                  \
      return err_val;                                                     \
    }                                                                     \
    if (index < 0 || index >= list->response.size()) {                    \
      status->status = InvalidArgument("index out of bounds");            \
      return err_val;                                                     \
    }                                                                     \
    status->status = ::tensorflow::OkStatus();                            \
    return list->response[index].accessor;                                \
  }

TF_DEVICELIST_METHOD(const char*, TF_DeviceListName, name().c_str(), nullptr);
TF_DEVICELIST_METHOD(const char*, TF_DeviceListType, device_type().c_str(),
                     nullptr);
TF_DEVICELIST_METHOD(int64_t, TF_DeviceListMemoryBytes, memory_limit(), -1);
TF_DEVICELIST_METHOD(uint64_t, TF_DeviceListIncarnation, incarnation(), 0);

#undef TF_DEVICELIST_METHOD

}  // end extern "C"

// --------------------------------------------------------------------------
// New Graph and Session API

// Helper functions -----------------------------------------------------------

namespace {

TF_Operation* ToOperation(Node* node) {
  return static_cast<TF_Operation*>(static_cast<void*>(node));
}

string OutputName(const TF_Output& output) {
  return StrCat(output.oper->node.name(), ":", output.index);
}

const tensorflow::AttrValue* GetAttrValue(TF_Operation* oper,
                                          const char* attr_name,
                                          TF_Status* status) {
  const tensorflow::AttrValue* attr = oper->node.attrs().Find(attr_name);
  if (attr == nullptr) {
    status->status = InvalidArgument("Operation '", oper->node.name(),
                                     "' has no attr named '", attr_name, "'.");
  }
  return attr;
}

TensorId ToTensorId(const TF_Output& output) {
  return TensorId(output.oper->node.name(), output.index);
}

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
std::vector<tensorflow::Output> OutputsFromTFOutputs(TF_Output* tf_outputs,
                                                     int n) {
  std::vector<tensorflow::Output> outputs(n);
  for (int i = 0; i < n; ++i) {
    outputs[i] =
        tensorflow::Output(&tf_outputs[i].oper->node, tf_outputs[i].index);
  }
  return outputs;
}

void TFOutputsFromOutputs(const std::vector<tensorflow::Output>& outputs,
                          TF_Output* tf_outputs) {
  for (int i = 0; i < outputs.size(); i++) {
    tf_outputs[i].oper = ToOperation(outputs[i].node());
    tf_outputs[i].index = outputs[i].index();
  }
}
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

}  // namespace

// Shape functions -----------------------------------------------------------

void TF_GraphSetTensorShape(TF_Graph* graph, TF_Output output,
                            const int64_t* dims, const int num_dims,
                            TF_Status* status) {
  Node* node = &output.oper->node;

  mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(node);
  if (ic == nullptr) {
    status->status =
        InvalidArgument("Node ", node->name(), " was not found in the graph");
    return;
  }
  tensorflow::shape_inference::ShapeHandle new_shape =
      tensorflow::ShapeHandleFromDims(ic, num_dims, dims);
  status->status = graph->refiner.SetShape(node, output.index, new_shape);
}

int TF_GraphGetTensorNumDims(TF_Graph* graph, TF_Output output,
                             TF_Status* status) {
  Node* node = &output.oper->node;

  mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(node);
  if (ic == nullptr) {
    status->status =
        InvalidArgument("Node ", node->name(), " was not found in the graph");
    return -1;
  }

  tensorflow::shape_inference::ShapeHandle shape = ic->output(output.index);

  // Unknown rank means the number of dimensions is -1.
  if (!ic->RankKnown(shape)) {
    return -1;
  }

  return ic->Rank(shape);
}

void TF_GraphGetTensorShape(TF_Graph* graph, TF_Output output, int64_t* dims,
                            int num_dims, TF_Status* status) {
  Node* node = &output.oper->node;

  mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(node);
  if (ic == nullptr) {
    status->status =
        InvalidArgument("Node ", node->name(), " was not found in the graph");
    return;
  }

  tensorflow::shape_inference::ShapeHandle shape = ic->output(output.index);

  int rank = -1;
  if (ic->RankKnown(shape)) {
    rank = ic->Rank(shape);
  }

  if (num_dims != rank) {
    status->status = InvalidArgument("Expected rank is ", num_dims,
                                     " but actual rank is ", rank);
    return;
  }

  if (num_dims == 0) {
    // Output shape is a scalar.
    return;
  }

  // Rank is greater than 0, so fill in the values, if known, and
  // -1 for unknown values.
  for (int i = 0; i < num_dims; ++i) {
    auto dim = ic->Dim(shape, i);
    int64_t value = -1;
    if (ic->ValueKnown(dim)) {
      value = ic->Value(dim);
    }
    dims[i] = value;
  }
}

// TF_OperationDescription functions ------------------------------------------

extern "C" {

TF_OperationDescription* TF_NewOperationLocked(TF_Graph* graph,
                                               const char* op_type,
                                               const char* oper_name)
    TF_EXCLUSIVE_LOCKS_REQUIRED(graph->mu) {
  return new TF_OperationDescription(graph, op_type, oper_name);
}

TF_OperationDescription* TF_NewOperation(TF_Graph* graph, const char* op_type,
                                         const char* oper_name) {
  mutex_lock l(graph->mu);
  return TF_NewOperationLocked(graph, op_type, oper_name);
}

void TF_SetDevice(TF_OperationDescription* desc, const char* device) {
  desc->node_builder.Device(device);
}

void TF_AddInput(TF_OperationDescription* desc, TF_Output input) {
  desc->node_builder.Input(&input.oper->node, input.index);
}

void TF_AddInputList(TF_OperationDescription* desc, const TF_Output* inputs,
                     int num_inputs) {
  std::vector<NodeBuilder::NodeOut> input_list;
  input_list.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_list.emplace_back(&inputs[i].oper->node, inputs[i].index);
  }
  desc->node_builder.Input(input_list);
}

void TF_AddControlInput(TF_OperationDescription* desc, TF_Operation* input) {
  desc->node_builder.ControlInput(&input->node);
}

void TF_ColocateWith(TF_OperationDescription* desc, TF_Operation* op) {
  desc->colocation_constraints.emplace(
      StrCat(tensorflow::kColocationGroupPrefix, op->node.name()));
}

void TF_SetAttrString(TF_OperationDescription* desc, const char* attr_name,
                      const void* value, size_t length) {
  absl::string_view s(static_cast<const char*>(value), length);
  desc->node_builder.Attr(attr_name, s);
}

void TF_SetAttrStringList(TF_OperationDescription* desc, const char* attr_name,
                          const void* const* values, const size_t* lengths,
                          int num_values) {
  if (strcmp(attr_name, tensorflow::kColocationAttrName) == 0) {
    desc->colocation_constraints.clear();
    for (int i = 0; i < num_values; ++i) {
      desc->colocation_constraints.emplace(static_cast<const char*>(values[i]),
                                           lengths[i]);
    }
  } else {
    std::vector<absl::string_view> v;
    v.reserve(num_values);
    for (int i = 0; i < num_values; ++i) {
      v.emplace_back(static_cast<const char*>(values[i]), lengths[i]);
    }
    desc->node_builder.Attr(attr_name, v);
  }
}

void TF_SetAttrInt(TF_OperationDescription* desc, const char* attr_name,
                   int64_t value) {
  desc->node_builder.Attr(attr_name, static_cast<int64_t>(value));
}

void TF_SetAttrIntList(TF_OperationDescription* desc, const char* attr_name,
                       const int64_t* values, int num_values) {
  desc->node_builder.Attr(
      attr_name, ArraySlice<const int64_t>(
                     reinterpret_cast<const int64_t*>(values), num_values));
}

void TF_SetAttrFloat(TF_OperationDescription* desc, const char* attr_name,
                     float value) {
  desc->node_builder.Attr(attr_name, value);
}

void TF_SetAttrFloatList(TF_OperationDescription* desc, const char* attr_name,
                         const float* values, int num_values) {
  desc->node_builder.Attr(attr_name,
                          ArraySlice<const float>(values, num_values));
}

void TF_SetAttrBool(TF_OperationDescription* desc, const char* attr_name,
                    unsigned char value) {
  desc->node_builder.Attr(attr_name, static_cast<bool>(value));
}

void TF_SetAttrBoolList(TF_OperationDescription* desc, const char* attr_name,
                        const unsigned char* values, int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  desc->node_builder.Attr(attr_name,
                          ArraySlice<const bool>(b.get(), num_values));
}

void TF_SetAttrType(TF_OperationDescription* desc, const char* attr_name,
                    TF_DataType value) {
  desc->node_builder.Attr(attr_name, static_cast<DataType>(value));
}

void TF_SetAttrTypeList(TF_OperationDescription* desc, const char* attr_name,
                        const TF_DataType* values, int num_values) {
  desc->node_builder.Attr(
      attr_name, ArraySlice<const DataType>(
                     reinterpret_cast<const DataType*>(values), num_values));
}

void TF_SetAttrPlaceholder(TF_OperationDescription* desc, const char* attr_name,
                           const char* placeholder) {
  tensorflow::AttrValue attr_value;
  attr_value.set_placeholder(placeholder);
  desc->node_builder.Attr(attr_name, attr_value);
}

void TF_SetAttrFuncName(TF_OperationDescription* desc, const char* attr_name,
                        const char* value, size_t length) {
  tensorflow::NameAttrList func_name;
  func_name.set_name(string(value, value + length));
  desc->node_builder.Attr(attr_name, func_name);
}

void TF_SetAttrShape(TF_OperationDescription* desc, const char* attr_name,
                     const int64_t* dims, int num_dims) {
  PartialTensorShape shape;
  if (num_dims >= 0) {
    shape = PartialTensorShape(absl::Span<const int64_t>(
        reinterpret_cast<const int64_t*>(dims), num_dims));
  }
  desc->node_builder.Attr(attr_name, shape);
}

void TF_SetAttrShapeList(TF_OperationDescription* desc, const char* attr_name,
                         const int64_t* const* dims, const int* num_dims,
                         int num_shapes) {
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(num_shapes);
  for (int i = 0; i < num_shapes; ++i) {
    if (num_dims[i] < 0) {
      shapes.emplace_back();
    } else {
      shapes.emplace_back(absl::Span<const int64_t>(
          reinterpret_cast<const int64_t*>(dims[i]), num_dims[i]));
    }
  }
  desc->node_builder.Attr(attr_name, shapes);
}

void TF_SetAttrTensorShapeProto(TF_OperationDescription* desc,
                                const char* attr_name, const void* proto,
                                size_t proto_len, TF_Status* status) {
  // shape.ParseFromArray takes an int as length, this function takes size_t,
  // make sure there is no information loss.
  if (proto_len > std::numeric_limits<int>::max()) {
    status->status = InvalidArgument(
        "proto_len (", proto_len,
        " bytes) is too large to be parsed by the protocol buffer library");
    return;
  }
  TensorShapeProto shape;
  if (shape.ParseFromArray(proto, static_cast<int>(proto_len))) {
    desc->node_builder.Attr(attr_name, shape);
    status->status = absl::OkStatus();
  } else {
    status->status = InvalidArgument("Unparseable TensorShapeProto");
  }
}

void TF_SetAttrTensorShapeProtoList(TF_OperationDescription* desc,
                                    const char* attr_name,
                                    const void* const* protos,
                                    const size_t* proto_lens, int num_shapes,
                                    TF_Status* status) {
  std::vector<TensorShapeProto> shapes;
  shapes.resize(num_shapes);
  for (int i = 0; i < num_shapes; ++i) {
    if (proto_lens[i] > std::numeric_limits<int>::max()) {
      status->status = InvalidArgument(
          "length of element ", i, " in the list (", proto_lens[i],
          " bytes) is too large to be parsed by the protocol buffer library");
      return;
    }
    if (!shapes[i].ParseFromArray(protos[i], static_cast<int>(proto_lens[i]))) {
      status->status =
          InvalidArgument("Unparseable TensorShapeProto at index ", i);
      return;
    }
  }
  desc->node_builder.Attr(attr_name, shapes);
  status->status = absl::OkStatus();
}

void TF_SetAttrTensor(TF_OperationDescription* desc, const char* attr_name,
                      TF_Tensor* value, TF_Status* status) {
  Tensor t;
  status->status = TF_TensorToTensor(value, &t);
  if (status->status.ok()) desc->node_builder.Attr(attr_name, t);
}

void TF_SetAttrTensorList(TF_OperationDescription* desc, const char* attr_name,
                          TF_Tensor* const* values, int num_values,
                          TF_Status* status) {
  status->status = absl::OkStatus();
  std::vector<Tensor> t;
  t.reserve(num_values);

  for (int i = 0; i < num_values && status->status.ok(); ++i) {
    Tensor v;
    status->status = TF_TensorToTensor(values[i], &v);
    t.emplace_back(v);
  }

  if (status->status.ok()) desc->node_builder.Attr(attr_name, t);
}

void TF_SetAttrValueProto(TF_OperationDescription* desc, const char* attr_name,
                          const void* proto, size_t proto_len,
                          TF_Status* status) {
  tensorflow::AttrValue attr_value;
  if (!attr_value.ParseFromArray(proto, proto_len)) {
    status->status = InvalidArgument("Unparseable AttrValue proto");
    return;
  }

  if (strcmp(attr_name, tensorflow::kColocationAttrName) == 0) {
    if (attr_value.value_case() != tensorflow::AttrValue::kList &&
        attr_value.value_case() != tensorflow::AttrValue::VALUE_NOT_SET) {
      status->status =
          InvalidArgument("Expected \"list\" field for \"",
                          tensorflow::kColocationAttrName, "\" attribute");
      return;
    }
    desc->colocation_constraints.clear();
    for (const string& location : attr_value.list().s()) {
      desc->colocation_constraints.insert(location);
    }
  } else {
    desc->node_builder.Attr(attr_name, std::move(attr_value));
  }

  status->status = absl::OkStatus();
}

TF_Operation* TF_FinishOperationLocked(TF_OperationDescription* desc,
                                       TF_Status* status)
    TF_EXCLUSIVE_LOCKS_REQUIRED(desc->graph->mu) {
  Node* ret = nullptr;

  if (desc->graph->name_map.count(desc->node_builder.node_name())) {
    status->status = InvalidArgument("Duplicate node name in graph: '",
                                     desc->node_builder.node_name(), "'");
  } else {
    if (!desc->colocation_constraints.empty()) {
      desc->node_builder.Attr(
          tensorflow::kColocationAttrName,
          std::vector<string>(desc->colocation_constraints.begin(),
                              desc->colocation_constraints.end()));
    }
    status->status = desc->node_builder.Finalize(&desc->graph->graph, &ret,
                                                 /*consume=*/true);

    if (status->status.ok()) {
      // Run shape inference function for newly added node.
      status->status = desc->graph->refiner.AddNode(ret);
    }
    if (status->status.ok()) {
      // Add the node to the name-to-node mapping.
      desc->graph->name_map[ret->name()] = ret;
    } else if (ret != nullptr) {
      desc->graph->graph.RemoveNode(ret);
      ret = nullptr;
    }
  }

  delete desc;

  return ToOperation(ret);
}

TF_Operation* TF_FinishOperation(TF_OperationDescription* desc,
                                 TF_Status* status) {
  mutex_lock l(desc->graph->mu);
  return TF_FinishOperationLocked(desc, status);
}

// TF_Operation functions
// ----------------------------------------------------------

const char* TF_OperationName(TF_Operation* oper) {
  return oper->node.name().c_str();
}

const char* TF_OperationOpType(TF_Operation* oper) {
  return oper->node.type_string().c_str();
}

const char* TF_OperationDevice(TF_Operation* oper) {
  return oper->node.requested_device().c_str();
}

int TF_OperationNumOutputs(TF_Operation* oper) {
  return oper->node.num_outputs();
}

TF_DataType TF_OperationOutputType(TF_Output oper_out) {
  return static_cast<TF_DataType>(
      oper_out.oper->node.output_type(oper_out.index));
}

int TF_OperationOutputListLength(TF_Operation* oper, const char* arg_name,
                                 TF_Status* status) {
  NameRangeMap name_ranges;
  status->status =
      NameRangesForNode(oper->node, oper->node.op_def(), nullptr, &name_ranges);
  if (!status->status.ok()) return -1;
  auto iter = name_ranges.find(arg_name);
  if (iter == name_ranges.end()) {
    status->status = InvalidArgument("Output arg '", arg_name, "' not found");
    return -1;
  }
  return iter->second.second - iter->second.first;
}

int TF_OperationNumInputs(TF_Operation* oper) {
  return oper->node.num_inputs();
}

TF_DataType TF_OperationInputType(TF_Input oper_in) {
  return static_cast<TF_DataType>(oper_in.oper->node.input_type(oper_in.index));
}

int TF_OperationInputListLength(TF_Operation* oper, const char* arg_name,
                                TF_Status* status) {
  NameRangeMap name_ranges;
  status->status =
      NameRangesForNode(oper->node, oper->node.op_def(), &name_ranges, nullptr);
  if (!status->status.ok()) return -1;
  auto iter = name_ranges.find(arg_name);
  if (iter == name_ranges.end()) {
    status->status = InvalidArgument("Input arg '", arg_name, "' not found");
    return -1;
  }
  return iter->second.second - iter->second.first;
}

TF_Output TF_OperationInput(TF_Input oper_in) {
  const tensorflow::Edge* edge;
  Status s = oper_in.oper->node.input_edge(oper_in.index, &edge);
  if (!s.ok()) {
    return {nullptr, -1};
  }

  return {ToOperation(edge->src()), edge->src_output()};
}

void TF_OperationAllInputs(TF_Operation* oper, TF_Output* inputs,
                           int max_inputs) {
  for (auto* edge : oper->node.in_edges()) {
    if (edge->dst_input() >= 0 && edge->dst_input() < max_inputs) {
      inputs[edge->dst_input()] = {ToOperation(edge->src()),
                                   edge->src_output()};
    }
  }
}

int TF_OperationOutputNumConsumers(TF_Output oper_out) {
  int count = 0;
  for (const auto* edge : oper_out.oper->node.out_edges()) {
    if (edge->src_output() == oper_out.index) {
      ++count;
    }
  }
  return count;
}

int TF_OperationOutputConsumers(TF_Output oper_out, TF_Input* consumers,
                                int max_consumers) {
  int count = 0;
  for (const auto* edge : oper_out.oper->node.out_edges()) {
    if (edge->src_output() == oper_out.index) {
      if (count < max_consumers) {
        consumers[count] = {ToOperation(edge->dst()), edge->dst_input()};
      }
      ++count;
    }
  }
  return count;
}

int TF_OperationNumControlInputs(TF_Operation* oper) {
  int count = 0;
  for (const auto* edge : oper->node.in_edges()) {
    if (edge->IsControlEdge() && !edge->src()->IsSource()) {
      ++count;
    }
  }
  return count;
}

int TF_OperationGetControlInputs(TF_Operation* oper,
                                 TF_Operation** control_inputs,
                                 int max_control_inputs) {
  int count = 0;
  for (const auto* edge : oper->node.in_edges()) {
    if (edge->IsControlEdge() && !edge->src()->IsSource()) {
      if (count < max_control_inputs) {
        control_inputs[count] = ToOperation(edge->src());
      }
      ++count;
    }
  }
  return count;
}

int TF_OperationNumControlOutputs(TF_Operation* oper) {
  int count = 0;
  for (const auto* edge : oper->node.out_edges()) {
    if (edge->IsControlEdge() && !edge->dst()->IsSink()) {
      ++count;
    }
  }
  return count;
}

int TF_OperationGetControlOutputs(TF_Operation* oper,
                                  TF_Operation** control_outputs,
                                  int max_control_outputs) {
  int count = 0;
  for (const auto* edge : oper->node.out_edges()) {
    if (edge->IsControlEdge() && !edge->dst()->IsSink()) {
      if (count < max_control_outputs) {
        control_outputs[count] = ToOperation(edge->dst());
      }
      ++count;
    }
  }
  return count;
}

TF_AttrMetadata TF_OperationGetAttrMetadata(TF_Operation* oper,
                                            const char* attr_name,
                                            TF_Status* status) {
  TF_AttrMetadata metadata;
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return metadata;
  switch (attr->value_case()) {
#define SINGLE_CASE(kK, attr_type, size_expr) \
  case tensorflow::AttrValue::kK:             \
    metadata.is_list = 0;                     \
    metadata.list_size = -1;                  \
    metadata.type = attr_type;                \
    metadata.total_size = size_expr;          \
    break;

    SINGLE_CASE(kS, TF_ATTR_STRING, attr->s().length());
    SINGLE_CASE(kI, TF_ATTR_INT, -1);
    SINGLE_CASE(kF, TF_ATTR_FLOAT, -1);
    SINGLE_CASE(kB, TF_ATTR_BOOL, -1);
    SINGLE_CASE(kType, TF_ATTR_TYPE, -1);
    SINGLE_CASE(kShape, TF_ATTR_SHAPE,
                attr->shape().unknown_rank() ? -1 : attr->shape().dim_size());
    SINGLE_CASE(kTensor, TF_ATTR_TENSOR, -1);
#undef SINGLE_CASE

    case tensorflow::AttrValue::kList:
      metadata.is_list = 1;
      metadata.list_size = 0;
      metadata.total_size = -1;
#define LIST_CASE(field, attr_type, ...)              \
  if (attr->list().field##_size() > 0) {              \
    metadata.type = attr_type;                        \
    metadata.list_size = attr->list().field##_size(); \
    __VA_ARGS__;                                      \
    break;                                            \
  }

      LIST_CASE(
          s, TF_ATTR_STRING, metadata.total_size = 0;
          for (int i = 0; i < attr->list().s_size();
               ++i) { metadata.total_size += attr->list().s(i).size(); });
      LIST_CASE(i, TF_ATTR_INT);
      LIST_CASE(f, TF_ATTR_FLOAT);
      LIST_CASE(b, TF_ATTR_BOOL);
      LIST_CASE(type, TF_ATTR_TYPE);
      LIST_CASE(
          shape, TF_ATTR_SHAPE, metadata.total_size = 0;
          for (int i = 0; i < attr->list().shape_size(); ++i) {
            const auto& s = attr->list().shape(i);
            metadata.total_size += s.unknown_rank() ? 0 : s.dim_size();
          });
      LIST_CASE(tensor, TF_ATTR_TENSOR);
      LIST_CASE(tensor, TF_ATTR_FUNC);
#undef LIST_CASE
      // All lists empty, determine the type from the OpDef.
      if (metadata.list_size == 0) {
        for (int i = 0; i < oper->node.op_def().attr_size(); ++i) {
          const auto& a = oper->node.op_def().attr(i);
          if (a.name() != attr_name) continue;
          const string& typestr = a.type();
          if (typestr == "list(string)") {
            metadata.type = TF_ATTR_STRING;
          } else if (typestr == "list(int)") {
            metadata.type = TF_ATTR_INT;
          } else if (typestr == "list(float)") {
            metadata.type = TF_ATTR_FLOAT;
          } else if (typestr == "list(bool)") {
            metadata.type = TF_ATTR_BOOL;
          } else if (typestr == "list(type)") {
            metadata.type = TF_ATTR_TYPE;
          } else if (typestr == "list(shape)") {
            metadata.type = TF_ATTR_SHAPE;
          } else if (typestr == "list(tensor)") {
            metadata.type = TF_ATTR_TENSOR;
          } else if (typestr == "list(func)") {
            metadata.type = TF_ATTR_FUNC;
          } else {
            status->status = InvalidArgument(
                "Attribute '", attr_name,
                "' has an empty value of an unrecognized type '", typestr, "'");
            return metadata;
          }
        }
      }
      break;

    case tensorflow::AttrValue::kPlaceholder:
      metadata.is_list = 0;
      metadata.list_size = -1;
      metadata.type = TF_ATTR_PLACEHOLDER;
      metadata.total_size = -1;
      break;

    case tensorflow::AttrValue::kFunc:
      metadata.is_list = 0;
      metadata.list_size = -1;
      metadata.type = TF_ATTR_FUNC;
      metadata.total_size = -1;
      break;

    case tensorflow::AttrValue::VALUE_NOT_SET:
      status->status =
          InvalidArgument("Attribute '", attr_name, "' has no value set");
      break;
  }
  return metadata;
}

void TF_OperationGetAttrString(TF_Operation* oper, const char* attr_name,
                               void* value, size_t max_length,
                               TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kS) {
    status->status =
        InvalidArgument("Attribute '", attr_name, "' is not a string");
    return;
  }
  if (max_length <= 0) {
    return;
  }
  const auto& s = attr->s();
  std::memcpy(value, s.data(), std::min<size_t>(s.length(), max_length));
}

void TF_OperationGetAttrStringList(TF_Operation* oper, const char* attr_name,
                                   void** values, size_t* lengths,
                                   int max_values, void* storage,
                                   size_t storage_size, TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kList) {
    status->status =
        InvalidArgument("Value for '", attr_name, "' is not a list");
    return;
  }
  const auto len = std::min(max_values, attr->list().s_size());
  char* p = static_cast<char*>(storage);
  for (int i = 0; i < len; ++i) {
    const string& s = attr->list().s(i);
    values[i] = p;
    lengths[i] = s.size();
    if ((p + s.size()) > (static_cast<char*>(storage) + storage_size)) {
      status->status = InvalidArgument(
          "Not enough storage to hold the requested list of strings");
      return;
    }
    memcpy(values[i], s.data(), s.size());
    p += s.size();
  }
}

#define DEFINE_GETATTR(func, c_type, cpp_type, list_field)                   \
  void func(TF_Operation* oper, const char* attr_name, c_type* value,        \
            TF_Status* status) {                                             \
    cpp_type v;                                                              \
    status->status =                                                         \
        tensorflow::GetNodeAttr(oper->node.attrs(), attr_name, &v);          \
    if (!status->status.ok()) return;                                        \
    *value = static_cast<c_type>(v);                                         \
  }                                                                          \
  void func##List(TF_Operation* oper, const char* attr_name, c_type* values, \
                  int max_values, TF_Status* status) {                       \
    const auto* attr = GetAttrValue(oper, attr_name, status);                \
    if (!status->status.ok()) return;                                        \
    if (attr->value_case() != tensorflow::AttrValue::kList) {                \
      status->status =                                                       \
          InvalidArgument("Value for '", attr_name, "' is not a list.");     \
      return;                                                                \
    }                                                                        \
    const auto len = std::min(max_values, attr->list().list_field##_size()); \
    for (int i = 0; i < len; ++i) {                                          \
      values[i] = static_cast<c_type>(attr->list().list_field(i));           \
    }                                                                        \
  }
DEFINE_GETATTR(TF_OperationGetAttrInt, int64_t, int64_t, i);
DEFINE_GETATTR(TF_OperationGetAttrFloat, float, float, f);
DEFINE_GETATTR(TF_OperationGetAttrBool, unsigned char, bool, b);
DEFINE_GETATTR(TF_OperationGetAttrType, TF_DataType, DataType, type);
#undef DEFINE_GETATTR

void TF_OperationGetAttrShape(TF_Operation* oper, const char* attr_name,
                              int64_t* value, int num_dims, TF_Status* status) {
  PartialTensorShape shape;
  status->status =
      tensorflow::GetNodeAttr(oper->node.attrs(), attr_name, &shape);
  if (!status->status.ok()) return;
  auto len = std::min(shape.dims(), num_dims);
  for (int i = 0; i < len; ++i) {
    value[i] = shape.dim_size(i);
  }
}

void TF_OperationGetAttrShapeList(TF_Operation* oper, const char* attr_name,
                                  int64_t** dims, int* num_dims, int num_shapes,
                                  int64_t* storage, int storage_size,
                                  TF_Status* status) {
  std::vector<PartialTensorShape> shapes;
  status->status =
      tensorflow::GetNodeAttr(oper->node.attrs(), attr_name, &shapes);
  if (!status->status.ok()) return;
  auto len = std::min(static_cast<int>(shapes.size()), num_shapes);
  int64_t* p = storage;
  int storage_left = storage_size;
  for (int i = 0; i < len; ++i) {
    // shapes[i].dims() == -1 for shapes with an unknown rank.
    int64_t n = shapes[i].dims();
    num_dims[i] = n;
    dims[i] = p;
    if (n < 0) {
      continue;
    }
    if (storage_left < n) {
      status->status = InvalidArgument(
          "Not enough storage to hold the requested list of shapes");
      return;
    }
    storage_left -= n;
    for (int j = 0; j < n; ++j, ++p) {
      *p = shapes[i].dim_size(j);
    }
  }
}

void TF_OperationGetAttrTensorShapeProto(TF_Operation* oper,
                                         const char* attr_name,
                                         TF_Buffer* value, TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kShape) {
    status->status =
        InvalidArgument("Value for '", attr_name, "' is not a shape.");
    return;
  }
  status->status = MessageToBuffer(attr->shape(), value);
}

void TF_OperationGetAttrTensorShapeProtoList(TF_Operation* oper,
                                             const char* attr_name,
                                             TF_Buffer** values, int max_values,
                                             TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  if (attr->value_case() != tensorflow::AttrValue::kList) {
    status->status =
        InvalidArgument("Value for '", attr_name, "' is not a list");
    return;
  }
  const auto len = std::min(max_values, attr->list().shape_size());
  for (int i = 0; i < len; ++i) {
    values[i] = TF_NewBuffer();
    status->status = MessageToBuffer(attr->list().shape(i), values[i]);
    if (!status->status.ok()) {
      // Delete everything allocated to far, the operation has failed.
      for (int j = 0; j <= i; ++j) {
        TF_DeleteBuffer(values[j]);
      }
      return;
    }
  }
}

void TF_OperationGetAttrTensor(TF_Operation* oper, const char* attr_name,
                               TF_Tensor** value, TF_Status* status) {
  *value = nullptr;
  Tensor t;
  status->status = tensorflow::GetNodeAttr(oper->node.attrs(), attr_name, &t);
  if (!status->status.ok()) return;
  *value = TF_TensorFromTensor(t, &status->status);
}

void TF_OperationGetAttrTensorList(TF_Operation* oper, const char* attr_name,
                                   TF_Tensor** values, int max_values,
                                   TF_Status* status) {
  std::vector<Tensor> ts;
  status->status = tensorflow::GetNodeAttr(oper->node.attrs(), attr_name, &ts);
  if (!status->status.ok()) return;
  const auto len = std::min(max_values, static_cast<int>(ts.size()));
  for (int i = 0; i < len; ++i) {
    values[i] = TF_TensorFromTensor(ts[i], &status->status);
  }
}

void TF_OperationGetAttrValueProto(TF_Operation* oper, const char* attr_name,
                                   TF_Buffer* output_attr_value,
                                   TF_Status* status) {
  const auto* attr = GetAttrValue(oper, attr_name, status);
  if (!status->status.ok()) return;
  status->status = MessageToBuffer(*attr, output_attr_value);
}

int TF_OperationGetNumAttrs(TF_Operation* oper) {
  return oper->node.attrs().size();
}

int TF_OperationGetAttrNameLength(TF_Operation* oper, int i) {
  auto attrs = oper->node.attrs();
  int count = 0;
  AttrValueMap::const_iterator it;
  for (it = attrs.begin(); it != attrs.end(); it++) {
    if (count == i) {
      return it->first.length();
    }
    count++;
  }
  return -1;
}

void TF_OperationGetAttrName(TF_Operation* oper, int i, char* output,
                             TF_Status* status) {
  auto attrs = oper->node.attrs();
  int count = 0;
  AttrValueMap::const_iterator it;
  for (it = attrs.begin(); it != attrs.end(); it++) {
    if (count == i) {
      strncpy(output, it->first.c_str(), it->first.length());
      status->status = absl::OkStatus();
      return;
    }
    count++;
  }
  status->status = OutOfRange("Operation only has ", count,
                              " attributes, can't get the ", i, "th");
}

void TF_OperationToNodeDef(TF_Operation* oper, TF_Buffer* output_node_def,
                           TF_Status* status) {
  status->status = MessageToBuffer(oper->node.def(), output_node_def);
}

// TF_Graph functions ---------------------------------------------------------

TF_Graph::TF_Graph()
    : graph(tensorflow::OpRegistry::Global()),
      refiner(graph.versions().producer(), graph.op_registry()),
      delete_requested(false),
      parent(nullptr),
      parent_inputs(nullptr) {
  // Tell the shape refiner to also run shape inference on functions.
  refiner.set_function_library_for_shape_inference(&graph.flib_def());
}

TF_Graph* TF_NewGraph() { return new TF_Graph; }

void TF_DeleteGraph(TF_Graph* g) {
  if (g == nullptr) return;
  g->mu.lock();
  g->delete_requested = true;
  const bool del = g->sessions.empty();
  g->mu.unlock();
  if (del) delete g;
}

TF_Operation* TF_GraphOperationByName(TF_Graph* graph, const char* oper_name) {
  mutex_lock l(graph->mu);
  auto iter = graph->name_map.find(oper_name);
  if (iter == graph->name_map.end()) {
    return nullptr;
  } else {
    return ToOperation(iter->second);
  }
}

TF_Operation* TF_GraphNextOperation(TF_Graph* graph, size_t* pos) {
  if (*pos == 0) {
    // Advance past the first sentinel nodes in every graph (the source & sink).
    *pos += 2;
  } else {
    // Advance to the next node.
    *pos += 1;
  }

  mutex_lock l(graph->mu);
  while (*pos < static_cast<size_t>(graph->graph.num_node_ids())) {
    Node* node = graph->graph.FindNodeId(*pos);
    // FindNodeId() returns nullptr for nodes that have been deleted.
    // We aren't currently allowing nodes to be deleted, but it is safer
    // to still check.
    if (node != nullptr) return ToOperation(node);
    *pos += 1;
  }

  // No more nodes.
  return nullptr;
}

void TF_GraphToGraphDef(TF_Graph* graph, TF_Buffer* output_graph_def,
                        TF_Status* status) {
  GraphDef def;
  {
    mutex_lock l(graph->mu);
    graph->graph.ToGraphDef(&def);
  }
  status->status = MessageToBuffer(def, output_graph_def);
}

void TF_GraphGetOpDef(TF_Graph* graph, const char* op_name,
                      TF_Buffer* output_op_def, TF_Status* status) {
  const OpDef* op_def;
  {
    mutex_lock l(graph->mu);
    status->status = graph->graph.op_registry()->LookUpOpDef(op_name, &op_def);
    if (!status->status.ok()) return;
  }
  status->status = MessageToBuffer(*op_def, output_op_def);
}

void TF_GraphVersions(TF_Graph* graph, TF_Buffer* output_version_def,
                      TF_Status* status) {
  VersionDef versions;
  {
    mutex_lock l(graph->mu);
    versions = graph->graph.versions();
  }
  status->status = MessageToBuffer(versions, output_version_def);
}

TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions() {
  return new TF_ImportGraphDefOptions;
}
void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions* opts) {
  delete opts;
}
void TF_ImportGraphDefOptionsSetPrefix(TF_ImportGraphDefOptions* opts,
                                       const char* prefix) {
  opts->opts.prefix = prefix;
}
void TF_ImportGraphDefOptionsSetDefaultDevice(TF_ImportGraphDefOptions* opts,
                                              const char* device) {
  opts->opts.default_device = device;
}

void TF_ImportGraphDefOptionsSetUniquifyNames(TF_ImportGraphDefOptions* opts,
                                              unsigned char uniquify_names) {
  opts->opts.uniquify_names = uniquify_names;
}

void TF_ImportGraphDefOptionsSetUniquifyPrefix(TF_ImportGraphDefOptions* opts,
                                               unsigned char uniquify_prefix) {
  opts->opts.uniquify_prefix = uniquify_prefix;
}

void TF_ImportGraphDefOptionsAddInputMapping(TF_ImportGraphDefOptions* opts,
                                             const char* src_name,
                                             int src_index, TF_Output dst) {
  opts->tensor_id_data.push_back(src_name);
  const string& src_name_str = opts->tensor_id_data.back();
  // We don't need to store dst's name in tensor_id_data, since `dst` must
  // outlive the ImportGraphDef call.
  opts->opts.input_map[TensorId(src_name_str, src_index)] = ToTensorId(dst);
}

void TF_ImportGraphDefOptionsRemapControlDependency(
    TF_ImportGraphDefOptions* opts, const char* src_name, TF_Operation* dst) {
  opts->opts.input_map[TensorId(src_name, tensorflow::Graph::kControlSlot)] =
      TensorId(dst->node.name(), tensorflow::Graph::kControlSlot);
}

extern void TF_ImportGraphDefOptionsAddControlDependency(
    TF_ImportGraphDefOptions* opts, TF_Operation* oper) {
  opts->opts.control_dependencies.push_back(oper->node.name());
}

void TF_ImportGraphDefOptionsAddReturnOutput(TF_ImportGraphDefOptions* opts,
                                             const char* oper_name, int index) {
  opts->tensor_id_data.push_back(oper_name);
  const string& oper_name_str = opts->tensor_id_data.back();
  opts->opts.return_tensors.emplace_back(oper_name_str, index);
}

int TF_ImportGraphDefOptionsNumReturnOutputs(
    const TF_ImportGraphDefOptions* opts) {
  return opts->opts.return_tensors.size();
}

void TF_ImportGraphDefOptionsAddReturnOperation(TF_ImportGraphDefOptions* opts,
                                                const char* oper_name) {
  opts->opts.return_nodes.push_back(oper_name);
}

int TF_ImportGraphDefOptionsNumReturnOperations(
    const TF_ImportGraphDefOptions* opts) {
  return opts->opts.return_nodes.size();
}

void TF_ImportGraphDefResultsReturnOutputs(TF_ImportGraphDefResults* results,
                                           int* num_outputs,
                                           TF_Output** outputs) {
  *num_outputs = results->return_tensors.size();
  *outputs = results->return_tensors.data();
}

void TF_ImportGraphDefResultsReturnOperations(TF_ImportGraphDefResults* results,
                                              int* num_opers,
                                              TF_Operation*** opers) {
  *num_opers = results->return_nodes.size();
  *opers = results->return_nodes.data();
}

void TF_ImportGraphDefResultsMissingUnusedInputMappings(
    TF_ImportGraphDefResults* results, int* num_missing_unused_input_mappings,
    const char*** src_names, int** src_indexes) {
  *num_missing_unused_input_mappings = results->missing_unused_key_names.size();
  *src_names = results->missing_unused_key_names.data();
  *src_indexes = results->missing_unused_key_indexes.data();
}

void TF_DeleteImportGraphDefResults(TF_ImportGraphDefResults* results) {
  delete results;
}

static void GraphImportGraphDefLocked(TF_Graph* graph, const GraphDef& def,
                                      const TF_ImportGraphDefOptions* opts,
                                      TF_ImportGraphDefResults* tf_results,
                                      TF_Status* status)
    TF_EXCLUSIVE_LOCKS_REQUIRED(graph->mu) {
  const int last_node_id = graph->graph.num_node_ids();
  tensorflow::ImportGraphDefResults results;
  status->status = tensorflow::ImportGraphDef(opts->opts, def, &graph->graph,
                                              &graph->refiner, &results);
  if (!status->status.ok()) return;

  // Add new nodes to name_map
  for (int i = last_node_id; i < graph->graph.num_node_ids(); ++i) {
    auto* node = graph->graph.FindNodeId(i);
    if (node != nullptr) graph->name_map[node->name()] = node;
  }

  // Populate return_tensors
  DCHECK(tf_results->return_tensors.empty());
  tf_results->return_tensors.resize(results.return_tensors.size());
  for (int i = 0; i < results.return_tensors.size(); ++i) {
    tf_results->return_tensors[i].oper =
        ToOperation(results.return_tensors[i].first);
    tf_results->return_tensors[i].index = results.return_tensors[i].second;
  }

  // Populate return_nodes
  DCHECK(tf_results->return_nodes.empty());
  tf_results->return_nodes.resize(results.return_nodes.size());
  for (int i = 0; i < results.return_nodes.size(); ++i) {
    tf_results->return_nodes[i] = ToOperation(results.return_nodes[i]);
  }

  // Populate missing unused map keys
  DCHECK(tf_results->missing_unused_key_names.empty());
  DCHECK(tf_results->missing_unused_key_indexes.empty());
  DCHECK(tf_results->missing_unused_key_names_data.empty());

  size_t size = results.missing_unused_input_map_keys.size();
  tf_results->missing_unused_key_names.resize(size);
  tf_results->missing_unused_key_indexes.resize(size);

  for (int i = 0; i < size; ++i) {
    TensorId id = results.missing_unused_input_map_keys[i];
    tf_results->missing_unused_key_names_data.emplace_back(id.first);
    tf_results->missing_unused_key_names[i] =
        tf_results->missing_unused_key_names_data.back().c_str();
    tf_results->missing_unused_key_indexes[i] = id.second;
  }
}

TF_ImportGraphDefResults* TF_GraphImportGraphDefWithResults(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Status* status) {
  GraphDef def;
  if (!tensorflow::ParseProtoUnlimited(&def, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return nullptr;
  }
  auto results = new TF_ImportGraphDefResults();
  mutex_lock l(graph->mu);
  GraphImportGraphDefLocked(graph, def, options, results, status);
  if (!status->status.ok()) {
    delete results;
    return nullptr;
  }
  return results;
}

TF_ImportGraphDefResults* TF_GraphImportGraphDefWithResultsNoSerialization(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Status* status) {
  const GraphDef* graph_def_ptr =
      reinterpret_cast<const GraphDef*>(graph_def->data);
  auto results = new TF_ImportGraphDefResults();
  mutex_lock l(graph->mu);
  GraphImportGraphDefLocked(graph, *graph_def_ptr, options, results, status);
  if (!status->status.ok()) {
    delete results;
    return nullptr;
  }
  return results;
}

void TF_GraphImportGraphDefWithReturnOutputs(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Output* return_outputs,
    int num_return_outputs, TF_Status* status) {
  if (num_return_outputs != options->opts.return_tensors.size()) {
    status->status = InvalidArgument("Expected 'num_return_outputs' to be ",
                                     options->opts.return_tensors.size(),
                                     ", got ", num_return_outputs);
    return;
  }
  if (num_return_outputs > 0 && return_outputs == nullptr) {
    status->status = InvalidArgument(
        "'return_outputs' must be preallocated to length ", num_return_outputs);
    return;
  }
  GraphDef def;
  if (!tensorflow::ParseProtoUnlimited(&def, graph_def->data,
                                       graph_def->length)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }
  TF_ImportGraphDefResults results;
  mutex_lock l(graph->mu);
  GraphImportGraphDefLocked(graph, def, options, &results, status);
  DCHECK_EQ(results.return_tensors.size(), num_return_outputs);
  memcpy(return_outputs, results.return_tensors.data(),
         num_return_outputs * sizeof(TF_Output));
}

void TF_GraphImportGraphDef(TF_Graph* graph, const TF_Buffer* graph_def,
                            const TF_ImportGraphDefOptions* options,
                            TF_Status* status) {
  TF_ImportGraphDefResults* results =
      TF_GraphImportGraphDefWithResults(graph, graph_def, options, status);
  TF_DeleteImportGraphDefResults(results);
}

// While loop functions -------------------------------------------------------

namespace {

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

// Creates a placeholder representing an input to the cond or body graph.
// TODO(skyewm): remove these from final graph
bool CreateInput(const TF_Output& parent_input, TF_Graph* g, const char* name,
                 TF_Output* input, TF_Status* status) {
  TF_OperationDescription* desc = TF_NewOperation(g, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", TF_OperationOutputType(parent_input));
  // TODO(skyewm): set placeholder shape
  TF_Operation* oper = TF_FinishOperation(desc, status);
  if (!status->status.ok()) return false;
  *input = {oper, 0};
  return true;
}

// Copies `src_graph` into `dst_graph`. Any node in `src_graph` with input
// `src_inputs[i]` will have that input replaced with `dst_inputs[i]`.  `prefix`
// will be prepended to copied node names. `control_deps` are nodes in
// `dst_graph` that the copied `src_graph` nodes will have control dependencies
// on. `return_nodes` are nodes in `src_graph`, and the new corresponding nodes
// in `dst_graph` will be returned. `return_nodes` must be non-null.
Status CopyGraph(Graph* src_graph, Graph* dst_graph,
                 tensorflow::ShapeRefiner* dst_refiner,
                 const TF_Output* src_inputs,
                 const std::vector<tensorflow::Output>& dst_inputs,
                 const string& prefix,
                 const std::vector<tensorflow::Operation>& control_deps,
                 const TF_Output* nodes_to_return, int nreturn_nodes,
                 std::vector<tensorflow::Output>* return_nodes) {
  DCHECK(return_nodes != nullptr);
  GraphDef gdef;
  src_graph->ToGraphDef(&gdef);

  tensorflow::ImportGraphDefOptions opts;
  opts.prefix = prefix;

  for (int i = 0; i < dst_inputs.size(); ++i) {
    opts.input_map[ToTensorId(src_inputs[i])] =
        TensorId(dst_inputs[i].node()->name(), dst_inputs[i].index());
  }
  opts.skip_mapped_nodes = true;

  for (const tensorflow::Operation& op : control_deps) {
    opts.control_dependencies.push_back(op.node()->name());
  }

  for (int i = 0; i < nreturn_nodes; ++i) {
    opts.return_tensors.push_back(ToTensorId(nodes_to_return[i]));
  }

  // TODO(skyewm): change to OutputTensor
  tensorflow::ImportGraphDefResults results;
  TF_RETURN_IF_ERROR(
      ImportGraphDef(opts, gdef, dst_graph, dst_refiner, &results));

  for (const auto& pair : results.return_tensors) {
    return_nodes->emplace_back(pair.first, pair.second);
  }
  return absl::OkStatus();
}

bool ValidateConstWhileParams(const TF_WhileParams& params, TF_Status* s) {
  if (params.cond_graph == nullptr || params.body_graph == nullptr ||
      params.cond_graph->parent == nullptr ||
      params.cond_graph->parent != params.body_graph->parent ||
      params.cond_graph->parent_inputs != params.body_graph->parent_inputs ||
      params.ninputs <= 0 || params.cond_inputs == nullptr ||
      params.body_inputs == nullptr || params.body_outputs == nullptr) {
    s->status = InvalidArgument(
        "TF_WhileParams must be created by successful TF_NewWhile() call");
    return false;
  }
  return true;
}

bool ValidateInputWhileParams(const TF_WhileParams& params, TF_Status* s) {
  if (params.cond_output.oper == nullptr) {
    s->status = InvalidArgument("TF_WhileParams `cond_output` field isn't set");
    return false;
  }
  for (int i = 0; i < params.ninputs; ++i) {
    if (params.body_outputs[i].oper == nullptr) {
      s->status = InvalidArgument("TF_WhileParams `body_outputs[", i, "]` ",
                                  "field isn't set");
      return false;
    }
  }
  if (params.name == nullptr) {
    s->status = InvalidArgument("TF_WhileParams `name` field is null");
    return false;
  }
  return true;
}

#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

void FreeWhileResources(const TF_WhileParams* params) {
  TF_DeleteGraph(params->cond_graph);
  TF_DeleteGraph(params->body_graph);
  delete[] params->cond_inputs;
  delete[] params->body_inputs;
  delete[] params->body_outputs;
}

TF_WhileParams EmptyWhileParams() {
  return {0,       nullptr, nullptr, {nullptr, 0},
          nullptr, nullptr, nullptr, nullptr};
}

}  // namespace

TF_WhileParams TF_NewWhile(TF_Graph* g, TF_Output* inputs, int ninputs,
                           TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Creating while loops is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
  return EmptyWhileParams();
#else
  if (ninputs == 0) {
    status->status =
        InvalidArgument("TF_NewWhile() must be passed at least one input");
    return EmptyWhileParams();
  }

  TF_Graph* cond_graph = TF_NewGraph();
  TF_Graph* body_graph = TF_NewGraph();
  cond_graph->parent = g;
  cond_graph->parent_inputs = inputs;
  body_graph->parent = g;
  body_graph->parent_inputs = inputs;

  TF_Output* cond_inputs = new TF_Output[ninputs];
  TF_Output cond_output = {nullptr, -1};
  TF_Output* body_inputs = new TF_Output[ninputs];
  TF_Output* body_outputs = new TF_Output[ninputs];
  for (int i = 0; i < ninputs; ++i) body_outputs[i] = {nullptr, -1};
  const char* name = nullptr;

  for (int i = 0; i < ninputs; ++i) {
    // TODO(skyewm): prefix names with underscore (requires some plumbing)
    if (!CreateInput(inputs[i], cond_graph, StrCat("cond_input", i).c_str(),
                     &cond_inputs[i], status)) {
      break;
    }
    if (!CreateInput(inputs[i], body_graph, StrCat("body_input", i).c_str(),
                     &body_inputs[i], status)) {
      break;
    }
  }

  TF_WhileParams params = {ninputs,    cond_graph,  cond_inputs,  cond_output,
                           body_graph, body_inputs, body_outputs, name};

  if (!status->status.ok()) {
    FreeWhileResources(&params);
    return EmptyWhileParams();
  }
  return params;
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
namespace {

// TODO(skyewm): make nodes in while loop unfetchable like in Python version
void TF_FinishWhileHelper(const TF_WhileParams* params, TF_Status* status,
                          TF_Output* outputs) {
  if (!ValidateInputWhileParams(*params, status)) return;

  TF_Graph* parent = params->cond_graph->parent;
  TF_Output* parent_inputs = params->cond_graph->parent_inputs;
  int num_loop_vars = params->ninputs;

  mutex_lock l(parent->mu);

  // 'cond_fn' copies the cond graph into the parent graph.
  tensorflow::ops::CondGraphBuilderFn cond_fn =
      [params, parent](const tensorflow::Scope& scope,
                       const std::vector<tensorflow::Output>& inputs,
                       tensorflow::Output* output) {
        DCHECK_EQ(scope.graph(), &parent->graph);
        std::vector<tensorflow::Output> cond_output;
        TF_RETURN_IF_ERROR(CopyGraph(
            &params->cond_graph->graph, &parent->graph, &parent->refiner,
            params->cond_inputs, inputs, scope.impl()->name(),
            scope.impl()->control_deps(), &params->cond_output,
            /* nreturn_nodes */ 1, &cond_output));
        *output = cond_output[0];
        return absl::OkStatus();
      };

  // 'body_fn' copies the body graph into the parent graph.
  tensorflow::ops::BodyGraphBuilderFn body_fn =
      [params, parent, num_loop_vars](
          const tensorflow::Scope& scope,
          const std::vector<tensorflow::Output>& inputs,
          std::vector<tensorflow::Output>* outputs) {
        DCHECK_EQ(scope.graph(), &parent->graph);
        TF_RETURN_IF_ERROR(
            CopyGraph(&params->body_graph->graph, &parent->graph,
                      &parent->refiner, params->body_inputs, inputs,
                      scope.impl()->name(), scope.impl()->control_deps(),
                      params->body_outputs, num_loop_vars, outputs));
        return absl::OkStatus();
      };

  // Create the while loop using an internal scope.
  tensorflow::Scope scope =
      NewInternalScope(&parent->graph, &status->status, &parent->refiner)
          .NewSubScope(params->name);

  const int first_new_node_id = parent->graph.num_node_ids();

  tensorflow::OutputList loop_outputs;
  status->status = tensorflow::ops::BuildWhileLoop(
      scope, OutputsFromTFOutputs(parent_inputs, num_loop_vars), cond_fn,
      body_fn, params->name, &loop_outputs);

  // Update name_map with newly-created ops.
  // TODO(skyewm): right now BuildWhileLoop() may alter the graph if it returns
  // a bad status. Once we fix this, we may want to return early instead of
  // executing the following code.
  for (int i = first_new_node_id; i < parent->graph.num_node_ids(); ++i) {
    Node* new_node = parent->graph.FindNodeId(i);
    if (new_node == nullptr) continue;
    parent->name_map[new_node->name()] = new_node;
  }

  // Populate 'outputs'.
  DCHECK_LE(loop_outputs.size(), num_loop_vars);
  for (int i = 0; i < loop_outputs.size(); ++i) {
    outputs[i] = {ToOperation(loop_outputs[i].node()), loop_outputs[i].index()};
  }
}

}  // namespace
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

void TF_FinishWhile(const TF_WhileParams* params, TF_Status* status,
                    TF_Output* outputs) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Creating while loops is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
#else
  // If it appears the caller created or modified `params`, don't free resources
  if (!ValidateConstWhileParams(*params, status)) return;
  TF_FinishWhileHelper(params, status, outputs);
  FreeWhileResources(params);
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

void TF_AbortWhile(const TF_WhileParams* params) { FreeWhileResources(params); }

void TF_AddGradients(TF_Graph* g, TF_Output* y, int ny, TF_Output* x, int nx,
                     TF_Output* dx, TF_Status* status, TF_Output* dy) {
  TF_AddGradientsWithPrefix(g, nullptr, y, ny, x, nx, dx, status, dy);
}

void TF_AddGradientsWithPrefix(TF_Graph* g, const char* prefix, TF_Output* y,
                               int ny, TF_Output* x, int nx, TF_Output* dx,
                               TF_Status* status, TF_Output* dy) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Adding gradients is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
#else
  std::vector<tensorflow::Output> y_arg = OutputsFromTFOutputs(y, ny);
  std::vector<tensorflow::Output> x_arg = OutputsFromTFOutputs(x, nx);
  std::vector<tensorflow::Output> dy_arg;

  {
    // We need to hold on to the lock while we have a scope that uses TF_Graph.
    mutex_lock graph_lock(g->mu);

    const int first_new_node_id = g->graph.num_node_ids();

    string prefix_cmp;
    const char* child_scope_name;
    if (prefix == nullptr) {
      child_scope_name = "gradients";
    } else {
      prefix_cmp = string(prefix) + "/";
      // The operation should fail if the provided name prefix has already been
      // used in this graph
      for (const auto& pair : g->name_map) {
        const string& name = pair.first;
        if ((name == prefix) || absl::StartsWith(name, prefix_cmp)) {
          status->status = InvalidArgument(
              "prefix [", prefix,
              "] conflicts with existing node in the graph named [", name, "]");
          return;
        }
      }
      child_scope_name = prefix;
    }
    tensorflow::Scope scope =
        NewInternalScope(&g->graph, &status->status, &g->refiner)
            .NewSubScope(child_scope_name);

    if (dx != nullptr) {
      std::vector<tensorflow::Output> dx_arg = OutputsFromTFOutputs(dx, ny);
      status->status =
          AddSymbolicGradients(scope, y_arg, x_arg, dx_arg, &dy_arg);
    } else {
      status->status = AddSymbolicGradients(scope, y_arg, x_arg, &dy_arg);
    }

    // Update g->name_map with the name_map from the scope, which will contain
    // the new gradient ops.
    for (int i = first_new_node_id; i < g->graph.num_node_ids(); ++i) {
      Node* n = g->graph.FindNodeId(i);
      if (n == nullptr) continue;

      // Adding the gradients to the graph can alter the prefix to prevent
      // name collisions only if this prefix has not been provided explicitly
      // by the user. If it was provided, assert that it remained intact.
      if (prefix != nullptr && !absl::StartsWith(n->name(), prefix_cmp)) {
        status->status = tensorflow::errors::Internal(
            "BUG: The gradients prefix have been unexpectedly altered when "
            "adding the nodes to the graph. This is a bug. Please file an "
            "issue at https://github.com/tensorflow/tensorflow/issues.");
        return;
      }
      // We have a convoluted scheme here: Using the C++ graph construction API
      // to add potentially many nodes to the graph without running the checks
      // (such as uniqueness of the names of nodes) we run with other functions
      // that add a node to the graph (like TF_FinishOperation).
      if (!g->name_map.insert(std::make_pair(n->name(), n)).second) {
        status->status = tensorflow::errors::Internal(
            "BUG: The API allowed construction of a graph with duplicate node "
            "names (",
            n->name(),
            "). This is a bug. Please file an issue at "
            "https://github.com/tensorflow/tensorflow/issues.");
      }
    }
  }

  // Unpack the results from grad_outputs_arg.
  TFOutputsFromOutputs(dy_arg, dy);
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

// TF_Session functions ----------------------------------------------

TF_Session::TF_Session(tensorflow::Session* s, TF_Graph* g)
    : session(s), graph(g), last_num_graph_nodes(0), extend_before_run(true) {}

TF_Session* TF_NewSession(TF_Graph* graph, const TF_SessionOptions* opt,
                          TF_Status* status) {
  Session* session;
  status->status = NewSession(opt->options, &session);
  if (status->status.ok()) {
    TF_Session* new_session = new TF_Session(session, graph);
    if (graph != nullptr) {
      mutex_lock l(graph->mu);
      graph->sessions[new_session] = "";
    }
    return new_session;
  } else {
    LOG(ERROR) << status->status;
    DCHECK_EQ(nullptr, session);
    return nullptr;
  }
}

TF_Session* TF_LoadSessionFromSavedModel(
    const TF_SessionOptions* session_options, const TF_Buffer* run_options,
    const char* export_dir, const char* const* tags, int tags_len,
    TF_Graph* graph, TF_Buffer* meta_graph_def, TF_Status* status) {
// TODO(sjr): Remove the IS_MOBILE_PLATFORM guard. This will require ensuring
// that the tensorflow/cc/saved_model:loader build target is mobile friendly.
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Loading a SavedModel is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
  return nullptr;
#else
  mutex_lock l(graph->mu);
  if (!graph->name_map.empty()) {
    status->status = InvalidArgument("Graph is non-empty.");
    return nullptr;
  }

  RunOptions run_options_proto;
  if (run_options != nullptr && !run_options_proto.ParseFromArray(
                                    run_options->data, run_options->length)) {
    status->status = InvalidArgument("Unparseable RunOptions proto");
    return nullptr;
  }

  std::unordered_set<string> tag_set;
  for (int i = 0; i < tags_len; i++) {
    tag_set.insert(string(tags[i]));
  }

  tensorflow::SavedModelBundle bundle;
  status->status =
      tensorflow::LoadSavedModel(session_options->options, run_options_proto,
                                 export_dir, tag_set, &bundle);
  if (!status->status.ok()) return nullptr;

  // Create a TF_Graph from the MetaGraphDef. This is safe as long as Session
  // extends using GraphDefs. The Graph instance is different, but equivalent
  // to the one used to create the session.
  //
  // TODO(jhseu): When Session is modified to take Graphs instead of
  // GraphDefs, return the Graph generated in LoadSavedModel().
  TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefResults results;
  GraphImportGraphDefLocked(graph, bundle.meta_graph_def.graph_def(),
                            import_opts, &results, status);
  TF_DeleteImportGraphDefOptions(import_opts);
  if (!status->status.ok()) return nullptr;

  if (meta_graph_def != nullptr) {
    status->status = MessageToBuffer(bundle.meta_graph_def, meta_graph_def);
    if (!status->status.ok()) return nullptr;
  }

  TF_Session* session = new TF_Session(bundle.session.release(), graph);

  graph->sessions[session] = "";
  session->last_num_graph_nodes = graph->graph.num_node_ids();
  return session;
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

void TF_CloseSession(TF_Session* s, TF_Status* status) {
  status->status = s->session->Close();
}

void TF_DeleteSession(TF_Session* s, TF_Status* status) {
  status->status = absl::OkStatus();
  if (s == nullptr) return;
  TF_Graph* const graph = s->graph;
  if (graph != nullptr) {
    graph->mu.lock();
    graph->sessions.erase(s);
    const bool del = graph->delete_requested && graph->sessions.empty();
    graph->mu.unlock();
    if (del) delete graph;
  }
  delete s->session;
  delete s;
}

void TF_SessionRun(TF_Session* session, const TF_Buffer* run_options,
                   const TF_Output* inputs, TF_Tensor* const* input_values,
                   int ninputs, const TF_Output* outputs,
                   TF_Tensor** output_values, int noutputs,
                   const TF_Operation* const* target_opers, int ntargets,
                   TF_Buffer* run_metadata, TF_Status* status) {
  // TODO(josh11b,mrry): Change Session to be able to use a Graph*
  // directly, instead of requiring us to serialize to a GraphDef and
  // call Session::Extend().
  if (session->extend_before_run &&
      !ExtendSessionGraphHelper(session, status)) {
    return;
  }

  TF_Run_Setup(noutputs, output_values, status);

  // Convert from TF_Output and TF_Tensor to a string and Tensor.
  std::vector<std::pair<string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(input_values, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = OutputName(inputs[i]);
  }

  // Convert from TF_Output to string names.
  std::vector<string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = OutputName(outputs[i]);
  }

  // Convert from TF_Operation* to string names.
  std::vector<string> target_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_names[i] = target_opers[i]->node.name();
  }

  // Actually run.
  TF_Run_Helper(session->session, nullptr, run_options, input_pairs,
                output_names, output_values, target_names, run_metadata,
                status);
}

void TF_SessionPRunSetup(TF_Session* session, const TF_Output* inputs,
                         int ninputs, const TF_Output* outputs, int noutputs,
                         const TF_Operation* const* target_opers, int ntargets,
                         const char** handle, TF_Status* status) {
  *handle = nullptr;

  if (session->extend_before_run &&
      !ExtendSessionGraphHelper(session, status)) {
    return;
  }

  std::vector<string> input_names(ninputs);
  for (int i = 0; i < ninputs; ++i) {
    input_names[i] = OutputName(inputs[i]);
  }

  std::vector<string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = OutputName(outputs[i]);
  }

  std::vector<string> target_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_names[i] = target_opers[i]->node.name();
  }

  string new_handle;
  status->status = session->session->PRunSetup(input_names, output_names,
                                               target_names, &new_handle);
  if (status->status.ok()) {
    char* buf = new char[new_handle.size() + 1];
    memcpy(buf, new_handle.c_str(), new_handle.size() + 1);
    *handle = buf;
  }
}

void TF_DeletePRunHandle(const char* handle) {
  delete[] handle;
  // TODO(suharshs): Free up any resources held by the partial run state.
}

void TF_SessionPRun(TF_Session* session, const char* handle,
                    const TF_Output* inputs, TF_Tensor* const* input_values,
                    int ninputs, const TF_Output* outputs,
                    TF_Tensor** output_values, int noutputs,
                    const TF_Operation* const* target_opers, int ntargets,
                    TF_Status* status) {
  // TODO(josh11b,mrry): Change Session to be able to use a Graph*
  // directly, instead of requiring us to serialize to a GraphDef and
  // call Session::Extend().
  if (session->extend_before_run &&
      !ExtendSessionGraphHelper(session, status)) {
    return;
  }

  TF_Run_Setup(noutputs, output_values, status);

  // Convert from TF_Output and TF_Tensor to a string and Tensor.
  std::vector<std::pair<string, Tensor>> input_pairs(ninputs);
  if (!TF_Run_Inputs(input_values, &input_pairs, status)) return;
  for (int i = 0; i < ninputs; ++i) {
    input_pairs[i].first = OutputName(inputs[i]);
  }

  // Convert from TF_Output to string names.
  std::vector<string> output_names(noutputs);
  for (int i = 0; i < noutputs; ++i) {
    output_names[i] = OutputName(outputs[i]);
  }

  // Convert from TF_Operation* to string names.
  std::vector<string> target_names(ntargets);
  for (int i = 0; i < ntargets; ++i) {
    target_names[i] = target_opers[i]->node.name();
  }

  TF_Run_Helper(session->session, handle, nullptr, input_pairs, output_names,
                output_values, target_names, nullptr, status);
}

unsigned char TF_TryEvaluateConstant(TF_Graph* graph, TF_Output output,
                                     TF_Tensor** result, TF_Status* status) {
  mutex_lock l(graph->mu);
  auto status_or = EvaluateConstantTensor(
      output.oper->node, output.index, graph->refiner,
      [](const Node&, int) { return std::optional<Tensor>(); },
      tensorflow::EvaluateConstantTensorRunner{
          graph->graph.op_registry(),
          graph->graph.versions().producer(),
      });
  if (!status_or.ok() || !status_or->has_value()) {
    *result = nullptr;
    status->status = std::move(status_or).status();
    return false;
  }
  *result = TF_TensorFromTensor(**status_or, &status->status);
  return status->status.ok();
}

TF_ApiDefMap* TF_NewApiDefMap(TF_Buffer* op_list_buffer, TF_Status* status) {
  tensorflow::OpList op_list;
  if (!op_list.ParseFromArray(op_list_buffer->data, op_list_buffer->length)) {
    status->status = InvalidArgument("Unparseable OpList");
    return nullptr;
  }
  status->status = absl::OkStatus();
  return new TF_ApiDefMap(op_list);
}

void TF_DeleteApiDefMap(TF_ApiDefMap* apimap) { delete apimap; }

void TF_ApiDefMapPut(TF_ApiDefMap* api_def_map, const char* text,
                     size_t text_len, TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "ApiDefMap is not supported on mobile.");
#else
  mutex_lock l(api_def_map->lock);
  if (api_def_map->update_docs_called) {
    status->status = FailedPrecondition(
        "TF_ApiDefMapPut cannot be called after TF_ApiDefMapGet has been "
        "called.");
    return;
  }
  string api_def_text(text, text_len);
  status->status = api_def_map->api_def_map.LoadApiDef(api_def_text);
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

TF_Buffer* TF_ApiDefMapGet(TF_ApiDefMap* api_def_map, const char* name,
                           size_t name_len, TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "ApiDefMap is not supported on mobile.");
  return nullptr;
#else
  mutex_lock l(api_def_map->lock);
  if (!api_def_map->update_docs_called) {
    api_def_map->api_def_map.UpdateDocs();
    api_def_map->update_docs_called = true;
  }
  string name_str(name, name_len);
  const auto* api_def = api_def_map->api_def_map.GetApiDef(name_str);
  if (api_def == nullptr) {
    return nullptr;
  }

  TF_Buffer* ret = TF_NewBuffer();
  status->status = MessageToBuffer(*api_def, ret);
  if (!status->status.ok()) {
    TF_DeleteBuffer(ret);
    return nullptr;
  }
  return ret;
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

TF_Buffer* TF_GetAllRegisteredKernels(TF_Status* status) {
  tensorflow::KernelList kernel_list = tensorflow::GetAllRegisteredKernels();
  TF_Buffer* ret = TF_NewBuffer();
  status->status = MessageToBuffer(kernel_list, ret);
  if (!status->status.ok()) {
    TF_DeleteBuffer(ret);
    return nullptr;
  }
  return ret;
}

TF_Buffer* TF_GetRegisteredKernelsForOp(const char* name, TF_Status* status) {
  tensorflow::KernelList kernel_list =
      tensorflow::GetRegisteredKernelsForOp(name);
  TF_Buffer* ret = TF_NewBuffer();
  status->status = MessageToBuffer(kernel_list, ret);
  if (!status->status.ok()) {
    TF_DeleteBuffer(ret);
    return nullptr;
  }
  return ret;
}

void TF_UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst,
                   TF_Status* status) {
  using tensorflow::RecordMutation;
  mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(&new_src.oper->node);

  if (ic->num_outputs() <= new_src.index) {
    status->status = tensorflow::errors::OutOfRange(
        "Cannot update edge. Output index [", new_src.index,
        "] is greater than the number of total outputs [", ic->num_outputs(),
        "].");
    return;
  }
  tensorflow::shape_inference::ShapeHandle shape = ic->output(new_src.index);

  tensorflow::shape_inference::InferenceContext* ic_dst =
      graph->refiner.GetContext(&dst.oper->node);
  if (ic_dst->num_inputs() <= dst.index) {
    status->status = tensorflow::errors::OutOfRange(
        "Cannot update edge. Input index [", dst.index,
        "] is greater than the number of total inputs [", ic_dst->num_inputs(),
        "].");
    return;
  }
  if (!ic_dst->MergeInput(dst.index, shape)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Cannot update edge, incompatible shapes: ", ic_dst->DebugString(shape),
        " and ", ic_dst->DebugString(ic_dst->input(dst.index)), ".");
    return;
  }
  status->status = graph->graph.UpdateEdge(&new_src.oper->node, new_src.index,
                                           &dst.oper->node, dst.index);

  if (TF_GetCode(status) == TF_OK) {
    // This modification only updates the destination node for
    // the purposes of running this graph in a session. Thus, we don't
    // record the source node as being modified.
    RecordMutation(graph, *dst.oper, "updating input tensor");
  }
}

// Apis that are corresponding to python c api. --------------------------

void TF_AddOperationControlInput(TF_Graph* graph, TF_Operation* op,
                                 TF_Operation* input) {
  using tensorflow::RecordMutation;
  mutex_lock l(graph->mu);
  graph->graph.AddControlEdge(&input->node, &op->node);
  RecordMutation(graph, *op, "adding control input");
}

void TF_SetAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
                TF_Buffer* attr_value_proto, TF_Status* status) {
  using tensorflow::RecordMutation;
  tensorflow::AttrValue attr_val;
  if (!attr_val.ParseFromArray(attr_value_proto->data,
                               attr_value_proto->length)) {
    status->status = absl::InvalidArgumentError("Invalid AttrValue proto");
    return;
  }

  mutex_lock l(graph->mu);
  op->node.AddAttr(attr_name, attr_val);
  RecordMutation(graph, *op, "setting attribute");
}

void TF_ClearAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
                  TF_Status* status) {
  using tensorflow::RecordMutation;
  mutex_lock l(graph->mu);
  op->node.ClearAttr(attr_name);
  RecordMutation(graph, *op, "clearing attribute");
}

void TF_SetFullType(TF_Graph* graph, TF_Operation* op,
                    const TF_Buffer* full_type_proto) {
  using tensorflow::RecordMutation;
  mutex_lock l(graph->mu);
  FullTypeDef full_type;
  full_type.ParseFromArray(full_type_proto->data, full_type_proto->length);
  *op->node.mutable_def()->mutable_experimental_type() = full_type;
  RecordMutation(graph, *op, "setting fulltype");
}

void TF_SetRequestedDevice(TF_Graph* graph, TF_Operation* op,
                           const char* device) {
  using tensorflow::RecordMutation;
  mutex_lock l(graph->mu);
  op->node.set_requested_device(device);
  RecordMutation(graph, *op, "setting device");
}

void TF_RemoveAllControlInputs(TF_Graph* graph, TF_Operation* op) {
  mutex_lock l(graph->mu);
  std::vector<const tensorflow::Edge*> control_edges;
  for (const tensorflow::Edge* edge : op->node.in_edges()) {
    if (!edge->IsControlEdge()) continue;
    control_edges.push_back(edge);
  }
  for (const tensorflow::Edge* edge : control_edges) {
    graph->graph.RemoveControlEdge(edge);
  }
}

void TF_SetRequireShapeInferenceFns(TF_Graph* graph, bool require) {
  mutex_lock l(graph->mu);
  graph->refiner.set_require_shape_inference_fns(require);
}

void TF_ExtendSession(TF_Session* session, TF_Status* status) {
  ExtendSessionGraphHelper(session, status);
  session->extend_before_run = false;
}

TF_Buffer* TF_GetHandleShapeAndType(TF_Graph* graph, TF_Output output) {
  Node* node = &output.oper->node;
  tensorflow::core::CppShapeInferenceResult::HandleData handle_data;
  handle_data.set_is_set(true);
  {
    mutex_lock l(graph->mu);
    tensorflow::shape_inference::InferenceContext* ic =
        graph->refiner.GetContext(node);
    CHECK(ic != nullptr);                       // Crash OK
    CHECK_LT(output.index, ic->num_outputs());  // Crash OK
    const auto* shapes_and_types =
        ic->output_handle_shapes_and_types(output.index);
    if (shapes_and_types == nullptr) return nullptr;

    for (const auto& p : *shapes_and_types) {
      auto* out_shape_and_type = handle_data.add_shape_and_type();
      ic->ShapeHandleToProto(p.shape, out_shape_and_type->mutable_shape());
      out_shape_and_type->set_dtype(p.dtype);
      *out_shape_and_type->mutable_type() = p.type;
    }
  }
  string str_data;
  handle_data.SerializeToString(&str_data);

  TF_Buffer* result = TF_NewBufferFromString(str_data.c_str(), str_data.size());
  return result;
}

void TF_SetHandleShapeAndType(TF_Graph* graph, TF_Output output,
                              const void* proto, size_t proto_len,
                              TF_Status* status) {
  tensorflow::core::CppShapeInferenceResult::HandleData handle_data;
  if (!handle_data.ParseFromArray(proto, proto_len)) {
    status->status =
        absl::InvalidArgumentError("Couldn't deserialize HandleData proto");
    return;
  }
  DCHECK(handle_data.is_set());

  tensorflow::mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(&output.oper->node);

  std::vector<tensorflow::shape_inference::ShapeAndType> shapes_and_types;
  for (const auto& shape_and_type_proto : handle_data.shape_and_type()) {
    tensorflow::shape_inference::ShapeHandle shape;
    status->status =
        ic->MakeShapeFromShapeProto(shape_and_type_proto.shape(), &shape);
    if (TF_GetCode(status) != TF_OK) return;
    shapes_and_types.emplace_back(shape, shape_and_type_proto.dtype(),
                                  shape_and_type_proto.type());
  }
  ic->set_output_handle_shapes_and_types(output.index, shapes_and_types);
}

void TF_AddWhileInputHack(TF_Graph* graph, TF_Output new_src, TF_Operation* dst,
                          TF_Status* status) {
  using tensorflow::RecordMutation;
  mutex_lock l(graph->mu);
  status->status = graph->graph.AddWhileInputHack(&new_src.oper->node,
                                                  new_src.index, &dst->node);
  if (TF_GetCode(status) == TF_OK) {
    // This modification only updates the destination node for
    // the purposes of running this graph in a session. Thus, we don't
    // record the source node as being modified.
    RecordMutation(graph, *dst, "adding input tensor");
  }
}

// -------------------------------------------------------------------

// TF_Server functions ----------------------------------------------

#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
TF_Server::TF_Server(std::unique_ptr<tensorflow::ServerInterface> server)
    : target(server->target()), server(std::move(server)) {}
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

TF_Server* TF_NewServer(const void* proto, size_t proto_len,
                        TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Server functionality is not supported on mobile");
  return nullptr;
#else
  tensorflow::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, static_cast<int>(proto_len))) {
    status->status = InvalidArgument(
        "Could not parse provided bytes into a ServerDef protocol buffer");
    return nullptr;
  }

  std::unique_ptr<tensorflow::ServerInterface> out_server;
  status->status = tensorflow::NewServer(server_def, &out_server);
  if (!status->status.ok()) return nullptr;

  return new TF_Server(std::move(out_server));
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

void TF_ServerStart(TF_Server* server, TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Server functionality is not supported on mobile");
#else
  status->status = server->server->Start();
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

void TF_ServerStop(TF_Server* server, TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Server functionality is not supported on mobile");
#else
  status->status = server->server->Stop();
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

void TF_ServerJoin(TF_Server* server, TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Server functionality is not supported on mobile");
#else
  status->status = server->server->Join();
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

const char* TF_ServerTarget(TF_Server* server) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  return nullptr;
#else
  return server->target.c_str();
#endif
}

void TF_DeleteServer(TF_Server* server) {
#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
  delete server;
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
}

void TF_RegisterLogListener(void (*listener)(const char*)) {
#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
  tensorflow::logging::RegisterListener(listener);
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
}

void TF_RegisterFilesystemPlugin(const char* plugin_filename,
                                 TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "FileSystem plugin functionality is not supported on mobile");
#else
  status->status = tensorflow::RegisterFilesystemPlugin(plugin_filename);
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

}  // end extern "C"
