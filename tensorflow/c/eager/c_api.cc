/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/runtime.h"
#ifdef TENSORFLOW_EAGER_USE_XLA
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#endif  // TENSORFLOW_EAGER_USE_XLA
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/python/framework/cpp_shape_inference.pb.h"

using tensorflow::int64;
using tensorflow::string;

namespace {
bool IsCPU(const tensorflow::Device* d) {
  return d == nullptr || d->tensorflow_gpu_device_info() == nullptr;
}

bool IsXLA(const tensorflow::Device* d) {
  if (d == nullptr) return false;
  const auto& device_type = d->attributes().device_type();
  return device_type.find("XLA") != std::string::npos;
}

string DeviceName(const tensorflow::Device* d) {
  return (d == nullptr) ? "cpu:0" : d->name();
}

#ifdef TENSORFLOW_EAGER_USE_XLA
std::atomic_int_fast64_t func_id_generator(0);
#endif  // TENSORFLOW_EAGER_USE_XLA

}  // namespace

extern "C" {

TFE_ContextOptions* TFE_NewContextOptions() { return new TFE_ContextOptions; }

void TFE_ContextOptionsSetConfig(TFE_ContextOptions* options, const void* proto,
                                 size_t proto_len, TF_Status* status) {
  TF_SetConfig(&options->session_options, proto, proto_len, status);
}

void TFE_ContextOptionsSetAsync(TFE_ContextOptions* options,
                                unsigned char async) {
  options->async = async;
}
void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions* options, TFE_ContextDevicePlacementPolicy policy) {
  options->policy = policy;
}

TF_CAPI_EXPORT extern void TFE_ContextSetAsyncForThread(TFE_Context* ctx,
                                                        unsigned char async,
                                                        TF_Status* status) {
  status->status = ctx->context.SetAsyncForThread(async);
}

void TFE_DeleteContextOptions(TFE_ContextOptions* options) { delete options; }

TFE_Context* TFE_NewContext(const TFE_ContextOptions* opts, TF_Status* status) {
  std::vector<tensorflow::Device*> devices;
  status->status = tensorflow::DeviceFactory::AddDevices(
      opts->session_options.options, "/job:localhost/replica:0/task:0",
      &devices);
  if (!status->status.ok()) {
    return nullptr;
  }
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr(
      new tensorflow::DeviceMgr(devices));
  tensorflow::Rendezvous* r =
      new tensorflow::IntraProcessRendezvous(device_mgr.get());
  return new TFE_Context(opts->session_options.options, opts->policy,
                         opts->async, std::move(device_mgr), r);
}

void TFE_DeleteContext(TFE_Context* ctx, TF_Status* status) { delete ctx; }

TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx, TF_Status* status) {
  TF_DeviceList* list = new TF_DeviceList;
  ctx->context.device_mgr()->ListDeviceAttributes(&list->response);
  return list;
}

void TFE_ContextClearCaches(TFE_Context* ctx) { ctx->context.ClearCaches(); }

void TFE_ContextSetThreadLocalDevicePlacementPolicy(
    TFE_Context* ctx, TFE_ContextDevicePlacementPolicy policy) {
  ctx->context.SetThreadLocalDevicePlacementPolicy(
      static_cast<tensorflow::ContextDevicePlacementPolicy>(policy));
}

// Note: this function looks up a thread local policy. So it should be called in
// the appropriate client thread. In particular, in async mode, it may not be
// safe to call this function from the async EagerExecutor threads.
extern TFE_ContextDevicePlacementPolicy TFE_ContextGetDevicePlacementPolicy(
    TFE_Context* ctx) {
  return static_cast<TFE_ContextDevicePlacementPolicy>(
      ctx->context.GetDevicePlacementPolicy());
}

void TFE_ContextAsyncWait(TFE_Context* ctx, TF_Status* status) {
  status->status = ctx->context.AsyncWait();
}

void TFE_ContextGetStatus(TFE_Context* ctx, TF_Status* status) {
  status->status = ctx->context.GetStatus();
}

void TFE_ContextAsyncClearError(TFE_Context* ctx) {
  ctx->context.ClearAsyncError();
}

TFE_TensorHandle* TFE_NewTensorHandle(TF_Tensor* t, TF_Status* status) {
  tensorflow::Tensor tensor;
  status->status = tensorflow::TF_TensorToTensor(t, &tensor);
  if (!status->status.ok()) return nullptr;
  return new TFE_TensorHandle(tensor, nullptr, nullptr);
}

void TFE_DeleteTensorHandle(TFE_TensorHandle* h) {
  DCHECK(h);
  if (h->handle) {
    h->handle->Unref();
  }
  delete h;
}

TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h) {
  return static_cast<TF_DataType>(h->handle->dtype);
}

int TFE_TensorHandleNumDims(TFE_TensorHandle* h, TF_Status* status) {
  const tensorflow::Tensor* t = nullptr;
  status->status = h->handle->Tensor(&t);
  return t == nullptr ? 0 : t->dims();
}

int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index,
                            TF_Status* status) {
  const tensorflow::Tensor* t = nullptr;
  status->status = h->handle->Tensor(&t);
  return t == nullptr ? 0 : t->dim_size(dim_index);
}

const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h, TF_Status* status) {
  tensorflow::Device* d = nullptr;
  status->status = h->handle->OpDevice(&d);
  return (d == nullptr) ? "/job:localhost/replica:0/task:0/device:CPU:0"
                        : d->name().c_str();
}

TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h, TF_Status* status) {
  // TODO(agarwal): move this implementation inside TFE_TensorHandle.
  tensorflow::Device* d = nullptr;
  tensorflow::Device* op_device = nullptr;
  const tensorflow::Tensor* t = nullptr;
  status->status = h->handle->TensorAndDevice(&t, &d, &op_device);
  if (!status->status.ok()) return nullptr;
  tensorflow::TensorHandle* h_cpu = nullptr;
  if (!IsCPU(d)) {
    status->status = h->handle->CopyToDevice(
        h->handle->Context(), h->handle->Context()->HostCPU(), &h_cpu);
    if (!status->status.ok()) {
      return nullptr;
    }
    status->status = h_cpu->TensorAndDevice(&t, &d, &op_device);
    if (!status->status.ok()) {
      h_cpu->Unref();
      return nullptr;
    }
  }
  TF_Tensor* retval = tensorflow::TF_TensorFromTensor(*t, status);
  if (h_cpu != nullptr) {
    h_cpu->Unref();
  }
  return retval;
}
}  // extern "C"

extern "C" {

TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_or_function_name,
                  TF_Status* status) {
  const char* name = op_or_function_name;  // Shorthand
  const tensorflow::AttrTypeMap* types;
  status->status = tensorflow::AttrTypeMapForOp(name, &types);
  if (status->status.ok()) return new TFE_Op(ctx, name, types);
  if (TF_GetCode(status) == TF_NOT_FOUND) {
    if (ctx->context.FindFunctionByName(name)) {
      status->status = tensorflow::Status::OK();
      return new TFE_Op(ctx, name, nullptr);
    }
  }
  return nullptr;
}

void TFE_DeleteOp(TFE_Op* op) { delete op; }

void TFE_OpSetDevice(TFE_Op* op, const char* device_name, TF_Status* status) {
  tensorflow::Device* d = nullptr;
  if (device_name != nullptr && strlen(device_name) > 0) {
    status->status = op->ctx->context.FindDeviceByName(device_name, &d);
  }
  op->device = d;
}

const char* TFE_OpGetDevice(TFE_Op* op, TF_Status* status) {
  tensorflow::Device* device =
      (op->device == nullptr) ? op->ctx->context.HostCPU() : op->device;
  return device->name().c_str();
}

void TFE_OpSetXLACompilation(TFE_Op* op, unsigned char enable) {
  op->use_xla = enable;
#ifndef TENSORFLOW_EAGER_USE_XLA
  LOG(WARNING) << "This call is a no-op, as the TensorFlow library is not "
                  "built with XLA support.";
#endif  // TENSORFLOW_EAGER_USE_XLA
}

void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* h, TF_Status* status) {
  h->handle->Ref();
  op->inputs.push_back(h->handle);
  op->attrs.NumInputs(op->inputs.size());
}

TF_AttrType TFE_OpGetAttrType(TFE_Op* op, const char* attr_name,
                              unsigned char* is_list, TF_Status* status) {
  TF_AttrType ret;
  if (op->is_function()) {
    status->status = tensorflow::errors::Unimplemented(
        "TODO(apassos): Support for attributes for TensorFlow functions is not "
        "ready yet.");
    return TF_ATTR_INT;  // The compiler requires that we return something.
  }
  status->status =
      tensorflow::AttrTypeByName(*op->attr_types, attr_name, &ret, is_list);
  return ret;
}

TF_AttrType TFE_OpNameGetAttrType(TFE_Context* ctx,
                                  const char* op_or_function_name,
                                  const char* attr_name, unsigned char* is_list,
                                  TF_Status* status) {
  TF_AttrType ret;
  TFE_Op* op = TFE_NewOp(ctx, op_or_function_name, status);
  if (!status->status.ok()) {
    return TF_ATTR_INT;  // Same dummy return as TFE_OpGetAttrType.
  }
  ret = TFE_OpGetAttrType(op, attr_name, is_list, status);
  TFE_DeleteOp(op);
  return ret;
}

void TFE_OpSetAttrString(TFE_Op* op, const char* attr_name, const char* value) {
  op->attrs.Set(attr_name, value);
}

void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name, int64_t value) {
  op->attrs.Set(attr_name, static_cast<int64>(value));
}

void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name, float value) {
  op->attrs.Set(attr_name, value);
}

void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name, unsigned char value) {
  op->attrs.Set(attr_name, (value == 0) ? false : true);
}

void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name, TF_DataType value) {
  op->attrs.Set(attr_name, static_cast<tensorflow::DataType>(value));
}

void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name, const int64_t* dims,
                        const int num_dims, TF_Status* out_status) {
  if (num_dims > tensorflow::TensorShape::MaxDimensions()) {
    TF_SetStatus(out_status, TF_INVALID_ARGUMENT,
                 tensorflow::strings::StrCat(
                     "Value specified for `", attr_name, "` has ", num_dims,
                     " dimensions which is over the limit of ",
                     tensorflow::TensorShape::MaxDimensions(), ".")
                     .c_str());
    return;
  }
  tensorflow::TensorShapeProto proto;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }
  }
  op->attrs.Set(attr_name, proto);
}

void TFE_OpSetAttrFunction(TFE_Op* op, const char* attr_name,
                           const TFE_Op* value) {
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(value->name);
  value->attrs.FillAttrValueMap(func->mutable_attr());
  op->attrs.Set(attr_name, attr_value);
}

#define TFE_OP_SET_ATTR_LIST(fn, type)                                \
  void fn(TFE_Op* op, const char* attr_name, const type* values,      \
          int num_values) {                                           \
    op->attrs.Set(attr_name, tensorflow::gtl::ArraySlice<const type>( \
                                 values, num_values));                \
  }
TFE_OP_SET_ATTR_LIST(TFE_OpSetAttrStringList, char*)
TFE_OP_SET_ATTR_LIST(TFE_OpSetAttrFloatList, float)
#undef TFE_OP_SET_ATTR_LIST

void TFE_OpSetAttrIntList(TFE_Op* op, const char* attr_name,
                          const int64_t* values, int num_values) {
  op->attrs.Set(attr_name,
                tensorflow::gtl::ArraySlice<const int64>(
                    reinterpret_cast<const int64*>(values), num_values));
}

void TFE_OpSetAttrTypeList(TFE_Op* op, const char* attr_name,
                           const TF_DataType* values, int num_values) {
  op->attrs.Set(
      attr_name,
      tensorflow::gtl::ArraySlice<const tensorflow::DataType>(
          reinterpret_cast<const tensorflow::DataType*>(values), num_values));
}

void TFE_OpSetAttrBoolList(TFE_Op* op, const char* attr_name,
                           const unsigned char* values, int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  op->attrs.Set(attr_name,
                tensorflow::gtl::ArraySlice<const bool>(b.get(), num_values));
}

void TFE_OpSetAttrShapeList(TFE_Op* op, const char* attr_name,
                            const int64_t** dims, const int* num_dims,
                            int num_values, TF_Status* out_status) {
  std::unique_ptr<tensorflow::TensorShapeProto[]> proto(
      new tensorflow::TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > tensorflow::TensorShape::MaxDimensions()) {
      TF_SetStatus(out_status, TF_INVALID_ARGUMENT,
                   tensorflow::strings::StrCat(
                       "Value specified for `", attr_name, "` has ", num_dims_i,
                       " dimensions which is over the limit of ",
                       tensorflow::TensorShape::MaxDimensions(), ".")
                       .c_str());
      return;
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  op->attrs.Set(attr_name,
                tensorflow::gtl::ArraySlice<tensorflow::TensorShapeProto>(
                    proto.get(), num_values));
}

void TFE_OpSetAttrFunctionList(TFE_Op* op, const char* attr_name,
                               const TFE_Op** value, int num_values) {
  std::unique_ptr<tensorflow::NameAttrList[]> funcs(
      new tensorflow::NameAttrList[num_values]);
  for (int i = 0; i < num_values; i++) {
    funcs[i].set_name(value[i]->name);
    value[i]->attrs.FillAttrValueMap(funcs[i].mutable_attr());
  }
  op->attrs.Set(attr_name,
                tensorflow::gtl::ArraySlice<const tensorflow::NameAttrList>(
                    funcs.get(), num_values));
}
}  // extern "C"

namespace {

// Initializes the step stats if needed.
void MaybeInitializeStepStats(tensorflow::StepStats* step_stats,
                              tensorflow::EagerContext* ctx) {
  // Lazily initialize the RunMetadata with information about all devices if
  // this is the first call.
  while (step_stats->dev_stats_size() < ctx->devices()->size()) {
    int device_idx = step_stats->dev_stats_size();
    auto* dev_stats = step_stats->add_dev_stats();
    dev_stats->set_device(ctx->devices()->at(device_idx)->name());
  }
}

int StepStatsDeviceIndex(tensorflow::StepStats* step_stats,
                         tensorflow::EagerContext* ctx,
                         tensorflow::Device* device) {
  // Find the current device's index.
  if (device == nullptr) {
    device = ctx->HostCPU();
  }
  for (int i = 0; i < ctx->devices()->size(); ++i) {
    if (ctx->devices()->at(i) == device ||
        ctx->devices()->at(i)->name() == device->name()) {
      return i;
    }
  }
  // TODO(apassos) do not fall back to host CPU if device is unknown.
  return 0;
}

tensorflow::Status ValidateInputTypeAndPlacement(
    tensorflow::EagerContext* ctx, tensorflow::Device* op_device, TFE_Op* op,
    const tensorflow::OpKernel* kernel, tensorflow::RunMetadata* run_metadata) {
  tensorflow::Device* host_device = ctx->HostCPU();
  const tensorflow::MemoryTypeVector& memtypes = kernel->input_memory_types();
  if (memtypes.size() != op->inputs.size()) {
    return tensorflow::errors::InvalidArgument(
        "expected ", memtypes.size(), " inputs, got ", op->inputs.size());
  }
  for (int i = 0; i < op->inputs.size(); ++i) {
    const tensorflow::Device* expected_device =
        memtypes[i] == tensorflow::HOST_MEMORY ? host_device : op_device;
    tensorflow::TensorHandle* handle = op->inputs[i];
    tensorflow::Device* handle_device = nullptr;
    TF_RETURN_IF_ERROR(handle->Device(&handle_device));
    const tensorflow::Device* actual_device =
        handle_device == nullptr ? host_device : handle_device;
    if (expected_device != actual_device) {
      switch (ctx->GetDevicePlacementPolicy()) {
        case tensorflow::DEVICE_PLACEMENT_SILENT_FOR_INT32:
          // TODO(xpan): See if we could bubble python related error up
          // to python level.
          if (handle->dtype == tensorflow::DT_INT32) {
            // Note: enabling silent copies of int32 tensors to match behavior
            // of graph mode.
            break;
          }
          TF_FALLTHROUGH_INTENDED;
        case tensorflow::DEVICE_PLACEMENT_EXPLICIT:
          return tensorflow::errors::InvalidArgument(
              "Tensors on conflicting devices:"
              " cannot compute ",
              op->name, " as input #", i, " was expected to be on ",
              expected_device->name(), " but is actually on ",
              actual_device->name(), " (operation running on ",
              op_device->name(), ")",
              " Tensors can be copied explicitly using .gpu() or .cpu() "
              "methods,"
              " or transparently copied by using tf.enable_eager_execution("
              "device_policy=tfe.DEVICE_PLACEMENT_SILENT). Copying tensors "
              "between devices"
              " may slow down your model");
        case tensorflow::DEVICE_PLACEMENT_WARN:
          LOG(WARNING) << "before computing " << op->name << " input #" << i
                       << " was expected to be on " << expected_device->name()
                       << " but is actually on " << actual_device->name()
                       << " (operation running on " << op_device->name()
                       << "). This triggers a copy which can be a performance "
                          "bottleneck.";
          break;
        case tensorflow::DEVICE_PLACEMENT_SILENT:  // Do nothing.
          break;
      }
      // We are only here if the policy is warn or silent copies, so we should
      // trigger a copy.
      auto pre_time = tensorflow::Env::Default()->NowMicros();
      tensorflow::TensorHandle* copied_tensor = nullptr;
      tensorflow::Status status = tensorflow::EagerCopyToDevice(
          handle, ctx, expected_device->name().c_str(), &copied_tensor);
      if (run_metadata != nullptr) {
        auto* step_stats = run_metadata->mutable_step_stats();
        MaybeInitializeStepStats(step_stats, ctx);
        // Record the sending on the source device for now.
        int device_idx = StepStatsDeviceIndex(step_stats, ctx, handle_device);
        auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
        auto* node_stats = dev_stats->add_node_stats();
        node_stats->set_node_name("_Send");
        node_stats->set_all_start_micros(pre_time);
        node_stats->set_op_end_rel_micros(
            tensorflow::Env::Default()->NowMicros() - pre_time);
      }
      if (!status.ok()) {
        if (copied_tensor != nullptr) copied_tensor->Unref();
        return tensorflow::errors::Internal(
            "Failed copying input tensor from ", actual_device->name(), " to ",
            expected_device->name(), " in order to run ", op->name, ": ",
            status.error_message());
      }
      handle->Unref();
      handle = copied_tensor;
      op->inputs[i] = copied_tensor;
    }
    if (handle->dtype != kernel->input_type(i)) {
      return tensorflow::errors::InvalidArgument(
          "cannot compute ", op->name, " as input #", i,
          " was expected to be a ",
          tensorflow::DataTypeString(kernel->input_type(i)),
          " tensor but is a ", tensorflow::DataTypeString(handle->dtype),
          " tensor");
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Device* SelectDevice(const tensorflow::NodeDef& ndef,
                                 TFE_Context* ctx, TF_Status* status) {
  tensorflow::DeviceSet ds;
  for (tensorflow::Device* d : *ctx->context.devices()) {
    ds.AddDevice(d);
  }
  tensorflow::DeviceTypeVector final_devices;
  status->status = tensorflow::SupportedDeviceTypesForNode(
      ds.PrioritizedDeviceTypeList(), ndef, &final_devices);
  if (!status->status.ok()) {
    return nullptr;
  }
  if (final_devices.empty()) {
    status->status = tensorflow::errors::Internal(
        "Could not find valid device for node ", ndef.DebugString());
    return nullptr;
  }
  for (tensorflow::Device* d : *ctx->context.devices()) {
    if (d->device_type() == final_devices[0].type_string()) {
      return d;
    }
  }
  status->status = tensorflow::errors::Unknown(
      "Could not find a device for node ", ndef.DebugString());
  return nullptr;
}

#ifdef TENSORFLOW_EAGER_USE_XLA
// Synthesizes and returns a wrapper function over `op`, which must be a
// primitive op (e.g. matmul).
//
// The wrapper function conforms to the function signature expected by
// _XlaLaunchOp, with input params ordered by <constants, (variable) args and
// resources>. For example, if the op has input params <Const1, Arg2, Const3,
// Resource4, Arg5>, they will be reordered to <Const1, Const3, Arg2, Arg5,
// Resource4> as the input params to the synthesized function.
//
// It populates `const_input_types`, `arg_input_types` and
// `op_input_to_func_input` based on the reordering results, that the caller can
// use them to build an _XlaLaunchOp. On error, it returns NULL, and sets
// `status` accordingly.
const tensorflow::FunctionDef* OpToFunction(
    TFE_Op* op, std::vector<TF_DataType>* const_input_types,
    std::vector<TF_DataType>* arg_input_types,
    tensorflow::gtl::FlatMap<int, int>* op_input_to_func_input,
    TF_Status* status) {
  DCHECK(!op->is_function());

  tensorflow::FunctionDef fdef;

  // Get the OpDef of the op we are trying to encapsulate.
  TFE_Context* ctx = op->ctx;
  const tensorflow::OpRegistrationData* op_data;
  {
    status->status = ctx->context.FindFunctionOpData(op->name, &op_data);
    if (!status->status.ok()) {
      return nullptr;
    }
  }
  const tensorflow::OpDef& op_def = op_data->op_def;

  tensorflow::OpDef* signature = fdef.mutable_signature();

  // Handle constant inputs.
  const std::unordered_set<string> const_inputs(
      *tensorflow::XlaOpRegistry::CompileTimeConstantInputs(op->name));

  // First add place holders for the input args, so that we can refer to them by
  // position in the next loop. Also tally up the resource inputs.
  int num_resource_inputs = 0;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    if (op_def.input_arg(i).type() == tensorflow::DT_RESOURCE) {
      ++num_resource_inputs;
    }
    signature->add_input_arg();
  }

  // Now we map the input params from `op_def` to `signature`, where the param
  // ordering for `signature` is: <constants, args, resources>.
  int const_index = 0;
  int arg_index = const_inputs.size();
  int resource_index = op_def.input_arg_size() - num_resource_inputs;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    const tensorflow::OpDef::ArgDef& op_input_arg = op_def.input_arg(i);
    tensorflow::OpDef::ArgDef* func_input_arg = nullptr;
    if (const_inputs.find(op_input_arg.name()) != const_inputs.end()) {
      VLOG(1) << "For const input, mapping op input " << i << " to func input "
              << const_index;
      (*op_input_to_func_input)[i] = const_index;
      func_input_arg = signature->mutable_input_arg(const_index++);
      const_input_types->push_back(
          static_cast<TF_DataType>(op->inputs[i]->dtype));
    } else if (op_input_arg.type() == tensorflow::DT_RESOURCE) {
      VLOG(1) << "For resource input, mapping op input " << i
              << " to func input " << resource_index;
      (*op_input_to_func_input)[i] = resource_index;
      func_input_arg = signature->mutable_input_arg(resource_index++);
    } else {
      VLOG(1) << "For arg input, mapping op input " << i << " to func input "
              << arg_index;
      (*op_input_to_func_input)[i] = arg_index;
      func_input_arg = signature->mutable_input_arg(arg_index++);
      arg_input_types->push_back(
          static_cast<TF_DataType>(op->inputs[i]->dtype));
    }

    func_input_arg->set_name(op_input_arg.name());
    func_input_arg->set_type(op->inputs[i]->dtype);
  }
  VLOG(1) << "Added OpDef Inputs: " << fdef.DebugString();

  // Resources args are at the end of the function input params, and we should
  // have iterated over all of them.
  DCHECK_EQ(signature->input_arg_size(), resource_index);

  // Make the synthesized function's name unique.
  signature->set_name(tensorflow::strings::StrCat(
      op_def.name(), func_id_generator.fetch_add(1)));

  // Add the node def and set its input names to match op_def's names.
  const tensorflow::NodeDef& ndef = op->attrs.BuildNodeDef();
  DCHECK_EQ(signature->input_arg_size(), ndef.input_size());
  *fdef.add_node_def() = ndef;
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    fdef.mutable_node_def(0)->set_input(i, op_def.input_arg(i).name());
  }
  VLOG(1) << "Added NodeDef: " << fdef.DebugString();

  // Fix the output names and set output types.
  for (int i = 0; i < op_def.output_arg_size(); ++i) {
    tensorflow::OpDef::ArgDef* arg = signature->add_output_arg();
    const tensorflow::OpDef::ArgDef& op_def_arg = op_def.output_arg(i);
    const string& out_tensor_name = tensorflow::strings::StrCat(
        ndef.name(), ":", op_def_arg.name(), ":", 0);
    arg->set_name(op_def_arg.name());
    (*fdef.mutable_ret())[op_def_arg.name()] = out_tensor_name;
    const string& type_attr = op_def_arg.type_attr();
    if (!type_attr.empty()) {
      auto i = ndef.attr().find(type_attr);
      if (i == ndef.attr().end()) {
        status->status = tensorflow::errors::InvalidArgument(
            tensorflow::strings::StrCat("Could not find attr ", type_attr,
                                        " in NodeDef ", ndef.DebugString()));
        return nullptr;
      }
      arg->set_type(i->second.type());
    }
  }
  VLOG(1) << "Fixed Output names and all types: " << fdef.DebugString();

  status->status = ctx->context.AddFunctionDef(fdef);
  if (!status->status.ok()) return nullptr;
  const auto ret = ctx->context.FindFunctionDef(signature->name());
  DCHECK(ret != nullptr);
  return ret;
}

// Builds an _XLALaunchOp as a wrapper over 'op', so that 'op' can be executed
// via XLA.
std::unique_ptr<TFE_Op> BuildXlaLaunch(TFE_Op* op, TF_Status* status) {
  VLOG(1) << "Creating _XlaLaunchOp for TFE_Op " << op->name;
  auto launch_op =
      std::unique_ptr<TFE_Op>(TFE_NewOp(op->ctx, "_XlaLaunch", status));
  if (TF_GetCode(status) != TF_OK) return nullptr;
  if (op->device) {
    TFE_OpSetDevice(launch_op.get(), op->device->name().c_str(), status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }

  const tensorflow::FunctionDef* fdef;
  { fdef = op->ctx->context.FindFunctionDef(op->name); }
  std::vector<TF_DataType> const_input_types;
  std::vector<TF_DataType> arg_input_types;
  tensorflow::gtl::FlatMap<int, int> op_input_to_func_input;
  if (fdef == nullptr) {
    // See if this is a primitive op, and if so create a function for it, so
    // that _XlaLaunchOp can access it.
    fdef = OpToFunction(op, &const_input_types, &arg_input_types,
                        &op_input_to_func_input, status);
    if (!status->status.ok()) return nullptr;
  } else {
    // TODO(hongm): XlaOpRegistry::CompileTimeConstantInputs() does not work for
    // functions, so we need to find another way to handle constant inputs.
    for (int i = const_input_types.size();
         i < fdef->signature().input_arg_size(); ++i) {
      VLOG(1) << "Adding Targs from input arg " << i;
      const tensorflow::OpDef::ArgDef& arg = fdef->signature().input_arg(i);
      arg_input_types.push_back(static_cast<TF_DataType>(arg.type()));
    }
  }
  DCHECK(fdef != nullptr);

  // Copy inputs and their devices.
  // Since input param reordering may have occurred between `op` and `launch_op`
  // via `op_input_to_func_input`, adjust the actual inputs accordingly.
  launch_op->inputs = op->inputs;
  for (tensorflow::TensorHandle* h : launch_op->inputs) {
    h->Ref();
  }
  if (!op_input_to_func_input.empty()) {
    DCHECK_EQ(op->inputs.size(), op_input_to_func_input.size());
    for (int i = 0; i < op_input_to_func_input.size(); ++i) {
      VLOG(1) << "mapping op input " << i << " to func input "
              << op_input_to_func_input[i];

      launch_op->inputs[op_input_to_func_input[i]] = op->inputs[i];
    }
  }
  launch_op->attrs.NumInputs(op->inputs.size());

  TFE_OpSetAttrTypeList(launch_op.get(), "Tconstants", const_input_types.data(),
                        const_input_types.size());

  // Set Targs and Nresources attrs.
  TFE_OpSetAttrTypeList(launch_op.get(), "Targs", arg_input_types.data(),
                        arg_input_types.size());
  const int num_resource_inputs = fdef->signature().input_arg_size() -
                                  const_input_types.size() -
                                  arg_input_types.size();
  TFE_OpSetAttrInt(launch_op.get(), "Nresources", num_resource_inputs);

  // Set Tresults attr.
  std::vector<TF_DataType> tresults;
  for (const tensorflow::OpDef::ArgDef& arg : fdef->signature().output_arg()) {
    tresults.push_back(static_cast<TF_DataType>(arg.type()));
  }
  TFE_OpSetAttrTypeList(launch_op.get(), "Tresults", tresults.data(),
                        tresults.size());

  // Set function attr.
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(fdef->signature().name());
  launch_op->attrs.Set("function", attr_value);

  return launch_op;
}
#endif  // TENSORFLOW_EAGER_USE_XLA

}  // namespace

extern "C" {

void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
                 TF_Status* status) {
  TFE_Context* ctx = op->ctx;
  status->status = ctx->context.GetStatus();
  if (!status->status.ok()) {
    return;
  }
#ifdef TENSORFLOW_EAGER_USE_XLA
  std::unique_ptr<TFE_Op> xla_launch_op;
  if (op->use_xla && op->name != "_XlaLaunch") {
    xla_launch_op = BuildXlaLaunch(op, status);
    if (!status->status.ok()) {
      return;
    }
    op = xla_launch_op.get();
  }
#endif  // TENSORFLOW_EAGER_USE_XLA
  // Ensure all resource-touching ops run in the device the resource is,
  // regardless of anything else that has been specified. This is identical to
  // the graph mode behavior.
  for (int i = 0; i < op->inputs.size(); ++i) {
    tensorflow::Device* input_op_device = nullptr;
    status->status = op->inputs[i]->OpDevice(&input_op_device);
    if (!status->status.ok()) return;
    VLOG(2) << "for op " << op->name << " input " << i << " "
            << tensorflow::DataTypeString(op->inputs[i]->dtype) << " "
            << (input_op_device == nullptr ? "cpu" : input_op_device->name())
            << " " << (op->device == nullptr ? "cpu" : op->device->name());
    if (op->inputs[i]->dtype == tensorflow::DT_RESOURCE &&
        (input_op_device != op->device || input_op_device == nullptr)) {
      tensorflow::Device* d =
          input_op_device == nullptr ? ctx->context.HostCPU() : input_op_device;
      VLOG(1) << "Changing device of operation " << op->name << " to "
              << d->name() << " because input #" << i
              << " is a resource in this device.";
      op->device = d;
    }
  }
  tensorflow::Device* device = op->device;

  tensorflow::Fprint128 cache_key =
      op->attrs.CacheKey(device == nullptr ? "unspecified" : device->name());
  tensorflow::KernelAndDevice* kernel = ctx->context.GetCachedKernel(cache_key);
  if (kernel == nullptr) {
    const tensorflow::NodeDef& ndef = op->attrs.BuildNodeDef();
    if (device == nullptr) {
      device = SelectDevice(ndef, ctx, status);
      if (!status->status.ok()) {
        return;
      }
    }
    CHECK(device != nullptr);
    if (ctx->context.LogDevicePlacement()) {
      LOG(INFO) << "Executing op " << ndef.op() << " in device "
                << device->name();
    }
    kernel = new tensorflow::KernelAndDevice(ctx->context.GetRendezvous());
    // Knowledge of the implementation of Init (and in-turn
    // FunctionLibraryRuntime::CreateKernel) tells us that ctx->func_lib_def
    // will be accessed, so grab on to the lock.
    // See WARNING comment in Execute (before kernel->Run) - would be nice to
    // rework to avoid this subtlety.
    tensorflow::tf_shared_lock l(*ctx->context.FunctionsMu());
    status->status = tensorflow::KernelAndDevice::Init(
        ndef, ctx->context.func_lib(device), kernel);
    if (!status->status.ok()) {
      delete kernel;
      return;
    }
    // Update output_dtypes inside `kernel`.
    const tensorflow::OpDef* op_def = nullptr;
    const tensorflow::FunctionDef* function_def =
        ctx->context.FuncLibDef()->Find(ndef.op());
    if (function_def != nullptr) {
      op_def = &(function_def->signature());
    }
    if (op_def == nullptr) {
      status->status = OpDefForOp(ndef.op().c_str(), &op_def);
      if (!status->status.ok()) {
        return;
      }
    }
    tensorflow::DataTypeVector input_dtypes;
    status->status = InOutTypesForNode(ndef, *op_def, &input_dtypes,
                                       kernel->mutable_output_dtypes());
    if (!status->status.ok()) {
      return;
    }
    ctx->context.AddKernelToCache(cache_key, kernel);
  }
  const tensorflow::DataTypeVector& output_dtypes = kernel->output_dtypes();
  const int output_dtypes_size = output_dtypes.size();
  if (output_dtypes_size > *num_retvals) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 tensorflow::strings::StrCat("Expecting ", output_dtypes.size(),
                                             " outputs, but *num_retvals is ",
                                             *num_retvals)
                     .c_str());
    return;
  }
  *num_retvals = output_dtypes_size;
  if (device == nullptr) {
    // TODO(apassos) debug how the assignment below might return a different
    // device from the one requested above.
    device = kernel->device();
  }
  status->status = ValidateInputTypeAndPlacement(
      &ctx->context, device, op, kernel->kernel(),
      ctx->context.ShouldStoreMetadata() ? ctx->context.RunMetadataProto()
                                         : nullptr);
  if (!status->status.ok()) return;
  std::unique_ptr<tensorflow::NodeExecStats> maybe_stats;
  if (ctx->context.ShouldStoreMetadata()) {
    maybe_stats.reset(new tensorflow::NodeExecStats);
    maybe_stats->set_node_name(op->name);
    maybe_stats->set_all_start_micros(tensorflow::Env::Default()->NowMicros());
    maybe_stats->set_op_start_rel_micros(0);
    maybe_stats->set_scheduled_micros(tensorflow::Env::Default()->NowMicros());
    // TODO(apassos) track referenced tensors
  }
  if (ctx->context.Async()) {
    // Note that for async mode, execution order will make sure that all
    // input handles are ready before executing them.
    // TODO(agarwal): Consider executing "cheap" kernels inline for performance.
    tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> handle_retvals(
        *num_retvals);
    tensorflow::uint64 id = op->ctx->context.NextId();
    for (int i = 0; i < *num_retvals; ++i) {
      tensorflow::TensorHandle* h =
          new tensorflow::TensorHandle(id, output_dtypes[i], &op->ctx->context);
      retvals[i] = new TFE_TensorHandle(h);
      handle_retvals[i] = h;
    }
    tensorflow::EagerNode* node = new tensorflow::ExecuteNode(
        id, &op->ctx->context, op->device, op->inputs, kernel,
        maybe_stats.release(), output_dtypes, handle_retvals);
    ctx->context.ExecutorAdd(node);
  } else {
    // Execute checks if retvals[i] is nullptr or not to figure if it needs to
    // allocate it.
    tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> handle_retvals(
        *num_retvals);
    status->status = tensorflow::EagerExecute(
        &op->ctx->context, op->device, op->inputs, kernel, maybe_stats.get(),
        handle_retvals.data(), *num_retvals);
    for (int i = 0; i < *num_retvals; ++i) {
      retvals[i] = new TFE_TensorHandle(handle_retvals[i]);
    }
  }
}

TFE_TensorHandle* TFE_TensorHandleCopyToDevice(TFE_TensorHandle* h,
                                               TFE_Context* ctx,
                                               const char* device_name,
                                               TF_Status* status) {
  tensorflow::TensorHandle* handle;
  status->status = tensorflow::EagerCopyToDevice(h->handle, &ctx->context,
                                                 device_name, &handle);
  if (status->status.ok()) {
    return new TFE_TensorHandle(handle);
  }
  return nullptr;
}

void TFE_ContextAddFunctionDef(TFE_Context* ctx,
                               const char* serialized_function_def, size_t size,
                               TF_Status* status) {
  tensorflow::FunctionDef function_def;
  if (!function_def.ParseFromArray(serialized_function_def, size)) {
    status->status =
        tensorflow::errors::InvalidArgument("Invalid FunctionDef proto");
    return;
  }
  status->status = ctx->context.AddFunctionDef(function_def);
}

void TFE_ContextAddFunction(TFE_Context* ctx, TF_Function* function,
                            TF_Status* status) {
  status->status = ctx->context.AddFunctionDef(function->fdef);
}

void TFE_ContextEnableRunMetadata(TFE_Context* ctx) {
  ctx->context.SetShouldStoreMetadata(true);
}

void TFE_ContextDisableRunMetadata(TFE_Context* ctx) {
  ctx->context.SetShouldStoreMetadata(false);
}

}  // extern "C"

TFE_TensorHandle* TFE_NewTensorHandle(const tensorflow::Tensor& t) {
  return new TFE_TensorHandle(t, nullptr, nullptr);
}

const tensorflow::Tensor* TFE_TensorHandleUnderlyingTensorInHostMemory(
    TFE_TensorHandle* h, TF_Status* status) {
  tensorflow::Device* d = nullptr;
  tensorflow::Device* op_device = nullptr;
  const tensorflow::Tensor* t = nullptr;
  status->status = h->handle->TensorAndDevice(&t, &d, &op_device);
  if (!status->status.ok()) return nullptr;
  if (d != nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "TFE_TensorHandle is placed in device (not host) memory. Cannot return "
        "a tensorflow::Tensor");
    return nullptr;
  }
  return t;
}

void TFE_ContextExportRunMetadata(TFE_Context* ctx, TF_Buffer* buf,
                                  TF_Status* status) {
  TFE_ContextAsyncWait(ctx, status);
  if (!status->status.ok()) return;
  tensorflow::mutex_lock ml(*ctx->context.MetadataMu());
  status->status = MessageToBuffer(*ctx->context.RunMetadataProto(), buf);
  ctx->context.RunMetadataProto()->Clear();
}

void TFE_GetResourceHandleShapeAndType(TF_Graph* graph, TF_Output output,
                                       TF_Buffer* output_proto,
                                       TF_Status* status) {
  tensorflow::Node* node = &output.oper->node;
  tensorflow::CppShapeInferenceResult::HandleData handle_data;
  handle_data.set_is_set(true);
  {
    tensorflow::mutex_lock l(graph->mu);
    tensorflow::shape_inference::InferenceContext* ic =
        graph->refiner.GetContext(node);
    CHECK(ic != nullptr);
    CHECK_LT(output.index, ic->num_outputs());
    const auto* shapes_and_types =
        ic->output_handle_shapes_and_types(output.index);
    if (shapes_and_types == nullptr) {
      output_proto->data = nullptr;
      output_proto->length = 0;
      output_proto->data_deallocator = nullptr;
      return;
    }

    for (const auto& p : *shapes_and_types) {
      auto* out_shape_and_type = handle_data.add_shape_and_type();
      ic->ShapeHandleToProto(p.shape, out_shape_and_type->mutable_shape());
      out_shape_and_type->set_dtype(p.dtype);
    }
  }
  status->status = MessageToBuffer(handle_data, output_proto);
}

void TFE_SetResourceHandleShapeAndType(TF_Graph* graph, TF_Output output,
                                       const void* proto, size_t proto_len,
                                       TF_Status* status) {
  tensorflow::CppShapeInferenceResult::HandleData handle_data;
  if (!handle_data.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Couldn't deserialize HandleData proto");
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
    if (status->status.ok()) return;
    shapes_and_types.emplace_back(shape, shape_and_type_proto.dtype());
  }
  ic->set_output_handle_shapes_and_types(output.index, shapes_and_types);
}

namespace {
TFE_Op* GetFunc(TFE_Context* ctx, const tensorflow::NameAttrList& func,
                TF_Status* status) {
  TFE_Op* func_op = TFE_NewOp(ctx, func.name().data(), status);
  for (const auto& attr : func.attr()) {
    if (TF_GetCode(status) != TF_OK) return nullptr;
    SetOpAttrValueScalar(ctx, func_op, attr.second, attr.first.data(), status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }
  return func_op;
}
}  // namespace

namespace tensorflow {
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status) {
  switch (default_value.value_case()) {
    case tensorflow::AttrValue::kS:
      TFE_OpSetAttrString(op, attr_name, default_value.s().data());
      break;
    case tensorflow::AttrValue::kI:
      TFE_OpSetAttrInt(op, attr_name, static_cast<int64_t>(default_value.i()));
      break;
    case tensorflow::AttrValue::kF:
      TFE_OpSetAttrFloat(op, attr_name, default_value.f());
      break;
    case tensorflow::AttrValue::kB:
      TFE_OpSetAttrBool(op, attr_name, default_value.b());
      break;
    case tensorflow::AttrValue::kType:
      TFE_OpSetAttrType(op, attr_name,
                        static_cast<TF_DataType>(default_value.type()));
      break;
    case tensorflow::AttrValue::kShape: {
      const auto& tensor_shape = default_value.shape();
      if (tensor_shape.unknown_rank()) {
        TFE_OpSetAttrShape(op, attr_name, nullptr, -1, status);
      } else {
        const auto num_dims = tensor_shape.dim_size();
        std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
        for (int i = 0; i < num_dims; ++i) {
          dims[i] = tensor_shape.dim(i).size();
        }
        TFE_OpSetAttrShape(op, attr_name, dims.get(), num_dims, status);
      }
    } break;
    case tensorflow::AttrValue::kFunc: {
      const auto func_op = GetFunc(ctx, default_value.func(), status);
      if (TF_GetCode(status) != TF_OK) return;
      // TODO(nareshmodi): TFE_OpSetAttrFunction and TFE_OpSetAttrFunctionList
      // require TFE_Op* and just convert it internally a NameAttrValue, so
      // consider adding an overload to the C API to make this case easier.
      TFE_OpSetAttrFunction(op, attr_name, func_op);
    } break;
    case tensorflow::AttrValue::kList:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::kTensor:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::kPlaceholder:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::VALUE_NOT_SET:
      TF_SetStatus(
          status, TF_UNIMPLEMENTED,
          tensorflow::strings::StrCat("Unable to get setfor default value: ",
                                      default_value.DebugString())
              .data());
  }
}
}  // namespace tensorflow

TFE_Op::~TFE_Op() {
  for (tensorflow::TensorHandle* h : inputs) {
    h->Unref();
  }
}
