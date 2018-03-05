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
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

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

TFE_ContextDevicePlacementPolicy PlacementPolicy(
    bool soft_placement, TFE_ContextDevicePlacementPolicy original_policy) {
  if (!soft_placement) {
    return original_policy;
  }
  if (original_policy == TFE_DEVICE_PLACEMENT_EXPLICIT ||
      original_policy == TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32) {
    return TFE_DEVICE_PLACEMENT_SILENT;
  }
  return original_policy;
}

extern "C" {

TFE_ContextOptions* TFE_NewContextOptions() { return new TFE_ContextOptions; }

void TFE_ContextOptionsSetConfig(TFE_ContextOptions* options, const void* proto,
                                 size_t proto_len, TF_Status* status) {
  TF_SetConfig(&options->session_options, proto, proto_len, status);
}

void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions* options, TFE_ContextDevicePlacementPolicy policy) {
  options->policy = policy;
}

void TFE_DeleteContextOptions(TFE_ContextOptions* options) { delete options; }

TFE_Context* TFE_NewContext(const TFE_ContextOptions* opts, TF_Status* status) {
  TF_Graph* graph = TF_NewGraph();
  TF_Session* session = TF_NewSession(graph, &opts->session_options, status);
  if (status->status.ok()) {
    if (session->device_mgr == nullptr || session->devices.empty()) {
      status->status = tensorflow::errors::InvalidArgument(
          "Provided TF_SessionOptions are not compatible with eager execution "
          "(perhaps the TF_SessionOptions alluded to session execution in a "
          "remote address space?)");
    }
  }
  if (!status->status.ok()) {
    TF_DeleteGraph(graph);
    return nullptr;
  }

  return new TFE_Context(*opts, session);
}

void TFE_DeleteContext(TFE_Context* ctx, TF_Status* status) {
  status->status = tensorflow::Status::OK();
  {
    tensorflow::mutex_lock ml(ctx->cache_mu);
    tensorflow::gtl::STLDeleteValues(&ctx->kernel_cache);
  }
  TF_Graph* graph = ctx->session->graph;
  TF_DeleteSession(ctx->session, status);
  TF_DeleteGraph(graph);
  ctx->rendezvous->Unref();
  delete ctx;
}

TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx, TF_Status* status) {
  return TF_SessionListDevices(ctx->session, status);
}

void TFE_ContextClearCaches(TFE_Context* ctx) {
  tensorflow::mutex_lock ml(ctx->cache_mu);
  tensorflow::gtl::STLDeleteValues(&ctx->kernel_cache);
}

void TFE_ContextSetThreadLocalDevicePlacementPolicy(
    TFE_Context* ctx, TFE_ContextDevicePlacementPolicy policy) {
  tensorflow::mutex_lock ml(ctx->policy_map_mu);
  ctx->thread_local_policies[std::this_thread::get_id()] = policy;
}

extern TFE_ContextDevicePlacementPolicy TFE_ContextGetDevicePlacementPolicy(
    TFE_Context* ctx) {
  tensorflow::mutex_lock ml(ctx->policy_map_mu);
  auto policy_map_it =
      ctx->thread_local_policies.find(std::this_thread::get_id());
  if (policy_map_it != ctx->thread_local_policies.end()) {
    return policy_map_it->second;
  }
  return ctx->policy;
}

TFE_TensorHandle* TFE_NewTensorHandle(TF_Tensor* t, TF_Status* status) {
  tensorflow::Tensor tensor;
  status->status = tensorflow::TF_TensorToTensor(t, &tensor);
  if (!status->status.ok()) return nullptr;
  return new TFE_TensorHandle(tensor, nullptr, nullptr);
}

void TFE_DeleteTensorHandle(TFE_TensorHandle* h) { delete h; }

TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h) {
  return static_cast<TF_DataType>(h->t.dtype());
}

int TFE_TensorHandleNumDims(TFE_TensorHandle* h, TF_Status* status) {
  status->status = tensorflow::Status::OK();
  return h->t.dims();
}

int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index,
                            TF_Status* status) {
  status->status = tensorflow::Status::OK();
  return h->t.dim_size(dim_index);
}

const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h, TF_Status* status) {
  status->status = tensorflow::Status::OK();
  return (h->op_device == nullptr)
             ? "/job:localhost/replica:0/task:0/device:CPU:0"
             : h->op_device->name().c_str();
}

TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h, TF_Status* status) {
  if (!IsCPU(h->d)) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 tensorflow::strings::StrCat(
                     "TFE_TensorHandle can be resolved iff it is on CPU (this "
                     "handle is on ",
                     h->d->name(),
                     "). Consider using TFE_TensorHandleCopyToDevice to get a "
                     "copy of the tensor on CPU")
                     .c_str());
    return nullptr;
  }
  return tensorflow::TF_TensorFromTensor(h->t, status);
}

TFE_TensorHandle* TFE_TensorHandleCopyToDevice(TFE_TensorHandle* h,
                                               TFE_Context* ctx,
                                               const char* device_name,
                                               TF_Status* status) {
  tensorflow::Device* dstd = ctx->devices()[0];
  if (device_name != nullptr && strlen(device_name) > 0) {
    status->status = ctx->session->device_mgr->LookupDevice(device_name, &dstd);
    if (!status->status.ok()) return nullptr;
  }

  tensorflow::Device* srcd = h->d == nullptr ? ctx->devices()[0] : h->d;
  bool is_same_device =
      (srcd == dstd) || (DeviceName(srcd) == DeviceName(dstd));
  const bool dst_cpu = IsCPU(dstd);
  const bool src_cpu = IsCPU(srcd);
  // both_on_cpu can be true and yet is_same_device is false, if one of src/dst
  // has device type XLA_CPU, and the other CPU.
  const bool both_on_cpu = src_cpu && dst_cpu;
  if (is_same_device || both_on_cpu) {
    dstd = dst_cpu ? nullptr : dstd;
    return new TFE_TensorHandle(h->t, dstd, dstd);
  }
  tensorflow::Tensor* src = &(h->t);
  if (!dst_cpu && (src->dtype() != tensorflow::DT_VARIANT &&
                   !tensorflow::DataTypeCanUseMemcpy(src->dtype()))) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        tensorflow::strings::StrCat("Can't copy Tensor with type ",
                                    tensorflow::DataTypeString(src->dtype()),
                                    " to device ", DeviceName(dstd), ".")
            .c_str());
    return nullptr;
  }
  tensorflow::AllocatorAttributes attr;
  if (src->dtype() == tensorflow::DT_VARIANT) {
    attr.set_on_host(true);
  }
  tensorflow::Tensor dst(dstd->GetAllocator(attr), src->dtype(), src->shape());
  if (src->shape().num_elements() == 0) {
    dstd = dst_cpu ? nullptr : dstd;
    return new TFE_TensorHandle(dst, dstd, dstd);
  }
  tensorflow::DeviceContext* src_device_context = nullptr;
  if (!src_cpu) {
    src_device_context = srcd->tensorflow_gpu_device_info()->default_context;
  }
  tensorflow::DeviceContext* dst_device_context = nullptr;
  if (!dst_cpu) {
    dst_device_context = dstd->tensorflow_gpu_device_info()->default_context;
  }
  // TODO(ashankar): The Sync() call below may be more aggressive than
  // necessary. It is based on knowledge of implementation details - that
  // GPU devices are implemented using 3 streams - one for host->device copies,
  // one for device->host copies and one for sending operations to the GPU.
  // With that setup, Sync()ing across all 3 streams should be sufficient
  // but more than necessary (since it waits for operations that might have
  // nothing to do with this tensor to complete).
  status->status = srcd->Sync();
  tensorflow::Notification n;
  tensorflow::CopyTensor::ViaDMA("copy", src_device_context, dst_device_context,
                                 srcd, dstd, tensorflow::AllocatorAttributes(),
                                 tensorflow::AllocatorAttributes(), src, &dst,
                                 [status, &n](const tensorflow::Status& s) {
                                   status->status = s;
                                   n.Notify();
                                 });
  n.WaitForNotification();
  return (TF_GetCode(status) == TF_OK)
             ? new TFE_TensorHandle(dst, dst_cpu ? nullptr : dstd,
                                    dst_cpu ? nullptr : dstd)
             : nullptr;
}

TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_or_function_name,
                  TF_Status* status) {
  const char* name = op_or_function_name;  // Shorthand
  const tensorflow::AttrTypeMap* types;
  status->status = tensorflow::AttrTypeMapForOp(name, &types);
  if (status->status.ok()) return new TFE_Op(ctx, name, types);
  if (TF_GetCode(status) == TF_NOT_FOUND) {
    tensorflow::mutex_lock l(ctx->functions_mu);
    if (ctx->func_lib_def.Find(name) != nullptr) {
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
    status->status =
        op->ctx->session->device_mgr->LookupDevice(device_name, &d);
    if (!status->status.ok()) return;
  }
  op->device = d;
}

const char* TFE_OpGetDevice(TFE_Op* op, TF_Status* status) {
  tensorflow::Device* device =
      (op->device == nullptr) ? op->ctx->devices()[0] : op->device;
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
  // Questionable heuristic ...
  // - If a device was explicitly set on the op, always use that.
  // - If not, place on the first non-host device seen.
  if (op->device == nullptr && !IsCPU(h->d)) {
    op->device = h->d;
  }
  if (!status->status.ok()) return;
  op->inputs.push_back(h->t);
  op->input_devices.push_back(h->d);
  op->input_op_devices.push_back(h->op_device);
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

namespace {

tensorflow::Status ValidateInputTypeAndPlacement(
    TFE_Context* ctx, tensorflow::Device* host_device,
    tensorflow::Device* op_device, TFE_Op* op,
    const tensorflow::OpKernel* kernel,
    std::vector<TFE_TensorHandle*>* copied_tensors) {
  const tensorflow::MemoryTypeVector& memtypes = kernel->input_memory_types();
  if (memtypes.size() != op->inputs.size()) {
    return tensorflow::errors::InvalidArgument(
        "expected ", memtypes.size(), " inputs, got ", op->inputs.size());
  }
  for (int i = 0; i < op->inputs.size(); ++i) {
    const tensorflow::Device* expected_device =
        memtypes[i] == tensorflow::HOST_MEMORY ? host_device : op_device;
    const tensorflow::Device* actual_device =
        op->input_devices[i] == nullptr ? host_device : op->input_devices[i];
    if (expected_device != actual_device) {
      switch (TFE_ContextGetDevicePlacementPolicy(ctx)) {
        case TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32:
          // TODO(xpan): See if we could bubble python related error up
          // to python level.
          if (op->inputs[i].dtype() == tensorflow::DT_INT32) {
            // Note: enabling silent copies of int32 tensors to match behavior
            // of graph mode.
            break;
          }
          TF_FALLTHROUGH_INTENDED;
        case TFE_DEVICE_PLACEMENT_EXPLICIT:
          return tensorflow::errors::InvalidArgument(
              "Tensors on conflicting devices:"
              " cannot compute ",
              op->name, " as input #", i, " was expected to be on ",
              expected_device->name(), " but is actually on ",
              actual_device->name(), " (operation running on ",
              op_device->name(), ")",
              " Tensors can be copied explicitly using .gpu() or .cpu(),"
              " or transparently copied by using tfe.enable_eager_execution("
              "tfe.DEVICE_PLACEMENT_SILENT). Copying tensors between devices"
              " may slow down your model");
        case TFE_DEVICE_PLACEMENT_WARN:
          LOG(WARNING) << "before computing " << op->name << " input #" << i
                       << " was expected to be on " << expected_device->name()
                       << " but is actually on " << actual_device->name()
                       << " (operation running on " << op_device->name()
                       << "). This triggers a copy which can be a performance "
                          "bottleneck.";
          break;
        case TFE_DEVICE_PLACEMENT_SILENT:  // Do nothing.
          break;
      }
      // We are only here if the policy is warn or silent copies, so we should
      // trigger a copy.
      TFE_TensorHandle original{op->inputs[i], op->input_devices[i],
                                op->device};
      TF_Status* s = TF_NewStatus();
      TFE_TensorHandle* copied_tensor = TFE_TensorHandleCopyToDevice(
          &original, ctx, expected_device->name().c_str(), s);
      if (!s->status.ok()) {
        tensorflow::Status status = s->status;
        delete s;
        return tensorflow::errors::Internal(
            "Failed copying input tensor from ", actual_device->name(), " to ",
            expected_device->name(), " in order to run ", op->name, ": ",
            status.error_message());
      }
      op->inputs[i] = copied_tensor->t;
      copied_tensors->push_back(copied_tensor);
      op->input_devices[i] = copied_tensor->d;
      delete s;
    }
    if (op->inputs[i].dtype() != kernel->input_type(i)) {
      return tensorflow::errors::InvalidArgument(
          "cannot compute ", op->name, " as input #", i,
          " was expected to be a ",
          tensorflow::DataTypeString(kernel->input_type(i)),
          " tensor but is a ",
          tensorflow::DataTypeString(op->inputs[i].dtype()), " tensor");
    }
  }
  return tensorflow::Status::OK();
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
    tensorflow::tf_shared_lock l(ctx->functions_mu);
    status->status = ctx->func_lib_def.LookUp(op->name, &op_data);
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
          static_cast<TF_DataType>(op->inputs[i].dtype()));
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
          static_cast<TF_DataType>(op->inputs[i].dtype()));
    }

    func_input_arg->set_name(op_input_arg.name());
    func_input_arg->set_type(op->inputs[i].dtype());
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

  tensorflow::mutex_lock l(ctx->functions_mu);
  status->status = ctx->func_lib_def.AddFunctionDef(fdef);
  if (!status->status.ok()) return nullptr;
  const auto ret = ctx->func_lib_def.Find(signature->name());
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
  {
    tensorflow::tf_shared_lock l(op->ctx->functions_mu);
    fdef = op->ctx->func_lib_def.Find(op->name);
  }
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
  launch_op->input_devices = op->input_devices;
  launch_op->input_op_devices = op->input_op_devices;
  if (!op_input_to_func_input.empty()) {
    DCHECK_EQ(op->inputs.size(), op_input_to_func_input.size());
    if (!op->input_devices.empty()) {
      DCHECK_EQ(op->input_devices.size(), op_input_to_func_input.size());
    }
    for (int i = 0; i < op_input_to_func_input.size(); ++i) {
      VLOG(1) << "mapping op input " << i << " to func input "
              << op_input_to_func_input[i];

      launch_op->inputs[op_input_to_func_input[i]] = op->inputs[i];
      if (!op->input_devices.empty()) {
        launch_op->input_devices[op_input_to_func_input[i]] =
            op->input_devices[i];
      }
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

tensorflow::Device* SelectDevice(const tensorflow::NodeDef& ndef,
                                 TFE_Context* ctx, TF_Status* status) {
  tensorflow::DeviceSet ds;
  for (tensorflow::Device* d : ctx->devices()) {
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
  for (tensorflow::Device* d : ctx->devices()) {
    if (d->device_type() == final_devices[0].type_string()) {
      return d;
    }
  }
  status->status = tensorflow::errors::Unknown(
      "Could not find a device for node ", ndef.DebugString());
  return nullptr;
}

}  // namespace

void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
                 TF_Status* status) {
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
  TFE_Context* ctx = op->ctx;
  tensorflow::Device* device = op->device;
  // Ensure all resource-touching ops run in the device the resource is,
  // regardless of anything else that has been specified. This is identical to
  // the graph mode behavior.
  for (int i = 0; i < op->inputs.size(); ++i) {
    if (op->inputs[i].dtype() == tensorflow::DT_RESOURCE &&
        op->input_op_devices[i] != device) {
      tensorflow::Device* d = op->input_op_devices[i] == nullptr
                                  ? ctx->devices()[0]
                                  : op->input_op_devices[i];
      VLOG(1) << "Changing device of operation " << op->name << " to "
              << d->name() << " because input #" << i
              << " is a resource in this device.";
      device = d;
      op->device = d;
    }
  }
  if (!ctx->soft_placement && device == nullptr) {
    // TODO(ashankar): ASSUMPTION: ctx->devices()[0] is always CPU
    device = ctx->devices()[0];
  }

  std::vector<tensorflow::Tensor> outputs(1);
  const tensorflow::MemoryTypeVector* output_memory_types = nullptr;
  tensorflow::Fprint128 cache_key =
      op->attrs.CacheKey(device == nullptr ? "unspecified" : device->name());
  tensorflow::KernelAndDevice* kernel;
  {
    tensorflow::tf_shared_lock l(ctx->cache_mu);
    kernel = tensorflow::gtl::FindPtrOrNull(ctx->kernel_cache, cache_key);
  }
  if (kernel == nullptr) {
    const tensorflow::NodeDef& ndef = op->attrs.BuildNodeDef();
    if (ctx->soft_placement && device == nullptr) {
      device = SelectDevice(ndef, ctx, status);
      if (!status->status.ok()) {
        return;
      }
    }
    CHECK(device != nullptr);
    if (ctx->log_device_placement) {
      LOG(INFO) << "Executing op " << ndef.op() << " in device "
                << device->name();
    }
    kernel = new tensorflow::KernelAndDevice(ctx->rendezvous);
    // Knowledge of the implementation of Init (and in-turn
    // FunctionLibraryRuntime::CreateKernel) tells us that ctx->func_lib_def
    // will be accessed, so grab on to the lock.
    // See WARNING comment below - would be nice to rework to avoid this
    // subtlety.
    tensorflow::tf_shared_lock l(ctx->functions_mu);
    status->status =
        tensorflow::KernelAndDevice::Init(ndef, ctx->func_lib(device), kernel);
    if (!status->status.ok()) {
      delete kernel;
      return;
    }
    // Update output_dtypes inside `kernel`.
    const tensorflow::OpDef* op_def = nullptr;
    const tensorflow::FunctionDef* function_def =
        ctx->func_lib_def.Find(ndef.op());
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
                                       kernel->output_dtypes());
    if (!status->status.ok()) {
      return;
    }
    tensorflow::mutex_lock ml(ctx->cache_mu);
    tensorflow::gtl::InsertOrUpdate(&(ctx->kernel_cache), cache_key, kernel);
  }
  if (device == nullptr) {
    // TODO(apassos) debug how the assignment below might return a different
    // device from the one requested above.
    device = kernel->device();
  }

  std::vector<TFE_TensorHandle*> copied_tensors;
  status->status = ValidateInputTypeAndPlacement(
      ctx, ctx->devices()[0], device, op, kernel->kernel(), &copied_tensors);
  output_memory_types = &kernel->kernel()->output_memory_types();
  if (!status->status.ok()) {
    for (auto* t : copied_tensors) {
      TFE_DeleteTensorHandle(t);
    }
    return;
  }
  std::unique_ptr<tensorflow::NodeExecStats> maybe_stats;
  if (ctx->should_store_metadata.load()) {
    maybe_stats.reset(new tensorflow::NodeExecStats);
    maybe_stats->set_node_name(op->name);
    maybe_stats->set_all_start_micros(tensorflow::Env::Default()->NowMicros());
    maybe_stats->set_op_start_rel_micros(0);
    maybe_stats->set_scheduled_micros(tensorflow::Env::Default()->NowMicros());
    // TODO(apassos) track referenced tensors
  }
  // WARNING: kernel->Run utilizes the FunctionLibraryRuntime
  // (ctx->func_lib(device)), which in turn holds a pointer to func_lib_def,
  // which is GUARDED_BY(ctx->functions_mu). But knowledge of the implementation
  // of FunctionLibraryRuntime tells us that func_lib_def is not accessed by
  // FunctionLibraryRuntime::Run(), so there is no thread-safety concern here.
  // This is quite subtle. Re-work things to make this better?  (Would it make
  // sense for FunctionLibraryRuntime to ensure thread-safe access to
  // FunctionLibraryDefinition?).  TODO(apassos) figure out how to record stats
  // for ops which are a part of functions.
  status->status = kernel->Run(&op->inputs, &outputs, maybe_stats.get());
  for (auto* t : copied_tensors) {
    TFE_DeleteTensorHandle(t);
  }
  if (!status->status.ok()) return;
  if (maybe_stats != nullptr) {
    maybe_stats->set_op_end_rel_micros(tensorflow::Env::Default()->NowMicros() -
                                       maybe_stats->all_start_micros());
    tensorflow::mutex_lock ml(ctx->metadata_mu);
    if (ctx->should_store_metadata.load()) {
      auto* step_stats = ctx->run_metadata.mutable_step_stats();
      // Lazily initialize the RunMetadata with information about all devices if
      // this is the first call.
      while (step_stats->dev_stats_size() < ctx->devices().size()) {
        step_stats->add_dev_stats();
      }
      // Find the current device's index.
      int device_idx = 0;
      for (int i = 0; i < ctx->devices().size(); ++i) {
        if (ctx->devices()[i] == device) {
          device_idx = i;
          break;
        }
      }
      // Populate the device stats for this device.
      auto* dev_stats = step_stats->mutable_dev_stats(device_idx);
      dev_stats->set_device(device->name());
      *dev_stats->add_node_stats() = *maybe_stats;
    }
  }
  *num_retvals = std::min<int>(*num_retvals, outputs.size());
  for (int i = 0; i < *num_retvals; ++i) {
    tensorflow::Device* d = IsCPU(device) ? nullptr : device;
    if (d != nullptr && output_memory_types != nullptr &&
        (*output_memory_types)[i] == tensorflow::HOST_MEMORY) {
      d = nullptr;
    }
    retvals[i] = new TFE_TensorHandle(outputs[i], d, device);
  }
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
  tensorflow::mutex_lock l(ctx->functions_mu);
  status->status = ctx->func_lib_def.AddFunctionDef(function_def);
}

void TFE_ContextAddFunction(TFE_Context* ctx, TF_Function* function,
                            TF_Status* status) {
  tensorflow::mutex_lock l(ctx->functions_mu);
  status->status = ctx->func_lib_def.AddFunctionDef(function->fdef);
}

}  // extern "C"

TFE_TensorHandle* TFE_NewTensorHandle(const tensorflow::Tensor& t) {
  return new TFE_TensorHandle(t, nullptr, nullptr);
}

const tensorflow::Tensor* TFE_TensorHandleUnderlyingTensorInHostMemory(
    TFE_TensorHandle* h, TF_Status* status) {
  if (h->d != nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "TFE_TensorHandle is placed in device (not host) memory. Cannot return "
        "a tensorflow::Tensor");
    return nullptr;
  }
  return &h->t;
}

void TFE_ContextEnableRunMetadata(TFE_Context* ctx) {
  ctx->should_store_metadata.store(true);
}

void TFE_ContextDisableRunMetadata(TFE_Context* ctx) {
  tensorflow::mutex_lock ml(ctx->metadata_mu);
  ctx->should_store_metadata.store(false);
  ctx->run_metadata.Clear();
}

void TFE_ContextExportRunMetadata(TFE_Context* ctx, TF_Buffer* buf,
                                  TF_Status* status) {
  tensorflow::mutex_lock ml(ctx->metadata_mu);
  status->status = MessageToBuffer(ctx->run_metadata, buf);
  ctx->run_metadata.Clear();
}
