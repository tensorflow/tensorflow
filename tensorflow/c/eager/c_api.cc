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
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

using tensorflow::int64;
using tensorflow::string;

namespace {
bool IsCPU(tensorflow::Device* d) {
  return d == nullptr || d->tensorflow_gpu_device_info() == nullptr;
}

string DeviceName(tensorflow::Device* d) {
  return (d == nullptr) ? "cpu:0" : d->name();
}
}  // namespace

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

  TFE_Context* ret = new TFE_Context(session);
  ret->policy = opts->policy;
  ret->pflr.reset(new tensorflow::ProcessFunctionLibraryRuntime(
      ret->session->device_mgr, opts->session_options.options.env,
      TF_GRAPH_DEF_VERSION, &ret->func_lib_def, {}));
  ret->rendezvous =
      new tensorflow::IntraProcessRendezvous(ret->session->device_mgr);

  return ret;
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

TFE_TensorHandle* TFE_NewTensorHandle(TF_Tensor* t, TF_Status* status) {
  tensorflow::Tensor tensor;
  status->status = tensorflow::TF_TensorToTensor(t, &tensor);
  if (!status->status.ok()) return nullptr;
  return new TFE_TensorHandle(tensor, nullptr);
}

void TFE_DeleteTensorHandle(TFE_TensorHandle* h) { delete h; }

TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h) {
  return static_cast<TF_DataType>(h->t.dtype());
}

int TFE_TensorHandleNumDims(TFE_TensorHandle* h) { return h->t.dims(); }

int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index) {
  return h->t.dim_size(dim_index);
}

const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h) {
  // This might be a bit confusing as a tensor on CPU can sometimes return
  // "CPU:0" and sometimes "/job:localhost/replica:0/task:0/cpu:0".
  // TODO(ashankar): Figure out which one would be nicer.
  return (h->d == nullptr) ? "CPU:0" : h->d->name().c_str();
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
  if (is_same_device) {
    return new TFE_TensorHandle(h->t, dst_cpu ? nullptr : dstd);
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
    return new TFE_TensorHandle(dst, dst_cpu ? nullptr : dstd);
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
             ? new TFE_TensorHandle(dst, dst_cpu ? nullptr : dstd)
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

static void TFE_OpSetDeviceHelper(TFE_Op* op, tensorflow::Device* device,
                                  TF_Status* status) {
  // Questionable heuristic: Place the op on the same device as the first input
  // placed outside of host memory?
  if (IsCPU(op->device) && !IsCPU(device)) {
    op->device = device;
  }
}

void TFE_OpSetDevice(TFE_Op* op, const char* device_name, TF_Status* status) {
  tensorflow::Device* d = nullptr;
  if (device_name != nullptr && strlen(device_name) > 0) {
    status->status =
        op->ctx->session->device_mgr->LookupDevice(device_name, &d);
    if (!status->status.ok()) return;
  }
  TFE_OpSetDeviceHelper(op, d, status);
}

void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* h, TF_Status* status) {
  TFE_OpSetDeviceHelper(op, h->d, status);
  if (!status->status.ok()) return;
  op->inputs.push_back(h->t);
  op->input_devices.push_back(h->d);
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
      tensorflow::AttrTypeByName(op->attr_types, attr_name, &ret, is_list);
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
      switch (ctx->policy) {
        case TFE_DEVICE_PLACEMENT_EXPLICIT:
          // TODO(xpan): See if we could bubble python related error up
          // to python level.
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
      TFE_TensorHandle original{op->inputs[i], op->input_devices[i]};
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
}  // namespace

void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
                 TF_Status* status) {
  TFE_Context* ctx = op->ctx;
  // TODO(ashankar): ASSUMPTION: ctx->devices()[0] is always CPU
  tensorflow::Device* device =
      (op->device == nullptr) ? ctx->devices()[0] : op->device;
  std::vector<tensorflow::Tensor> outputs(1);
  const tensorflow::MemoryTypeVector* output_memory_types = nullptr;
  tensorflow::Fprint128 cache_key = op->attrs.CacheKey(device->name());
  tensorflow::KernelAndDevice* kernel;
  {
    tensorflow::tf_shared_lock l(ctx->cache_mu);
    kernel = tensorflow::gtl::FindPtrOrNull(ctx->kernel_cache, cache_key);
  }
  if (kernel == nullptr) {
    const tensorflow::NodeDef& ndef = op->attrs.BuildNodeDef();
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
    tensorflow::mutex_lock ml(ctx->cache_mu);
    tensorflow::gtl::InsertOrUpdate(&(ctx->kernel_cache), cache_key, kernel);
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
    retvals[i] = new TFE_TensorHandle(outputs[i], d);
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
  return new TFE_TensorHandle(t, nullptr);
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
