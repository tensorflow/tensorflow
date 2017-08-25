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
#include "tensorflow/c/eager/runtime.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
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

struct TFE_Context {
  explicit TFE_Context(TF_Session* s) : session(s) {}

  // TFE_Context is an extension of TF_Session. And TF_Session needs a TF_Graph.
  TF_Session* session;
  tensorflow::Rendezvous* rendezvous;

  tensorflow::mutex functions_mu;
  tensorflow::FunctionLibraryDefinition func_lib_def GUARDED_BY(functions_mu){
      tensorflow::OpRegistry::Global(), {}};

  // One FunctionLibraryRuntime per device.
  // func_libs[i] is the FunctionLibraryRuntime corresponding to
  // session->devices[i].
  std::unique_ptr<tensorflow::ProcessFunctionLibraryRuntime> pflr;

  std::unordered_map<tensorflow::Fprint128, tensorflow::KernelAndDevice*,
                     tensorflow::Fprint128Hasher>
      kernel_cache;

  tensorflow::FunctionLibraryRuntime* func_lib(tensorflow::Device* d) {
    return pflr->GetFLR(d->name());
  }

  const std::vector<tensorflow::Device*>& devices() { return session->devices; }
};

struct TFE_TensorHandle {
  TFE_TensorHandle(const tensorflow::Tensor& t, tensorflow::Device* d)
      : t(t), d(d) {}

  tensorflow::Tensor t;
  // TODO(ashankar): d == nullptr iff local CPU
  // This was expedient, but perhaps worth revisiting ('d' should always be a
  // valid pointer?)
  // This can be done if TFE_NewOp() and the TFE_TensorHandle constructors are
  // provided with the appropriate TFE_Context.
  //
  // TODO(ashankar): Reference count TFE_Context to ensure that 'd' of a
  // TFE_TensorHandle does not outlive the TFE_Context from which it came?
  tensorflow::Device* d;
};

struct TFE_Op {
  TFE_Op(TFE_Context* ctx, const char* op, const tensorflow::AttrTypeMap* t)
      : ctx(ctx), name(op), attrs(op), attr_types(t), device(nullptr) {}

  bool const is_function() const { return attr_types == nullptr; }

  TFE_Context* ctx;  // Must outlive the TFE_Op.
  const char* name;
  tensorflow::AttrBuilder attrs;
  const tensorflow::AttrTypeMap* attr_types;
  std::vector<tensorflow::Tensor> inputs;
  std::vector<tensorflow::Device*> input_devices;
  tensorflow::Device* device;
};

extern "C" {

TFE_Context* TFE_NewContext(const TF_SessionOptions* opts, TF_Status* status) {
  TF_Graph* graph = TF_NewGraph();
  TF_Session* session = TF_NewSession(graph, opts, status);
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
  ret->pflr.reset(new tensorflow::ProcessFunctionLibraryRuntime(
      ret->session->device_mgr, opts->options.env, TF_GRAPH_DEF_VERSION,
      &ret->func_lib_def, {}));
  ret->rendezvous =
      new tensorflow::IntraProcessRendezvous(ret->session->device_mgr);

  return ret;
}

void TFE_DeleteContext(TFE_Context* ctx, TF_Status* status) {
  status->status = tensorflow::Status::OK();
  tensorflow::gtl::STLDeleteValues(&ctx->kernel_cache);
  TF_Graph* graph = ctx->session->graph;
  TF_DeleteSession(ctx->session, status);
  TF_DeleteGraph(graph);
  ctx->rendezvous->Unref();
  delete ctx;
}

TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx, TF_Status* status) {
  return TF_SessionListDevices(ctx->session, status);
}

TFE_TensorHandle* TFE_NewTensorHandle(TF_Tensor* t) {
  return new TFE_TensorHandle(
      tensorflow::TensorCApi::MakeTensor(t->dtype, t->shape, t->buffer),
      nullptr);
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
  if (is_same_device) {
    return new TFE_TensorHandle(h->t, dst_cpu ? nullptr : dstd);
  }
  const bool src_cpu = IsCPU(srcd);
  if (src_cpu == dst_cpu) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        tensorflow::strings::StrCat(
            "TFE_TensorHandleCopyToDevice requires either the source "
            "TFE_TensorHandle be on or the destination device be on CPU "
            "or be the same (they are ",
            DeviceName(srcd), " and ", DeviceName(dstd), " in this call)")
            .c_str());
    return nullptr;
  }
  tensorflow::Tensor* src = &(h->t);
  if (src_cpu) {
    tensorflow::Tensor dst(
        dstd->GetAllocator(tensorflow::AllocatorAttributes()), src->dtype(),
        src->shape());
    tensorflow::Notification n;
    dstd->tensorflow_gpu_device_info()->default_context->CopyCPUTensorToDevice(
        src, dstd, &dst, [status, &n](const tensorflow::Status& s) {
          status->status = s;
          n.Notify();
        });
    n.WaitForNotification();
    return (TF_GetCode(status) == TF_OK) ? new TFE_TensorHandle(dst, dstd)
                                         : nullptr;
  }
  CHECK(dst_cpu);
  tensorflow::Tensor dst(src->dtype(), src->shape());
  tensorflow::Notification n;
  // TODO(ashankar): The Sync() call below may be more aggressive than
  // necessary. It is based on knowledge of implementation details - that
  // GPU devices are implemented using 3 streams - one for host->device copies,
  // one for device->host copies and one for sending operations to the GPU.
  // With that setup, Sync()ing across all 3 streams should be sufficient
  // but more than necessary (since it waits for operations that might have
  // nothing to do with this tensor to complete).
  status->status = srcd->Sync();
  if (!status->status.ok()) return nullptr;
  srcd->tensorflow_gpu_device_info()->default_context->CopyDeviceTensorToCPU(
      src, "IGNORE_MY_TENSOR_NAME", srcd, &dst,
      [status, &n](const tensorflow::Status& s) {
        status->status = s;
        n.Notify();
      });
  n.WaitForNotification();
  return (TF_GetCode(status) == TF_OK) ? new TFE_TensorHandle(dst, nullptr)
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

void TFE_OpSetDevice(TFE_Op* op, TFE_Context* ctx, const char* device_name,
                     TF_Status* status) {
  tensorflow::Device* d = nullptr;
  if (device_name != nullptr && strlen(device_name) > 0) {
    status->status = ctx->session->device_mgr->LookupDevice(device_name, &d);
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
    tensorflow::Device* host_device, tensorflow::Device* op_device, TFE_Op* op,
    const tensorflow::OpKernel* kernel) {
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
      return tensorflow::errors::InvalidArgument(
          "cannot compute ", op->name, " as input #", i,
          " was expected to be on ", expected_device->name(),
          " but is actually on ", actual_device->name(),
          " (operation running on ", op_device->name(), ")");
    }
    if (op->inputs[i].dtype() != kernel->input_type(i)) {
      return tensorflow::errors::InvalidArgument(
          "cannot compute ", op->name, " as input #", i,
          " was expected to be a ",
          tensorflow::DataType_Name(kernel->input_type(i)), " tensor but is a ",
          tensorflow::DataType_Name(op->inputs[i].dtype()), " tensor");
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
  tensorflow::KernelAndDevice* kernel =
      tensorflow::gtl::FindPtrOrNull(ctx->kernel_cache, cache_key);
  if (kernel == nullptr) {
    const tensorflow::NodeDef& ndef = op->attrs.BuildNodeDef();
    kernel = new tensorflow::KernelAndDevice(ctx->rendezvous);
    if (!op->is_function()) {
      status->status =
          tensorflow::KernelAndDevice::InitOp(device, ndef, kernel);
    } else {
      // Knowledge of the implementation of InitFn (and in-turn
      // FunctionLibraryRuntime::CreateKernel) tells us that ctx->func_lib_def
      // will be accessed, so grab on to the lock.
      // See WARNING comment below - would be nice to rework to avoid this
      // subtlety.
      tensorflow::mutex_lock l(ctx->functions_mu);
      status->status = tensorflow::KernelAndDevice::InitFn(
          ndef, ctx->func_lib(device), kernel);
    }
    if (!status->status.ok()) {
      return;
    }
    tensorflow::gtl::InsertOrUpdate(&(ctx->kernel_cache), cache_key, kernel);
  }
  status->status = ValidateInputTypeAndPlacement(ctx->devices()[0], device, op,
                                                 kernel->kernel());
  output_memory_types = &kernel->kernel()->output_memory_types();
  if (!status->status.ok()) {
    return;
  }
  // WARNING: kernel->Run utilizes the FunctionLibraryRuntime
  // (ctx->func_lib(device)), which in turn holds a pointer to func_lib_def,
  // which is GUARDED_BY(ctx->functions_mu). But knowledge of the implementation
  // of FunctionLibraryRuntime tells use that func_lib_def is not accessed by
  // FunctionLibraryRuntime::Run(), so there is no thread-safety concern here.
  // This is quite subtle. Re-work things to make this better?  (Would it make
  // sense for FunctionLibraryRuntime to ensure thread-safe access to
  // FunctionLibraryDefinition?).
  status->status = kernel->Run(&op->inputs, &outputs);
  if (!status->status.ok()) return;
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
