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

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/custom_device.h"
#include "tensorflow/core/common_runtime/eager/custom_device_op_handler.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/version.h"

// "tensorflow/core/platform/platform.h" must be included first before using
// PLATFORM_GOOGLE, IS_MOBILE_PLATFORM, etc.
#if defined(PLATFORM_GOOGLE) && !defined(LIBTPU_ON_GCE)
#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"
#include "tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.h"
#endif  // PLATFORM_GOOGLE && !LIBTPU_ON_GCE

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"
#endif  // !IS_MOBILE_PLATFORM

using tensorflow::string;

namespace {

string DeviceName(const tensorflow::Device* d) {
  return (d == nullptr) ? "cpu:0" : d->name();
}

// Annotate eager runtime construction context to the given `function_def` as
// an attribute.
void AnnotateEagerRuntimeConstructionContext(
    tensorflow::FunctionDef& function_def) {
  tensorflow::AttrValue value;
  SetAttrValue("kEagerRuntime", &value);
  (*function_def.mutable_attr())["_construction_context"] = value;
}

}  // namespace

extern "C" {

TFE_ContextOptions* TFE_NewContextOptions() { return new TFE_ContextOptions; }

void TFE_ContextOptionsSetConfig(TFE_ContextOptions* options, const void* proto,
                                 size_t proto_len, TF_Status* status) {
  TF_SetConfig(&options->session_options, proto, proto_len, status);
}

void TFE_ContextOptionsSetAsync(TFE_ContextOptions* options,
                                unsigned char enable) {
  options->async = enable;
}

void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions* options, TFE_ContextDevicePlacementPolicy policy) {
  options->device_placement_policy = policy;
}

void TFE_DeleteContextOptions(TFE_ContextOptions* options) { delete options; }

TFE_Context* TFE_NewContext(const TFE_ContextOptions* opts, TF_Status* status) {
  if (opts->use_tfrt) {
#if defined(PLATFORM_GOOGLE) && !defined(LIBTPU_ON_GCE)
    tfrt::tf::ContextInterface* tfrt_context = new tfrt::tf::ContextInterface(
        opts->session_options.options,
        static_cast<tensorflow::ContextDevicePlacementPolicy>(
            opts->device_placement_policy),
        opts->async, opts->use_tfrt_distributed_runtime);
#if !defined(IS_MOBILE_PLATFORM)
    tfrt_context->SetDistributedManager(
        tfrt::tf::CreateDistributedManagerContext(
            tfrt_context->GetCoreRuntime()->GetHostContext()));
#endif  // !IS_MOBILE_PLATFORM
    return tensorflow::wrap(tfrt_context);
#else
    status->status = tensorflow::errors::Unimplemented("TFRT is not supported");
    return nullptr;
#endif  // PLATFORM_GOOGLE && !LIBTPU_ON_GCE
  }
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  status->status = tensorflow::DeviceFactory::AddDevices(
      opts->session_options.options, "/job:localhost/replica:0/task:0",
      &devices);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr(
      new tensorflow::DynamicDeviceMgr(std::move(devices)));

  tensorflow::Rendezvous* r =
      new tensorflow::IntraProcessRendezvous(device_mgr.get());
  tensorflow::EagerContext* eager_context = new tensorflow::EagerContext(
      opts->session_options.options,
      static_cast<tensorflow::ContextDevicePlacementPolicy>(
          opts->device_placement_policy),
      opts->async, device_mgr.release(),
      /*device_mgr_owned*/ true, r,
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/opts->run_eager_op_as_function,
      /*jit_compile_rewrite=*/opts->jit_compile_rewrite);
#if !defined(IS_MOBILE_PLATFORM)
  eager_context->SetDistributedManager(
      std::make_unique<tensorflow::EagerContextDistributedManager>(
          eager_context));
#endif  // !IS_MOBILE_PLATFORM
  return tensorflow::wrap(eager_context);
}

void TFE_DeleteContext(TFE_Context* ctx) {
  if (ctx == nullptr) {
    return;
  }

  // ctx->RefCountIsOne() should be true here.
  tensorflow::unwrap(ctx)->Release();
}

TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx, TF_Status* status) {
  TF_DeviceList* l = new TF_DeviceList;
  tensorflow::unwrap(ctx)->ListDevices(&l->response);
  return l;
}

void TFE_ContextClearCaches(TFE_Context* ctx) {
  tensorflow::unwrap(ctx)->ClearCachesAndThreadExecutors();
}

// Set server_def on the context, possibly updating it.
TF_CAPI_EXPORT extern void TFE_ContextSetServerDef(TFE_Context* ctx,
                                                   int keep_alive_secs,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  tensorflow::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  }
  status->status =
      tensorflow::unwrap(ctx)->GetDistributedManager()->SetOrUpdateServerDef(
          server_def, /*reset_context=*/true, keep_alive_secs);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextUpdateServerDef(TFE_Context* ctx,
                                                      int keep_alive_secs,
                                                      const void* proto,
                                                      size_t proto_len,
                                                      TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  tensorflow::ServerDef server_def;
  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  } else if (context->GetContextId() ==
             tensorflow::EagerContext::kInvalidContextId) {
    status->status = tensorflow::errors::InvalidArgument(
        "Trying to update a context with invalid context id.");
  }
  status->status =
      tensorflow::unwrap(ctx)->GetDistributedManager()->SetOrUpdateServerDef(
          server_def, /*reset_context=*/false, keep_alive_secs);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern bool TFE_ContextCheckAlive(TFE_Context* ctx,
                                                 const char* worker_name,
                                                 TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
  return false;
#else   // !defined(IS_MOBILE_PLATFORM)
  bool is_alive;
  status->status =
      tensorflow::unwrap(ctx)->GetDistributedManager()->CheckRemoteAlive(
          worker_name, &is_alive);
  return is_alive;
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextAsyncWait(TFE_Context* ctx,
                                                TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::Status::OK();
#else   // !defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::unwrap(ctx)->AsyncWait();
#endif  // !IS_MOBILE_PLATFORM
}

void TFE_ContextSetThreadLocalDevicePlacementPolicy(
    TFE_Context* ctx, TFE_ContextDevicePlacementPolicy policy) {
  tensorflow::unwrap(ctx)->SetThreadLocalDevicePlacementPolicy(
      static_cast<tensorflow::ContextDevicePlacementPolicy>(policy));
}

// Note: this function looks up a thread local policy. So it should be called in
// the appropriate client thread. In particular, in async mode, it may not be
// safe to call this function from the async EagerExecutor threads.
extern TFE_ContextDevicePlacementPolicy TFE_ContextGetDevicePlacementPolicy(
    TFE_Context* ctx) {
  return static_cast<TFE_ContextDevicePlacementPolicy>(
      tensorflow::unwrap(ctx)->GetDevicePlacementPolicy());
}

TFE_TensorHandle* TFE_NewTensorHandle(const TF_Tensor* t, TF_Status* status) {
  tensorflow::Tensor tensor;
  status->status = tensorflow::TF_TensorToTensor(t, &tensor);
  if (!status->status.ok()) return nullptr;

  return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
}

void TFE_DeleteTensorHandle(TFE_TensorHandle* h) {
  if (h == nullptr) return;

  tensorflow::profiler::TraceMe activity(
      "TFE_DeleteTensorHandle", tensorflow::profiler::TraceMeLevel::kInfo);
  if (h) {
    tensorflow::unwrap(h)->Release();
  }
}

TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h) {
  return static_cast<TF_DataType>(tensorflow::unwrap(h)->DataType());
}

int TFE_TensorHandleNumDims(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int num_dims = -1;
  status->status = tensorflow::unwrap(h)->NumDims(&num_dims);
  return num_dims;
}

int64_t TFE_TensorHandleNumElements(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int64_t num_elements = -1;
  status->status = tensorflow::unwrap(h)->NumElements(&num_elements);
  return num_elements;
}

int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index,
                            TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int64_t dim = -1;
  status->status = tensorflow::unwrap(h)->Dim(dim_index, &dim);
  return dim;
}

const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return tensorflow::unwrap(h)->DeviceName(&status->status);
}

const char* TFE_TensorHandleBackingDeviceName(TFE_TensorHandle* h,
                                              TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return tensorflow::unwrap(h)->BackingDeviceName(&status->status);
}

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_TensorHandleCopySharingTensor(
    TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  return tensorflow::wrap(tensorflow::unwrap(h)->Copy());
}

TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  tensorflow::AbstractTensorInterface* t =
      tensorflow::unwrap(h)->Resolve(&status->status);
  if (t == nullptr) {
    return nullptr;
  }

  return new TF_Tensor{t};
}

void* TFE_TensorHandleDevicePointer(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::ImmediateExecutionTensorHandle* unwrapped_handle =
      tensorflow::unwrap(h);
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (tensorflow::CustomDeviceTensorHandle::classof(unwrapped_handle)) {
    return tensorflow::down_cast<tensorflow::CustomDeviceTensorHandle*>(
               unwrapped_handle)
        ->DevicePointer();
  }
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (!tensorflow::TensorHandle::classof(unwrapped_handle)) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::TensorHandleFromInterface(unwrapped_handle);

  if (handle->Type() != tensorflow::TensorHandle::LOCAL) {
    status->status = tensorflow::errors::InvalidArgument(
        "TFE_TensorHandleDevicePointer may not be called on a ",
        handle->TypeString(), " tensor handle.");
    return nullptr;
  }
  tensorflow::Device* device(handle->device());
  if (device != nullptr) {
    status->status = device->Sync();
    if (!status->status.ok()) {
      return nullptr;
    }
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }
  return const_cast<void*>(
      static_cast<const void*>(tensor->tensor_data().data()));
}

namespace tensorflow {
namespace {
class CustomDeviceAPI : public tensorflow::CustomDevice {
 public:
  CustomDeviceAPI(TFE_Context* context, TFE_CustomDevice device, void* info,
                  string name)
      : context_(context), device_(device), info_(info), name_(name) {}

  ~CustomDeviceAPI() override { device_.delete_device(info_); }

  const string& name() override { return name_; }

  tensorflow::Status CopyTensorToDevice(
      ImmediateExecutionTensorHandle* handle,
      ImmediateExecutionTensorHandle** result) override {
    handle->Ref();
    TF_Status status;
    TFE_TensorHandle* result_handle = device_.copy_tensor_to_device(
        context_, tensorflow::wrap(handle), &status, info_);
    handle->Release();
    if (!status.status.ok()) return status.status;
    *result = tensorflow::unwrap(result_handle);
    (*result)->Ref();
    TFE_DeleteTensorHandle(result_handle);
    return status.status;
  }

  tensorflow::Status CopyTensorFromDevice(
      ImmediateExecutionTensorHandle* handle,
      const tensorflow::string& target_device_name,
      ImmediateExecutionTensorHandle** result) override {
    TF_Status status;
    handle->Ref();
    TFE_TensorHandle* result_handle = device_.copy_tensor_from_device(
        context_, tensorflow::wrap(handle), target_device_name.c_str(), &status,
        info_);
    handle->Release();
    if (!status.status.ok()) return status.status;
    *result = tensorflow::unwrap(result_handle);
    (*result)->Ref();
    TFE_DeleteTensorHandle(result_handle);
    return status.status;
  }

  tensorflow::Status Execute(const ImmediateExecutionOperation* op,
                             ImmediateExecutionTensorHandle** retvals,
                             int* num_retvals) override {
    std::vector<TFE_TensorHandle*> outputs(*num_retvals);
    TF_Status status;
    device_.execute(tensorflow::wrap(op), num_retvals, outputs.data(), &status,
                    info_);
    if (status.status.ok()) {
      for (int i = 0; i < *num_retvals; ++i) {
        retvals[i] = tensorflow::unwrap(outputs[i]);
        retvals[i]->Ref();
        TFE_DeleteTensorHandle(outputs[i]);
      }
    }
    return status.status;
  }

  tensorflow::Status Pack(absl::Span<ImmediateExecutionTensorHandle*> handles,
                          ImmediateExecutionTensorHandle** result) override {
    TF_Status status;
    *result = tensorflow::unwrap(device_.pack(context_,
                                              tensorflow::wrap(handles.data()),
                                              handles.size(), &status, info_));
    return status.status;
  }

 private:
  TFE_Context* context_;
  TFE_CustomDevice device_;
  void* info_;
  string name_;
};

// An adapter which wraps the shape/data produced by C custom devices and uses
// it to implement custom device methods.
class CAPICustomDeviceTensorHandle
    : public tensorflow::CustomDeviceTensorHandle {
 public:
  CAPICustomDeviceTensorHandle(tensorflow::ImmediateExecutionContext* context,
                               tensorflow::CustomDevice* device,
                               tensorflow::DataType dtype, void* data,
                               TFE_CustomDeviceTensorHandleMethods methods)
      : tensorflow::CustomDeviceTensorHandle(context, device, dtype),
        data_(data),
        methods_(methods) {}

  ~CAPICustomDeviceTensorHandle() override { methods_.deallocator(data_); }
  void* DevicePointer() const override { return data_; }
  Status NumDims(int* num_dims) const override {
    TF_Status s;
    *num_dims = methods_.num_dims(data_, &s);
    return s.status;
  }
  Status Dim(int dim_index, int64_t* dim) const override {
    TF_Status s;
    *dim = methods_.dim(data_, dim_index, &s);
    return s.status;
  }

  bool PreferCustomSummarizer() const override {
    return methods_.summarize != nullptr;
  }

  Status SummarizeValue(std::string& summary) const override {
    if (methods_.summarize == nullptr) {
      return tensorflow::CustomDeviceTensorHandle::SummarizeValue(summary);
    }
    TF_Status c_status;
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> summary_buffer(
        methods_.summarize(data_, &c_status), TF_DeleteBuffer);
    if (!c_status.status.ok()) {
      return c_status.status;
    }
    summary = std::string(reinterpret_cast<const char*>(summary_buffer->data),
                          summary_buffer->length);
    return Status::OK();
  }

 private:
  void* const data_;
  const TFE_CustomDeviceTensorHandleMethods methods_;
};

}  // namespace
}  // namespace tensorflow

TFE_TensorHandle* TFE_NewCustomDeviceTensorHandle(
    TFE_Context* ctx, const char* device_name, TF_DataType dtype, void* data,
    TFE_CustomDeviceTensorHandleMethods methods, TF_Status* status) {
  tensorflow::ImmediateExecutionContext* context = tensorflow::unwrap(ctx);
  tensorflow::CustomDevice* device = nullptr;
  if (!context->GetCustomDeviceOpHandler().FindCustomDeviceFromName(device_name,
                                                                    &device)) {
    methods.deallocator(data);
    status->status =
        tensorflow::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }
  return tensorflow::wrap(new tensorflow::CAPICustomDeviceTensorHandle(
      context, device, *reinterpret_cast<tensorflow::DataType*>(&dtype), data,
      methods));
}

TFE_TensorHandle* TFE_NewTensorHandleFromDeviceMemory(
    TFE_Context* ctx, const char* device_name, TF_DataType dtype,
    const int64_t* dims, int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg, TF_Status* status) {
  tensorflow::Device* device = nullptr;
  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  status->status = context->FindDeviceFromName(device_name, &device);
  if (!status->status.ok()) {
    deallocator(data, len, deallocator_arg);
    status->status =
        tensorflow::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }
  std::vector<int64_t> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<int64_t>(dims[i]);
  }

  // TODO(apassos) do we need to wrap the deallocator here to make sure to sync
  // the device?
  TF_ManagedBuffer* buf =
      new TF_ManagedBuffer(data, len, deallocator, deallocator_arg,
                           /*owns_memory=*/false);

  tensorflow::Tensor t(static_cast<tensorflow::DataType>(dtype),
                       tensorflow::TensorShape(dimvec), buf);
  buf->Unref();
  return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(
      std::move(t), device, device, context));
}

// This function will block till the operation that produces `h` has
// completed. This is only valid on local TFE_TensorHandles. Returns the size in
// bytes of the memory pointed to by the device pointer returned above.
size_t TFE_TensorHandleDeviceMemorySize(TFE_TensorHandle* h,
                                        TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return 0;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::TensorHandleFromInterface(tensorflow::unwrap(h));
  if (handle->Type() != tensorflow::TensorHandle::LOCAL) {
    status->status = tensorflow::errors::InvalidArgument(
        "TFE_TensorHandleDeviceMemorySize may not be called on a ",
        handle->TypeString(), " tensor handle.");
    return 0;
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return 0;
  }
  return tensor->TotalBytes();
}

TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_or_function_name,
                  TF_Status* status) {
  tensorflow::ImmediateExecutionOperation* new_op =
      tensorflow::unwrap(ctx)->CreateOperation();
  status->status = new_op->Reset(op_or_function_name, nullptr);
  if (!status->status.ok()) {
    new_op->Release();
    new_op = nullptr;
  }
  return tensorflow::wrap(new_op);
}

void TFE_DeleteOp(TFE_Op* op) {
  if (op == nullptr) {
    return;
  }

  tensorflow::unwrap(op)->Release();
}

const char* TFE_OpGetName(const TFE_Op* op, TF_Status* status) {
  return tensorflow::unwrap(op)->Name().c_str();
}

TFE_Context* TFE_OpGetContext(const TFE_Op* op, TF_Status* status) {
  return tensorflow::wrap(tensorflow::unwrap(op)->GetContext());
}

void TFE_OpSetDevice(TFE_Op* op, const char* device_name, TF_Status* status) {
  status->status = tensorflow::unwrap(op)->SetDeviceName(device_name);
}

const char* TFE_OpGetDevice(const TFE_Op* op, TF_Status* status) {
  return tensorflow::unwrap(op)->DeviceName().c_str();
}

void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* input, TF_Status* status) {
  status->status = tensorflow::unwrap(op)->AddInput(tensorflow::unwrap(input));
}

void TFE_OpAddInputList(TFE_Op* op, TFE_TensorHandle** inputs, int num_inputs,
                        TF_Status* status) {
  status->status = tensorflow::unwrap(op)->AddInputList(
      {reinterpret_cast<tensorflow::AbstractTensorHandle**>(
           tensorflow::unwrap(inputs)),
       static_cast<size_t>(num_inputs)});
}

extern int TFE_OpGetFlatInputCount(const TFE_Op* op, TF_Status* status) {
  return tensorflow::unwrap(op)->GetInputs().size();
}

extern TFE_TensorHandle* TFE_OpGetFlatInput(const TFE_Op* op, int index,
                                            TF_Status* status) {
  return tensorflow::wrap(tensorflow::unwrap(op)->GetInputs()[index]);
}

TF_AttrType TFE_OpGetAttrType(TFE_Op* op, const char* attr_name,
                              unsigned char* is_list, TF_Status* status) {
  TF_AttrType ret = TF_ATTR_INT;
  const tensorflow::AttrTypeMap* attr_types_;
  bool is_function;
  status->status = tensorflow::AttrTypeMapForOp(
      tensorflow::unwrap(op)->Name().c_str(), &attr_types_, &is_function);
  if (!status->status.ok()) {
    return ret;
  }
  status->status =
      tensorflow::AttrTypeByName(*attr_types_, attr_name, &ret, is_list);
  return ret;
}

TF_AttrType TFE_OpNameGetAttrType(TFE_Context* ctx,
                                  const char* op_or_function_name,
                                  const char* attr_name, unsigned char* is_list,
                                  TF_Status* status) {
  TF_AttrType ret;
  TFE_Op* op = TFE_NewOp(ctx, op_or_function_name, status);
  if (status->status.ok()) {
    ret = TFE_OpGetAttrType(op, attr_name, is_list, status);
  } else {
    ret = TF_ATTR_INT;  // Same dummy return as TFE_OpGetAttrType.
  }
  TFE_DeleteOp(op);
  return ret;
}

void TFE_OpSetAttrString(TFE_Op* op, const char* attr_name, const void* value,
                         size_t length) {
  auto s = tensorflow::unwrap(op)->SetAttrString(
      attr_name, static_cast<const char*>(value), length);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name, int64_t value) {
  auto s = tensorflow::unwrap(op)->SetAttrInt(attr_name, value);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name, float value) {
  auto s = tensorflow::unwrap(op)->SetAttrFloat(attr_name, value);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name, unsigned char value) {
  auto s = tensorflow::unwrap(op)->SetAttrBool(attr_name,
                                               (value == 0) ? false : true);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name, TF_DataType value) {
  auto s = tensorflow::unwrap(op)->SetAttrType(
      attr_name, static_cast<tensorflow::DataType>(value));
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name, const int64_t* dims,
                        const int num_dims, TF_Status* out_status) {
  out_status->status =
      tensorflow::unwrap(op)->SetAttrShape(attr_name, dims, num_dims);
}

void TFE_OpSetAttrFunction(TFE_Op* op, const char* attr_name,
                           const TFE_Op* value) {
  auto s = tensorflow::unwrap(op)->SetAttrFunction(
      attr_name, tensorflow::unwrap(const_cast<TFE_Op*>(value)));
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFunctionName(TFE_Op* op, const char* attr_name,
                               const char* data, size_t length) {
  auto s = tensorflow::unwrap(op)->SetAttrFunctionName(attr_name, data, length);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrTensor(TFE_Op* op, const char* attr_name, TF_Tensor* tensor,
                         TF_Status* status) {
  tensorflow::Tensor t;
  status->status = TF_TensorToTensor(tensor, &t);
  tensorflow::TensorInterface interface(t);
  status->status = tensorflow::unwrap(op)->SetAttrTensor(attr_name, &interface);
}

void TFE_OpSetAttrStringList(TFE_Op* op, const char* attr_name,
                             const void* const* values, const size_t* lengths,
                             int num_values) {
  auto s = tensorflow::unwrap(op)->SetAttrStringList(attr_name, values, lengths,
                                                     num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFloatList(TFE_Op* op, const char* attr_name,
                            const float* values, int num_values) {
  auto s =
      tensorflow::unwrap(op)->SetAttrFloatList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrIntList(TFE_Op* op, const char* attr_name,
                          const int64_t* values, int num_values) {
  auto s =
      tensorflow::unwrap(op)->SetAttrIntList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrTypeList(TFE_Op* op, const char* attr_name,
                           const TF_DataType* values, int num_values) {
  auto s = tensorflow::unwrap(op)->SetAttrTypeList(
      attr_name, reinterpret_cast<const tensorflow::DataType*>(values),
      num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrBoolList(TFE_Op* op, const char* attr_name,
                           const unsigned char* values, int num_values) {
  auto s =
      tensorflow::unwrap(op)->SetAttrBoolList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrShapeList(TFE_Op* op, const char* attr_name,
                            const int64_t** dims, const int* num_dims,
                            int num_values, TF_Status* out_status) {
  out_status->status = tensorflow::unwrap(op)->SetAttrShapeList(
      attr_name, dims, num_dims, num_values);
}

void TFE_OpSetAttrFunctionList(TFE_Op* op, const char* attr_name,
                               const TFE_Op** value, int num_values) {
  auto s = tensorflow::unwrap(op)->SetAttrFunctionList(
      attr_name, {reinterpret_cast<const tensorflow::AbstractOperation**>(
                      tensorflow::unwrap(value)),
                  static_cast<size_t>(num_values)});
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrValueProto(const TFE_Op* op, const char* attr_name,
                             const void* proto, size_t proto_len,
                             TF_Status* status) {
  tensorflow::AttrValue attr_value;
  if (!attr_value.ParseFromArray(proto, proto_len)) {
    status->status =
        tensorflow::errors::InvalidArgument("Unparseable AttrValue proto");
    return;
  }
  if (op == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "Got a null or uninitialized `op` argument");
    return;
  }
  tensorflow::EagerOperation* operation =
      OperationFromInterface(tensorflow::unwrap(const_cast<TFE_Op*>(op)));
  operation->MutableAttrs()->Set(attr_name, attr_value);
}

TF_CAPI_EXPORT extern int TFE_OpGetInputLength(TFE_Op* op,
                                               const char* input_name,
                                               TF_Status* status) {
  int ret = -1;
  status->status = tensorflow::unwrap(op)->InputLength(input_name, &ret);
  return ret;
}

TF_CAPI_EXPORT extern int TFE_OpGetOutputLength(TFE_Op* op,
                                                const char* output_name,
                                                TF_Status* status) {
  int ret = -1;
  status->status = tensorflow::unwrap(op)->OutputLength(output_name, &ret);
  return ret;
}

void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
                 TF_Status* status) {
  tensorflow::ImmediateExecutionOperation* unwrapped_op =
      tensorflow::unwrap(op);

  status->status =
      unwrapped_op->GetContext()->GetCustomDeviceOpHandler().Execute(
          unwrapped_op,
          reinterpret_cast<tensorflow::ImmediateExecutionTensorHandle**>(
              retvals),
          num_retvals);
}

TFE_TensorHandle* TFE_TensorHandleCopyToDevice(TFE_TensorHandle* h,
                                               TFE_Context* ctx,
                                               const char* device_name,
                                               TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  tensorflow::ImmediateExecutionContext* unwrapped_ctx =
      tensorflow::unwrap(ctx);

  auto* result =
      unwrapped_ctx->GetCustomDeviceOpHandler().CopyTensorHandleToDevice(
          unwrapped_ctx, tensorflow::unwrap(h), device_name, &status->status);

  if (status->status.ok()) {
    return tensorflow::wrap(result);
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

  AnnotateEagerRuntimeConstructionContext(function_def);
  status->status = tensorflow::unwrap(ctx)->AddFunctionDef(function_def);
}

void TFE_ContextAddFunction(TFE_Context* ctx, TF_Function* function,
                            TF_Status* status) {
  AnnotateEagerRuntimeConstructionContext(function->fdef);
  status->status = tensorflow::unwrap(ctx)->AddFunctionDefWithStackTraces(
      function->fdef, function->stack_traces);
}

void TFE_ContextRemoveFunction(TFE_Context* ctx, const char* name,
                               TF_Status* status) {
  status->status = tensorflow::unwrap(ctx)->RemoveFunction(name);
}

unsigned char TFE_ContextHasFunction(TFE_Context* ctx, const char* name) {
  return tensorflow::unwrap(ctx)->FindFunctionDef(name) != nullptr;
}

void TFE_ContextEnableRunMetadata(TFE_Context* ctx) {
  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(true);
}

void TFE_ContextDisableRunMetadata(TFE_Context* ctx) {
  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(false);
}

}  // extern "C"

TFE_TensorHandle* TFE_NewTensorHandle(const tensorflow::Tensor& t,
                                      TF_Status* status) {
  return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(t));
}

void TFE_ContextExportRunMetadata(TFE_Context* ctx, TF_Buffer* buf,
                                  TF_Status* status) {
  auto* context = tensorflow::unwrap(ctx);
  status->status = context->AsyncWait();
  if (!status->status.ok()) return;
  auto run_metadata = context->ExportRunMetadata();
  status->status = MessageToBuffer(*run_metadata, buf);
}

namespace {
TFE_Op* GetFunc(TFE_Context* ctx, const tensorflow::NameAttrList& func,
                TF_Status* status) {
  TFE_Op* func_op = TFE_NewOp(ctx, func.name().data(), status);
  for (const auto& attr : func.attr()) {
    if (!status->status.ok()) return nullptr;
    SetOpAttrValueScalar(ctx, func_op, attr.second, attr.first.data(), status);
    if (!status->status.ok()) return nullptr;
  }
  return func_op;
}
}  // namespace

void TFE_ContextStartStep(TFE_Context* ctx) {
  tensorflow::unwrap(ctx)->StartStep();
}

void TFE_ContextEndStep(TFE_Context* ctx) {
  tensorflow::unwrap(ctx)->EndStep();
}

const TFE_OpAttrs* TFE_OpGetAttrs(const TFE_Op* op) {
  return tensorflow::wrap(tensorflow::unwrap(op)->GetOpAttrs());
}

void TFE_OpAddAttrs(TFE_Op* op, const TFE_OpAttrs* attrs) {
  tensorflow::unwrap(op)->AddAttrs(tensorflow::unwrap(attrs));
}

void TFE_OpAttrsSerialize(const TFE_OpAttrs* attrs, TF_Buffer* buf,
                          TF_Status* status) {
  tensorflow::NameAttrList name_and_attrs;
  tensorflow::unwrap(attrs)->GetNameAttrList(&name_and_attrs);
  status->status = MessageToBuffer(name_and_attrs, buf);
}

namespace tensorflow {
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status) {
  switch (default_value.value_case()) {
    case tensorflow::AttrValue::kS: {
      const string& v = default_value.s();
      TFE_OpSetAttrString(op, attr_name, v.data(), v.size());
      break;
    }
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
      if (!status->status.ok()) return;
      // TODO(nareshmodi): TFE_OpSetAttrFunction and TFE_OpSetAttrFunctionList
      // require TFE_Op* and just convert it internally a NameAttrValue, so
      // consider adding an overload to the C API to make this case easier.
      TFE_OpSetAttrFunction(op, attr_name, func_op);
      TFE_DeleteOp(func_op);
    } break;
    case tensorflow::AttrValue::kList: {
      // String
      if (const int s_size = default_value.list().s_size()) {
        absl::InlinedVector<const void*, 4> values_vector;
        values_vector.reserve(s_size);
        absl::InlinedVector<size_t, 4> lengths_vector;
        lengths_vector.reserve(s_size);
        for (int i = 0; i < s_size; ++i) {
          const string& v = default_value.list().s(i);
          values_vector.push_back(v.data());
          lengths_vector.push_back(v.size());
        }
        TFE_OpSetAttrStringList(op, attr_name, values_vector.data(),
                                lengths_vector.data(), s_size);
      }

      // Int
      if (const int i_size = default_value.list().i_size()) {
        absl::InlinedVector<int64_t, 4> i_vector;
        i_vector.reserve(i_size);
        for (int i = 0; i < i_size; ++i) {
          i_vector.push_back(default_value.list().i(i));
        }
        TFE_OpSetAttrIntList(op, attr_name, i_vector.data(), i_size);
      }
      // Float
      if (const int f_size = default_value.list().f_size()) {
        absl::InlinedVector<float, 4> f_vector;
        f_vector.reserve(f_size);
        for (int i = 0; i < f_size; ++i) {
          f_vector.push_back(default_value.list().f(i));
        }
        TFE_OpSetAttrFloatList(op, attr_name, f_vector.data(), f_size);
      }
      // Bool
      if (const int b_size = default_value.list().b_size()) {
        absl::InlinedVector<unsigned char, 4> b_vector;
        b_vector.reserve(b_size);
        for (int i = 0; i < b_size; i++) {
          b_vector.push_back(default_value.list().b(i));
        }
        TFE_OpSetAttrBoolList(op, attr_name, b_vector.data(), b_size);
      }
      // Type
      if (const int type_size = default_value.list().type_size()) {
        absl::InlinedVector<unsigned int, 4> type_vector;
        type_vector.reserve(type_size);
        for (int i = 0; i < type_size; ++i) {
          type_vector.push_back(default_value.list().type(i));
        }
        TFE_OpSetAttrTypeList(
            op, attr_name,
            reinterpret_cast<const TF_DataType*>(type_vector.data()),
            type_size);
      }

      // Rest are not supported.
      if (default_value.list().shape_size() > 0 ||
          default_value.list().func_size() > 0 ||
          default_value.list().tensor_size() > 0) {
        TF_SetStatus(
            status, TF_UNIMPLEMENTED,
            tensorflow::strings::StrCat("Unable to get setfor default value: ",
                                        default_value.DebugString())
                .data());
      }
    } break;
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

namespace {
TFE_TensorHandle* DefaultCustomDevicePack(TFE_Context* context,
                                          TFE_TensorHandle** handles,
                                          int num_handles, TF_Status* status,
                                          void* device_info) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "This custom device does not support packing tensors.");
  return nullptr;
}
}  // namespace

extern "C" {

void TFE_RegisterCustomDevice(TFE_Context* ctx, TFE_CustomDevice device,
                              const char* device_name, void* device_info,
                              TF_Status* status) {
  // Fill in default values for optional functionality.
  if (device.pack == nullptr) {
    device.pack = &DefaultCustomDevicePack;
  }
  auto custom_device = std::make_unique<tensorflow::CustomDeviceAPI>(
      ctx, device, device_info, device_name);
  status->status = tensorflow::unwrap(ctx)->RegisterCustomDevice(
      device_name, std::move(custom_device));
}

}  // extern "C"
