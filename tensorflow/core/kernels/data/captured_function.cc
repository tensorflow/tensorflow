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
#include "tensorflow/core/kernels/data/captured_function.h"

#include <utility>

#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/framework/resource_handle.pb_text.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/public/session_options.h"


namespace tensorflow {

/* static */
Status CapturedFunction::Create(
    OpKernelContext* ctx, const NameAttrList& func, int graph_def_version,
    std::vector<Tensor> captured_inputs,
    std::unique_ptr<CapturedFunction>* out_function) {
  // NOTE(mrry): We need to assign a name to the device, and we choose
  // the same name as the calling context's device so that we do not
  // need to rewrite resource handles that are found in `captured_inputs`.
  Device* device =
      new ThreadPoolDevice(SessionOptions(), ctx->device()->attributes().name(),
                           Bytes(256 << 20), DeviceLocality(), cpu_allocator());

// TODO(mrry): Handle arbitrary resource types, which might require a
// redesign (or opening up access to `ResourceMgr::DoLookup()` and
// `ResourceMgr::DoCreate()` to this code).
#define HANDLE_RESOURCE_TYPE(ResourceType)                                     \
  if (input_handle.hash_code() == MakeTypeIndex<ResourceType>().hash_code()) { \
    ResourceType* resource;                                                    \
    Status s = LookupResource(ctx, input_handle, &resource);                   \
    if (errors::IsNotFound(s)) {                                               \
      return errors::FailedPrecondition(                                       \
          "Failed to capture resource named \"", input_handle.name(),          \
          "\" in a dataset function. You may need to initialize it "           \
          "explicitly before initializing an iterator that uses it.");         \
    } else if (!s.ok()) {                                                      \
      return s;                                                                \
    }                                                                          \
    ResourceType* already_created_resource;                                    \
    /* Look up the resource in the this function's resource manager, in case   \
     * it has already been created. */                                         \
    s = device->resource_manager()->Lookup(input_handle.container(),           \
                                           input_handle.name(),                \
                                           &already_created_resource);         \
    if (s.ok()) {                                                              \
      CHECK_EQ(resource, already_created_resource);                            \
      resource->Unref();                                                       \
      already_created_resource->Unref();                                       \
    } else {                                                                   \
      if (errors::IsNotFound(s)) {                                             \
        TF_RETURN_IF_ERROR(device->resource_manager()->Create(                 \
            input_handle.container(), input_handle.name(), resource));         \
      } else {                                                                 \
        return s;                                                              \
      }                                                                        \
    }                                                                          \
    continue;                                                                  \
  }

  for (size_t i = 0; i < captured_inputs.size(); ++i) {
    if (captured_inputs[i].dtype() == DT_RESOURCE) {
      // Extract the resource from `ctx->resource_manager()` and
      // insert it into `device->resource_manager()` so that it can be
      // used when the function executes.
      ResourceHandle input_handle =
          captured_inputs[i].scalar<ResourceHandle>()();
      HANDLE_RESOURCE_TYPE(lookup::LookupInterface);
      HANDLE_RESOURCE_TYPE(QueueInterface);
      HANDLE_RESOURCE_TYPE(Var);
      return errors::Unimplemented(
          "Cannot currently capture resource '",
          ProtoDebugString(input_handle),
          "' in a dataset function (type not supported).");
    }
  }
#undef HANDLE_RESOURCE_TYPE

  std::unique_ptr<DeviceMgr> device_mgr(new DeviceMgr({device}));
  std::unique_ptr<FunctionLibraryDefinition> flib_def(
      new FunctionLibraryDefinition(
          *ctx->function_library()->GetFunctionLibraryDefinition()));
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(device_mgr.get(), ctx->env(),
                                        graph_def_version, flib_def.get(),
                                        {} /* TODO(mrry): OptimizerOptions? */,
                                        nullptr /* TODO(mrry): ClusterFLR */));

  FunctionLibraryRuntime* lib = pflr->GetFLR(device->name());

  FunctionLibraryRuntime::Handle f_handle;
  TF_RETURN_IF_ERROR(
      lib->Instantiate(func.name(), AttrSlice(&func.attr()), &f_handle));
  const FunctionBody* fbody = lib->GetFunctionBody(f_handle);
  if (fbody == nullptr) {
    return errors::Internal("Failed to instantiate function body.");
  }

  out_function->reset(new CapturedFunction(
      device, std::move(device_mgr), std::move(flib_def), std::move(pflr), lib,
      f_handle, std::move(captured_inputs), fbody->ret_types));
  return Status::OK();
}

namespace {
class CallFrameBase : public CallFrameInterface {
 public:
  explicit CallFrameBase(DataTypeSlice ret_types)
      : ret_types_(ret_types), retvals_(ret_types.size()) {}

  // Caller methods.
  Status ConsumeRetvals(std::vector<Tensor>* retvals) {
    retvals->reserve(retvals_.size());
    int i = 0;
    for (auto&& val : retvals_) {
      if (!val) {
        return errors::Internal("No return value for index ", i, ".");
      }
      retvals->emplace_back(std::move(val.value()));
      ++i;
    }
    return Status::OK();
  }

  size_t num_retvals() const override { return retvals_.size(); }

  // Callee methods.
  Status SetRetval(int index, const Tensor& val) override {
    if (index < retvals_.size() && val.dtype() == ret_types_[index] &&
        !retvals_[index]) {
      retvals_[index] = val;
      return Status::OK();
    } else if (index >= retvals_.size()) {
      return errors::InvalidArgument("Return value ", index,
                                     " is out of range.");
    } else if (val.dtype() != ret_types_[index]) {
      return errors::InvalidArgument("Expected type ",
                                     DataTypeString(ret_types_[index]),
                                     " for return value ", index, " but got ",
                                     DataTypeString(val.dtype()), ".");
    } else {
      return errors::Internal("Attempted to set return value ", index,
                              " more than once.");
    }
  }

 private:
  DataTypeSlice ret_types_;
  std::vector<gtl::optional<Tensor>> retvals_;
  TF_DISALLOW_COPY_AND_ASSIGN(CallFrameBase);
};

class OwnedArgsCallFrame : public CallFrameBase {
 public:
  OwnedArgsCallFrame(std::vector<Tensor>&& args,
                     const std::vector<Tensor>* captured_inputs,
                     DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(std::move(args)),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override {
    if (index < args_.size() && args_[index].IsInitialized()) {
      // TODO(mrry): Consider making `CallFrameInterface::GetArg` non-const in
      // order to be able to `std::move(args_[index])` into `*val`.
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else if (index >= args_.size() + captured_inputs_->size()) {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    } else {
      return errors::Internal("Attempted to get argument ", index,
                              " more than once.");
    }
  }

 private:
  std::vector<Tensor> args_;
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

class BorrowedArgsCallFrame : public CallFrameBase {
 public:
  BorrowedArgsCallFrame(const std::vector<Tensor>& args,
                        const std::vector<Tensor>* captured_inputs,
                        DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(args),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override {
    if (index < args_.size() && args_[index].IsInitialized()) {
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else if (index >= args_.size() + captured_inputs_->size()) {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    } else {
      return errors::Internal("Attempted to get argument ", index,
                              " more than once.");
    }
  }

 private:
  const std::vector<Tensor>& args_;                   // Not owned.
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

}  // namespace

Status CapturedFunction::Run(FunctionLibraryRuntime::Options f_opts,
                             std::vector<Tensor>&& args,
                             std::vector<Tensor>* rets) {
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  auto c_mgr = new CancellationManager;
  auto frame =
      new OwnedArgsCallFrame(std::move(args), &captured_inputs_, ret_types_);
  f_opts.cancellation_manager = c_mgr;
  Notification n;
  Status s;
  lib_->Run(f_opts, f_handle_, frame,
            [rets, c_mgr, frame, &n, &s](Status func_status) {
              delete c_mgr;
              s.Update(func_status);
              if (s.ok()) {
                s = frame->ConsumeRetvals(rets);
              }
              delete frame;
              n.Notify();
            });
  n.WaitForNotification();
  return s;
}

Status CapturedFunction::RunWithBorrowedArgs(
    FunctionLibraryRuntime::Options f_opts, const std::vector<Tensor>& args,
    std::vector<Tensor>* rets) {
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  auto c_mgr = new CancellationManager;
  BorrowedArgsCallFrame frame(args, &captured_inputs_, ret_types_);
  f_opts.cancellation_manager = c_mgr;
  Notification n;
  Status s;
  lib_->Run(f_opts, f_handle_, &frame,
            [rets, c_mgr, &frame, &n, &s](Status func_status) {
              delete c_mgr;
              s.Update(func_status);
              if (s.ok()) {
                s = frame.ConsumeRetvals(rets);
              }
              n.Notify();
            });
  n.WaitForNotification();
  return s;
}

void CapturedFunction::RunAsync(FunctionLibraryRuntime::Options f_opts,
                                std::vector<Tensor>&& args,
                                std::vector<Tensor>* rets,
                                FunctionLibraryRuntime::DoneCallback done) {
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  auto c_mgr = new CancellationManager;
  auto frame =
      new OwnedArgsCallFrame(std::move(args), &captured_inputs_, ret_types_);
  f_opts.cancellation_manager = c_mgr;
  lib_->Run(f_opts, f_handle_, frame,
            std::bind(
                [rets, c_mgr, frame](FunctionLibraryRuntime::DoneCallback done,
                                     // Begin unbound arguments.
                                     Status s) {
                  delete c_mgr;
                  if (s.ok()) {
                    s = frame->ConsumeRetvals(rets);
                  }
                  delete frame;
                  done(s);
                },
                std::move(done), std::placeholders::_1));
}

CapturedFunction::CapturedFunction(
    Device* device, std::unique_ptr<DeviceMgr> device_mgr,
    std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, FunctionLibraryRuntime::Handle f_handle,
    std::vector<Tensor> captured_inputs, DataTypeSlice ret_types)
    : device_(device),
      device_mgr_(std::move(device_mgr)),
      flib_def_(std::move(flib_def)),
      pflr_(std::move(pflr)),
      lib_(lib),
      f_handle_(f_handle),
      captured_inputs_(std::move(captured_inputs)),
      ret_types_(ret_types) {}

}  // namespace tensorflow
