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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {

/* static */
Status CapturedFunction::Create(
    const NameAttrList& func, std::vector<Tensor> captured_inputs,
    std::unique_ptr<CapturedFunction>* out_function) {
  out_function->reset(new CapturedFunction(func, std::move(captured_inputs)));
  return Status::OK();
}

CapturedFunction::~CapturedFunction() {}

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

Status CapturedFunction::MaybeInstantiate(
    IteratorContext* ctx, FunctionLibraryRuntime::Handle* out_handle) {
  mutex_lock l(mu_);
  if (lib_ == nullptr) {
    // The context's runtime will be used for all subsequent calls.
    lib_ = ctx->lib();
    DCHECK(f_handle_ == kInvalidHandle);
    FunctionLibraryRuntime::InstantiateOptions inst_opts;
    inst_opts.overlay_lib = ctx->function_library().get();
    inst_opts.state_handle = std::to_string(random::New64());
    TF_RETURN_IF_ERROR(lib_->Instantiate(func_.name(), AttrSlice(&func_.attr()),
                                         inst_opts, &f_handle_));
    const FunctionBody* fbody = lib_->GetFunctionBody(f_handle_);
    if (fbody == nullptr) {
      return errors::Internal("Failed to instantiate function body.");
    }
    ret_types_ = fbody->ret_types;
  } else {
    // TODO(mrry): Consider moving this under a shared lock, as it is
    // the common case.
    if (ctx->lib() != lib_) {
      return errors::Internal(
          "Captured function was called with a different "
          "FunctionLibraryRuntime*, which is not permitted.");
    }
  }
  *out_handle = f_handle_;
  return Status::OK();
}

Status CapturedFunction::Run(IteratorContext* ctx, std::vector<Tensor>&& args,
                             std::vector<Tensor>* rets) {
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(MaybeInstantiate(ctx, &handle));

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = CapturedFunction::generate_step_id();
  ScopedStepContainer step_container(f_opts.step_id, [ctx](const string& name) {
    ctx->lib()->device()->resource_manager()->Cleanup(name).IgnoreError();
  });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  OwnedArgsCallFrame frame(std::move(args), &captured_inputs_, ret_types_);
  Notification n;
  Status s;
  ctx->lib()->Run(f_opts, handle, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

Status CapturedFunction::RunWithBorrowedArgs(IteratorContext* ctx,
                                             const std::vector<Tensor>& args,
                                             std::vector<Tensor>* rets) {
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(MaybeInstantiate(ctx, &handle));

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = CapturedFunction::generate_step_id();
  ScopedStepContainer step_container(f_opts.step_id, [ctx](const string& name) {
    ctx->lib()->device()->resource_manager()->Cleanup(name).IgnoreError();
  });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  BorrowedArgsCallFrame frame(args, &captured_inputs_, ret_types_);
  Notification n;
  Status s;

  ctx->lib()->Run(f_opts, handle, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

void CapturedFunction::RunAsync(IteratorContext* ctx,
                                std::vector<Tensor>&& args,
                                std::vector<Tensor>* rets,
                                FunctionLibraryRuntime::DoneCallback done) {
  // NOTE(mrry): This method does not transfer ownership of `ctx`, and it may
  // be deleted before `done` is called. Take care not to capture `ctx` in any
  // code that may execute asynchronously in this function.
  FunctionLibraryRuntime::Handle handle;
  Status s = MaybeInstantiate(ctx, &handle);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto frame =
      new OwnedArgsCallFrame(std::move(args), &captured_inputs_, ret_types_);

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = CapturedFunction::generate_step_id();
  ResourceMgr* resource_mgr = ctx->lib()->device()->resource_manager();
  auto step_container = new ScopedStepContainer(
      f_opts.step_id, [resource_mgr](const string& name) {
        resource_mgr->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = step_container;
  f_opts.runner = ctx->runner();
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  auto c_mgr = new CancellationManager;
  f_opts.cancellation_manager = c_mgr;

  tf_shared_lock l(mu_);
  ctx->lib()->Run(f_opts, handle, frame,
                  std::bind(
                      [rets, step_container, c_mgr, frame](
                          FunctionLibraryRuntime::DoneCallback done,
                          // Begin unbound arguments.
                          Status s) {
                        delete step_container;
                        delete c_mgr;
                        if (s.ok()) {
                          s = frame->ConsumeRetvals(rets);
                        }
                        delete frame;
                        done(s);
                      },
                      std::move(done), std::placeholders::_1));
}

CapturedFunction::CapturedFunction(const NameAttrList& func,
                                   std::vector<Tensor> captured_inputs)
    : func_(func),
      lib_(nullptr),
      f_handle_(kInvalidHandle),
      captured_inputs_(std::move(captured_inputs)) {}

}  // namespace tensorflow
