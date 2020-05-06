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
#include "tensorflow/core/framework/types.h"
#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#endif
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef FunctionLibraryRuntime::Handle FHandle;
typedef std::vector<Tensor> TensorVec;

namespace {

// Helper to instantiate function "func" in the library "lib".
Status Instantiate(FunctionLibraryRuntime* lib, const NameAttrList& func,
                   FunctionLibraryRuntime::Handle* handle) {
  return lib->Instantiate(func.name(), AttrSlice(&func.attr()), handle);
}

Status Instantiate(OpKernelContext* ctx, const NameAttrList& func,
                   FunctionLibraryRuntime::Handle* handle) {
  FunctionLibraryRuntime::InstantiateOptions opts;
  opts.executor_type = ctx->executor_type();
  return ctx->function_library()->Instantiate(
      func.name(), AttrSlice(&func.attr()), opts, handle);
}

// If "t" is a scalar of a supported type, returns t != 0 in "*v".
Status ToBool(gtl::ArraySlice<Tensor> t, bool* v) {
  if (t.size() != 1) {
    return errors::InvalidArgument(
        "Expected a single scalar which can be converted to a boolean, got ",
        t.size(), " tensors.");
  }
  if (TensorShapeUtils::IsScalar(t[0].shape())) {
    switch (t[0].dtype()) {
#define CASE(T)                   \
  case DataTypeToEnum<T>::value:  \
    *v = t[0].scalar<T>()() != 0; \
    break;

      CASE(float);
      CASE(double);
      CASE(int32);
      CASE(uint8);
      CASE(int16);
      CASE(int8);
      CASE(int64);
#undef CASE
      case DT_BOOL:
        *v = t[0].scalar<bool>()();
        break;
      case DT_STRING:
        *v = !t[0].scalar<tstring>()().empty();
        break;
      default:
        return errors::InvalidArgument(DataTypeString(t[0].dtype()),
                                       " cannot be converted to a boolean");
    }
  } else {
    *v = t[0].NumElements() > 0;
  }
  return Status::OK();
}

// Sets "rets" to be the output of "ctx". Validates rets' types based
// on "kernel".
Status SetOutputs(const OpKernel* kernel, OpKernelContext* ctx,
                  gtl::ArraySlice<Tensor> rets) {
  if (rets.size() != ctx->num_outputs()) {
    return errors::Internal("Expect to produce ", ctx->num_outputs(),
                            " tensors, but only get ", rets.size());
  }
  for (int i = 0; i < rets.size(); ++i) {
    if (rets[i].dtype() != kernel->output_type(i)) {
      return errors::Internal("Expect ", i, "-th output is of type ",
                              DataTypeString(kernel->output_type(i)),
                              " but get ", DataTypeString(rets[i].dtype()));
    }
    ctx->set_output(i, rets[i]);
  }
  return Status::OK();
}

void SetRunOptions(OpKernelContext* ctx, FunctionLibraryRuntime::Options* opts,
                   bool always_collect_stats) {
  opts->rendezvous = ctx->rendezvous();
  opts->cancellation_manager = ctx->cancellation_manager();
  if (always_collect_stats) {
    opts->stats_collector = ctx->stats_collector();
  }
  opts->runner = ctx->runner();
  opts->run_all_kernels_inline = ctx->run_all_kernels_inline();
  opts->step_container = ctx->step_container();
}

class IfOp : public AsyncOpKernel {
 public:
  explicit IfOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    auto lib = ctx->function_library();
    OP_REQUIRES(ctx, lib != nullptr, errors::Internal("No function library"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("then_branch", &then_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("else_branch", &else_func_));
  }

  ~IfOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FHandle then_handle;
    FHandle else_handle;
    OP_REQUIRES_OK_ASYNC(ctx, GetHandles(ctx, &then_handle, &else_handle),
                         done);
    bool cond;
    OP_REQUIRES_OK(ctx, ToBool({ctx->input(0)}, &cond));
    (new State(this, ctx, cond, then_handle, else_handle, done))->Start();
  }

 private:
  NameAttrList then_func_;
  NameAttrList else_func_;

  mutex mu_;
  std::unordered_map<FunctionLibraryRuntime*, std::pair<FHandle, FHandle>>
      handles_ GUARDED_BY(mu_);

  class State {
   public:
    State(IfOp* kernel, OpKernelContext* ctx, bool cond, FHandle then_handle,
          FHandle else_handle, DoneCallback done)
        : kernel_(kernel),
          ctx_(ctx),
          cond_(cond),
          then_handle_(then_handle),
          else_handle_(else_handle),
          done_(std::move(done)),
          lib_(CHECK_NOTNULL(ctx_->function_library())) {
      SetRunOptions(ctx_, &opts_, true /* always_collect_stats */);
      for (int i = 1; i < ctx_->num_inputs(); ++i) {
        args_.push_back(ctx_->input(i));
      }
    }

    ~State() {}

    void Start() {
      FHandle handle = cond_ ? then_handle_ : else_handle_;
      rets_.clear();
      profiler::TraceMe trace_me(
          [&] {
            return absl::StrCat("IfOp #parent_step_id=", ctx_->step_id(),
                                ",function_step_id=", opts_.step_id, "#");
          },
          /*level=*/2);
      lib_->Run(
          // Evaluate one of the branch.
          opts_, handle, args_, &rets_,
          // Done callback
          [this](Status s) {
            if (s.ok()) {
              s = SetOutputs(kernel_, ctx_, rets_);
            }
            ctx_->SetStatus(s);
            DoneCallback captured_done(std::move(done_));
            delete this;
            captured_done();
          });
    }

   private:
    IfOp* const kernel_;
    OpKernelContext* const ctx_;
    const bool cond_;
    FHandle then_handle_;
    FHandle else_handle_;
    DoneCallback done_;
    FunctionLibraryRuntime* const lib_;
    FunctionLibraryRuntime::Options opts_;
    TensorVec args_;
    TensorVec rets_;
  };

  Status GetHandles(OpKernelContext* ctx, FHandle* then_handle,
                    FHandle* else_handle) {
    // TODO(b/37549631): Because this op has `SetIsStateful()` in its
    // op registration, this kernel may be shared by multiple
    // subgraphs, which have different associated
    // `FunctionLibraryRuntime` objects and hence different `FHandle`
    // namespaces. We currently work around this by caching the map
    // from `FunctionLibraryRuntime*` to `FHandle` pairs for the two
    // functions this op uses.
    auto lib = ctx->function_library();
    if (lib == nullptr) return errors::Internal("No function library");
    *then_handle = kInvalidHandle;
    *else_handle = kInvalidHandle;
    {
      tf_shared_lock l(mu_);
      const auto iter = handles_.find(lib);
      if (TF_PREDICT_TRUE(iter != handles_.end())) {
        *then_handle = iter->second.first;
        *else_handle = iter->second.second;
      }
    }
    if (TF_PREDICT_FALSE(*then_handle == kInvalidHandle)) {
      mutex_lock l(mu_);
      const auto iter = handles_.find(lib);
      if (TF_PREDICT_TRUE(iter != handles_.end())) {
        *then_handle = iter->second.first;
        *else_handle = iter->second.second;
      } else {
        TF_RETURN_IF_ERROR(Instantiate(ctx, then_func_, then_handle));
        TF_RETURN_IF_ERROR(Instantiate(ctx, else_func_, else_handle));
        handles_[lib] = {*then_handle, *else_handle};
      }
    }
    return Status::OK();
  }
};

class CaseOp : public AsyncOpKernel {
 public:
  explicit CaseOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    auto lib = ctx->function_library();
    OP_REQUIRES(ctx, lib != nullptr, errors::Internal("No function library"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("branches", &branch_funcs_));
  }

  ~CaseOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library"), done);

    // TODO(b/37549631): Because this op has `SetIsStateful()` in its op
    // registration, this kernel may be shared by multiple subgraphs, which have
    // different associated `FunctionLibraryRuntime` objects and hence different
    // `FHandle` namespaces. So we must call Instantiate() to make sure we get
    // the correct function handles with respect to `lib`. Note the underlying
    // `lib->Instantiate()` caches the created function handles, so calling
    // `Instantiate()` repeatedly on the same `lib` and function is cheap.
    std::vector<FHandle> branch_handles(branch_funcs_.size());
    for (int i = 0; i < branch_funcs_.size(); i++) {
      OP_REQUIRES_OK_ASYNC(
          ctx, Instantiate(lib, branch_funcs_[i], &branch_handles[i]), done);
    }

    const Tensor& branch_index = ctx->input(0);
    OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsScalar(branch_index.shape()),
                      errors::InvalidArgument("branch_index must be scalar"),
                      done);
    int32 branch = branch_index.scalar<int32>()();
    (new State(this, ctx, branch, branch_handles, done))->Start();
  }

 private:
  std::vector<NameAttrList> branch_funcs_;

  class State {
   public:
    State(CaseOp* kernel, OpKernelContext* ctx, int branch,
          std::vector<FHandle> branch_handles, DoneCallback done)
        : kernel_(kernel),
          ctx_(ctx),
          branch_(branch),
          branch_handles_(branch_handles),
          done_(std::move(done)),
          lib_(CHECK_NOTNULL(ctx_->function_library())) {
      SetRunOptions(ctx_, &opts_, true /* always_collect_stats */);
      for (int i = 1; i < ctx_->num_inputs(); ++i) {
        args_.push_back(ctx_->input(i));
      }
    }

    ~State() {}

    void Start() {
      int branch = branch_;
      // The last branch is the default branch.
      if (branch < 0 || branch >= branch_handles_.size()) {
        branch = branch_handles_.size() - 1;
      }
      rets_.clear();
      profiler::TraceMe trace_me(
          [&] {
            return absl::StrCat("CaseOp #parent_step_id=", ctx_->step_id(),
                                ",function_step_id=", opts_.step_id, "#");
          },
          /*level=*/2);
      lib_->Run(
          // Evaluate one of the branch.
          opts_, branch_handles_[branch], args_, &rets_,
          // Done callback
          [this](Status s) {
            if (s.ok()) {
              s = SetOutputs(kernel_, ctx_, rets_);
            }
            ctx_->SetStatus(s);
            DoneCallback captured_done(std::move(done_));
            delete this;
            captured_done();
          });
    }

   private:
    CaseOp* const kernel_;
    OpKernelContext* const ctx_;
    const int branch_;
    std::vector<FHandle> branch_handles_;
    DoneCallback done_;
    FunctionLibraryRuntime* const lib_;
    FunctionLibraryRuntime::Options opts_;
    TensorVec args_;
    TensorVec rets_;
  };
};

// TODO(drpng): remove this.
REGISTER_KERNEL_BUILDER(Name("_If").Device(DEVICE_CPU), IfOp);
REGISTER_KERNEL_BUILDER(Name("_If").Device(DEVICE_GPU).HostMemory("cond"),
                        IfOp);

REGISTER_KERNEL_BUILDER(Name("If").Device(DEVICE_CPU), IfOp);
REGISTER_KERNEL_BUILDER(Name("If").Device(DEVICE_GPU).HostMemory("cond"), IfOp);

REGISTER_KERNEL_BUILDER(Name("Case").Device(DEVICE_CPU), CaseOp);
REGISTER_KERNEL_BUILDER(
    Name("Case").Device(DEVICE_GPU).HostMemory("branch_index"), CaseOp);

REGISTER_KERNEL_BUILDER(Name("StatelessIf").Device(DEVICE_CPU), IfOp);
REGISTER_KERNEL_BUILDER(
    Name("StatelessIf").Device(DEVICE_GPU).HostMemory("cond"), IfOp);

class WhileOp : public AsyncOpKernel {
 public:
  explicit WhileOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cond", &cond_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("body", &body_func_));
  }

  ~WhileOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    if (ctx->run_all_kernels_inline()) {
      // Use the non-callback-based implementation when kernels (and function
      // callbacks) execute inline to avoid stack overflow.
      OP_REQUIRES_OK_ASYNC(ctx, DoComputeSync(ctx), done);
    } else {
      FHandle cond_handle;
      FHandle body_handle;
      OP_REQUIRES_OK_ASYNC(ctx, GetHandles(ctx, &cond_handle, &body_handle),
                           done);
      (new State(this, ctx, cond_handle, body_handle, done))->Start();
    }
  }

  void Compute(OpKernelContext* ctx) override {
    // Use the non-callback-based implementation when the synchronous Compute()
    // method is invoked, because the caller is explicitly donating a thread.
    Status s = DoComputeSync(ctx);
    // NOTE: Unfortunately, we cannot use OP_REQUIRES_OK here, because this is
    // still an AsyncOpKernel, and there is a run-time check to avoid calling
    // OP_REQUIRES_OK in AsyncOpKernel::ComputeAsync() (which would deadlock in
    // the event of an error).
    if (TF_PREDICT_FALSE(!s.ok())) {
      ctx->SetStatus(s);
    }
  }

 private:
  NameAttrList cond_func_;
  NameAttrList body_func_;

  mutex mu_;
  std::unordered_map<FunctionLibraryRuntime*, std::pair<FHandle, FHandle>>
      handles_ GUARDED_BY(mu_);

  static string EvalCondTraceString(
      OpKernelContext* ctx, const FunctionLibraryRuntime::Options& opts) {
    return absl::StrCat("WhileOp-EvalCond #parent_step_id=", ctx->step_id(),
                        ",function_step_id=", opts.step_id, "#");
  }

  static string StartBodyTraceString(
      OpKernelContext* ctx, const FunctionLibraryRuntime::Options& opts) {
    return absl::StrCat("WhileOp-StartBody #parent_step_id=", ctx->step_id(),
                        ",function_step_id=", opts.step_id, "#");
  }

  static Status CondResultToBool(OpKernelContext* ctx,
                                 const FunctionLibraryRuntime::Options& opts,
                                 const Tensor& cond_t, bool* out_result) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    const DeviceBase::GpuDeviceInfo* gpu_device_info =
        ctx->device()->tensorflow_gpu_device_info();
    const bool is_hostmem_dtype =
        cond_t.dtype() == DT_INT32 || cond_t.dtype() == DT_INT64;
    if (!is_hostmem_dtype && gpu_device_info &&
        (opts.rets_alloc_attrs.empty() ||
         !opts.rets_alloc_attrs[0].on_host())) {
      // Copy the ret value to host if it's allocated on device.
      Device* device = down_cast<Device*>(ctx->device());
      DeviceContext* device_ctx = ctx->op_device_context();
      Tensor host_cond_t = Tensor(cond_t.dtype(), cond_t.shape());
      TF_RETURN_IF_ERROR(device_ctx->CopyDeviceTensorToCPUSync(
          &cond_t, /*tensor_name=*/"", device, &host_cond_t));
      return ToBool({host_cond_t}, out_result);
    }
#endif
    return ToBool({cond_t}, out_result);
  }

  // The initial loop variable args are the inputs to the kernel.
  //
  // We attempt to forward the input so that it can be consumed inside the
  // body function (and participate in buffer forwarding, etc.).
  static void GetArgsFromContext(OpKernelContext* ctx,
                                 std::vector<Tensor>* out_args,
                                 DataTypeVector* out_var_types) {
    const int num_loop_vars = ctx->num_inputs();
    out_args->reserve(num_loop_vars);
    out_var_types->resize(num_loop_vars);
    for (int i = 0; i < num_loop_vars; ++i) {
      const Tensor& input = ctx->input(i);
      (*out_var_types)[i] = input.dtype();
      std::unique_ptr<Tensor> maybe_forwarded_input = ctx->forward_input(
          i, /* output_index= */ OpKernelContext::Params::kNoReservation,
          input.dtype(), input.shape(), ctx->input_memory_type(i),
          ctx->input_alloc_attr(i));
      if (maybe_forwarded_input) {
        out_args->push_back(std::move(*maybe_forwarded_input));
      } else {
        out_args->push_back(input);
      }
    }
  }

  class BodyFuncCallFrame : public CallFrameInterface {
   public:
    BodyFuncCallFrame(std::vector<Tensor>* args, std::vector<Tensor>* retvals,
                      DataTypeSlice ret_types)
        : args_(args), retvals_(retvals), ret_types_(ret_types) {}

    size_t num_args() const override { return args_->size(); }
    size_t num_retvals() const override { return retvals_->size(); }

    Status GetArg(int index, const Tensor** val) override {
      if (index < args_->size()) {
        *val = &(*args_)[index];
        return Status::OK();
      } else {
        return errors::InvalidArgument("Argument ", index, " is out of range.");
      }
    }

    void ConsumeArg(int index, Tensor* val) override {
      DCHECK_GE(index, 0);
      DCHECK_LT(index, args_->size());
      *val = std::move((*args_)[index]);
    }
    bool CanConsumeArg(int index) const override {
      return index >= 0 && index < args_->size();
    }

    Status SetRetval(int index, const Tensor& val) override {
      if (TF_PREDICT_FALSE(index < 0)) {
        return errors::InvalidArgument(
            "Expected non-negative return value index, but got: ", index, ".");
      } else if (TF_PREDICT_FALSE(index >= retvals_->size())) {
        return errors::InvalidArgument("While loop body returned ", index + 1,
                                       " arguments. Expected: ", num_retvals(),
                                       ".");
      } else if (TF_PREDICT_FALSE(val.dtype() != ret_types_[index])) {
        return errors::InvalidArgument("Expected type ",
                                       DataTypeString(ret_types_[index]),
                                       " for return value ", index, " but got ",
                                       DataTypeString(val.dtype()), ".");
      }
      (*retvals_)[index] = val;
      return Status::OK();
    }

   private:
    std::vector<Tensor>* const args_;     // Not owned.
    std::vector<Tensor>* const retvals_;  // Not owned.
    DataTypeSlice ret_types_;

    TF_DISALLOW_COPY_AND_ASSIGN(BodyFuncCallFrame);
  };

  class State {
   public:
    State(WhileOp* kernel, OpKernelContext* ctx, FHandle cond_handle,
          FHandle body_handle, DoneCallback done)
        : kernel_(kernel),
          ctx_(ctx),
          cond_handle_(cond_handle),
          body_handle_(body_handle),
          done_(std::move(done)),
          lib_(CHECK_NOTNULL(ctx_->function_library())) {
      SetRunOptions(ctx_, &opts_, false /* always_collect_stats */);
      GetArgsFromContext(ctx, &args_, &loop_var_types_);
      body_frame_ =
          absl::make_unique<BodyFuncCallFrame>(&args_, &rets_, loop_var_types_);
    }

    ~State() {}

    void Start() { EvalCond(); }

   private:
    WhileOp* const kernel_;
    OpKernelContext* const ctx_;
    const FHandle cond_handle_;
    const FHandle body_handle_;
    const DoneCallback done_;
    FunctionLibraryRuntime* const lib_;
    FunctionLibraryRuntime::Options opts_;
    TensorVec args_;
    TensorVec rets_;
    DataTypeVector loop_var_types_;
    std::unique_ptr<BodyFuncCallFrame> body_frame_;

    void EvalCond() {
      profiler::TraceMe trace_me(
          [&] { return EvalCondTraceString(ctx_, opts_); },
          /*level=*/2);
      lib_->Run(
          // Evaluate the condition.
          opts_, cond_handle_, args_, &rets_,
          // Done cb.
          [this](const Status& s) {
            if (!s.ok()) {
              return Finish(s);
            }
            StartBody();
          });
    }

    void StartBody() {
      Status s;
      if (rets_.size() != 1) {
        s = errors::InvalidArgument(
            "Expected a single scalar return value from WhileOp cond, got ",
            rets_.size(), " tensors.");
        return Finish(s);
      }

      if (!s.ok()) {
        return Finish(s);
      }
      bool cond;
      s = CondResultToBool(ctx_, opts_, rets_[0], &cond);
      if (!s.ok()) {
        return Finish(s);
      }

      if (!cond) {
        return Finish(Status::OK());
      }
      rets_.clear();
      rets_.resize(args_.size());
      profiler::TraceMe trace_me(
          [&] { return StartBodyTraceString(ctx_, opts_); },
          /*level=*/2);
      lib_->Run(
          // Evaluate the body.
          opts_, body_handle_, body_frame_.get(),
          // Done callback
          [this](const Status& s) {
            if (!s.ok()) {
              return Finish(s);
            }
            if (args_.size() != rets_.size()) {
              return Finish(errors::InvalidArgument(
                  "While loop body returned ", rets_.size(),
                  " arguments. Expected: ", args_.size()));
            }
            args_.clear();
            using std::swap;
            swap(args_, rets_);
            EvalCond();
          });
    }

    void Finish(Status s) {
      if (s.ok()) {
        s = SetOutputs(kernel_, ctx_, args_);
      }
      ctx_->SetStatus(s);
      done_();
      delete this;
    }
  };

  Status DoComputeSync(OpKernelContext* ctx) {
    FHandle cond_handle;
    FHandle body_handle;
    TF_RETURN_IF_ERROR(GetHandles(ctx, &cond_handle, &body_handle));
    auto lib = ctx->function_library();
    FunctionLibraryRuntime::Options opts;
    SetRunOptions(ctx, &opts, false /* always_collect_stats */);

    // Pre-allocate argument and return value vectors for the cond and body
    // functions.
    std::vector<Tensor> args;
    const int num_loop_vars = ctx->num_inputs();
    DataTypeVector loop_var_types(num_loop_vars);
    GetArgsFromContext(ctx, &args, &loop_var_types);
    std::vector<Tensor> cond_rets;
    cond_rets.reserve(1);
    std::vector<Tensor> body_rets;
    body_rets.reserve(num_loop_vars);

    // Implement the logic of the while loop as a single C++ do-while loop that
    // executes the cond and body functions synchronously.
    do {
      // Evaluate the cond function on the current loop variables.
      {
        profiler::TraceMe trace_me(
            [&] { return EvalCondTraceString(ctx, opts); },
            /*level=*/2);
        TF_RETURN_IF_ERROR(lib->RunSync(opts, cond_handle, args, &cond_rets));
      }
      if (cond_rets.size() != 1) {
        return errors::InvalidArgument(
            "Expected a single scalar return value from WhileOp cond, got ",
            cond_rets.size(), " tensors.");
      }

      // If the cond function evaluates to false, we are done: output the
      // current loop variables.
      bool cond_result;
      TF_RETURN_IF_ERROR(
          CondResultToBool(ctx, opts, cond_rets[0], &cond_result));
      if (!cond_result) {
        return SetOutputs(this, ctx, args);
      }

      // Evaluate the body function on the current loop variables, to get an
      // updated vector of loop variables.
      {
        profiler::TraceMe trace_me(
            [&] { return StartBodyTraceString(ctx, opts); },
            /*level=*/2);
        body_rets.resize(num_loop_vars);
        BodyFuncCallFrame call_frame(&args, &body_rets, loop_var_types);
        TF_RETURN_IF_ERROR(lib->RunSync(opts, body_handle, &call_frame));
      }
      std::swap(body_rets, args);
      body_rets.clear();
    } while (true);
  }

  Status GetHandles(OpKernelContext* ctx, FHandle* cond_handle,
                    FHandle* body_handle) {
    // TODO(b/37549631): Because this op has `SetIsStateful()` in its
    // op registration, this kernel may be shared by multiple
    // subgraphs, which have different associated
    // `FunctionLibraryRuntime` objects and hence different `FHandle`
    // namespaces. We currently work around this by caching the map
    // from `FunctionLibraryRuntime*` to `FHandle` pairs for the two
    // functions this op uses.
    auto lib = ctx->function_library();
    if (lib == nullptr) return errors::Internal("No function library");
    *cond_handle = kInvalidHandle;
    *body_handle = kInvalidHandle;
    {
      tf_shared_lock l(mu_);
      const auto iter = handles_.find(lib);
      if (TF_PREDICT_TRUE(iter != handles_.end())) {
        *cond_handle = iter->second.first;
        *body_handle = iter->second.second;
      }
    }
    if (TF_PREDICT_FALSE(*cond_handle == kInvalidHandle)) {
      mutex_lock l(mu_);
      const auto iter = handles_.find(lib);
      if (TF_PREDICT_TRUE(iter != handles_.end())) {
        *cond_handle = iter->second.first;
        *body_handle = iter->second.second;
      } else {
        TF_RETURN_IF_ERROR(Instantiate(ctx, cond_func_, cond_handle));
        TF_RETURN_IF_ERROR(Instantiate(ctx, body_func_, body_handle));
        handles_[lib] = {*cond_handle, *body_handle};
      }
    }
    return Status::OK();
  }
};
// TODO(drpng): remove these.
REGISTER_KERNEL_BUILDER(Name("_While").Device(DEVICE_CPU), WhileOp);
REGISTER_KERNEL_BUILDER(Name("_While").Device(DEVICE_GPU), WhileOp);

REGISTER_KERNEL_BUILDER(Name("While").Device(DEVICE_CPU), WhileOp);
REGISTER_KERNEL_BUILDER(Name("While").Device(DEVICE_GPU), WhileOp);

REGISTER_KERNEL_BUILDER(Name("StatelessWhile").Device(DEVICE_CPU), WhileOp);
REGISTER_KERNEL_BUILDER(Name("StatelessWhile").Device(DEVICE_GPU), WhileOp);

class ToBoolOp : public OpKernel {
 public:
  explicit ToBoolOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    bool b;
    OP_REQUIRES_OK(ctx, ToBool({ctx->input(0)}, &b));
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<bool>()() = b;
  }
};

REGISTER_KERNEL_BUILDER(Name("ToBool").Device(DEVICE_CPU), ToBoolOp);

Status GetScalar(OpKernelContext* ctx, int index, int32* value,
                 const char* label) {
  Tensor t = ctx->input(index);
  if (!TensorShapeUtils::IsScalar(t.shape())) {
    return errors::InvalidArgument(label, " must be a scalar, but ",
                                   t.shape().DebugString());
  }
  *value = t.scalar<int32>()();
  return Status::OK();
}

class ForOp : public AsyncOpKernel {
 public:
  explicit ForOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    auto lib = ctx->function_library();
    OP_REQUIRES(ctx, lib != nullptr, errors::Internal("No function library"));
    const NameAttrList* func;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("body", &func));
    OP_REQUIRES_OK(ctx, Instantiate(lib, *func, &body_handle_));
  }

  ~ForOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    (new State(this, ctx, done))->Start();
  }

 private:
  FHandle body_handle_;

  class State {
   public:
    State(ForOp* kernel, OpKernelContext* ctx, DoneCallback done)
        : kernel_(kernel),
          ctx_(ctx),
          done_(std::move(done)),
          lib_(CHECK_NOTNULL(ctx_->function_library())),
          args_(1 + ctx_->num_inputs() - 3) {
      args_[0] = Tensor(DT_INT32, {});
      iter_ = &args_[0].scalar<int32>()();

      const int32 num_loop_inputs = ctx_->num_inputs() - 3;
      rets_.reserve(num_loop_inputs);
      for (int i = 0; i < num_loop_inputs; ++i) {
        rets_.push_back(ctx_->input(3 + i));
      }
    }

    ~State() {}

    void Start() {
      Status s = StartLoop();
      if (!s.ok()) Finish(s);
    }

   private:
    ForOp* const kernel_;
    OpKernelContext* const ctx_;
    const DoneCallback done_;
    FunctionLibraryRuntime* const lib_;
    FunctionLibraryRuntime::Options opts_;
    TensorVec args_;
    TensorVec rets_;

    int32* iter_;  // points to args_[0].
    int32 limit_;
    int32 delta_;

    // If an error e is returned, caller must call Finish(e).
    // If OK is returned, the async loop execution has been started.
    Status StartLoop() {
      SetRunOptions(ctx_, &opts_, false /* always_collect_stats */);

      TF_RETURN_IF_ERROR(GetScalar(ctx_, 0, iter_, "start"));
      TF_RETURN_IF_ERROR(GetScalar(ctx_, 1, &limit_, "limit"));
      TF_RETURN_IF_ERROR(GetScalar(ctx_, 2, &delta_, "delta"));

      if ((delta_ > 0 && *iter_ <= limit_) ||
          (delta_ < 0 && *iter_ >= limit_) ||
          (delta_ == 0 && *iter_ == limit_)) {
        RunNext();
        return Status::OK();
      } else {
        return errors::InvalidArgument("Invalid start/limit/delta: ", *iter_,
                                       " ", limit_, " ", delta_);
      }
    }

    void RunNext() {
      bool done_loop;
      if (delta_ > 0) {
        done_loop = *iter_ >= limit_;
      } else {
        done_loop = *iter_ <= limit_;
      }
      if (done_loop) {
        Finish(Status::OK());
        return;
      }

      if (rets_.size() >= args_.size()) {
        Finish(errors::InvalidArgument(
            "For loop body returned ", rets_.size(),
            " arguments. Expected: ", args_.size() - 1));
        return;
      }
      for (int i = 0; i < rets_.size(); ++i) {
        args_[1 + i] = std::move(rets_[i]);
      }
      rets_.clear();
      profiler::TraceMe trace_me(
          [&] {
            return absl::StrCat("ForOp #parent_step_id=", ctx_->step_id(),
                                ",function_step_id=", opts_.step_id, "#");
          },
          /*level=*/2);
      lib_->Run(opts_, kernel_->body_handle_, args_, &rets_,
                [this](const Status& s) {
                  if (s.ok()) {
                    *iter_ += delta_;
                    RunNext();
                  } else {
                    Finish(s);
                  }
                });
    }

    void Finish(Status s) {
      if (s.ok()) {
        s = SetOutputs(kernel_, ctx_, rets_);
      }
      ctx_->SetStatus(s);
      done_();
      delete this;
    }
  };
};

REGISTER_KERNEL_BUILDER(Name("For").Device(DEVICE_CPU), ForOp);
REGISTER_KERNEL_BUILDER(Name("For")
                            .Device(DEVICE_GPU)
                            .HostMemory("start")
                            .HostMemory("limit")
                            .HostMemory("delta"),
                        ForOp);

// FakeParamOp allocates a tensor with a shape conforming to the expected
// output. This is necessary if the value will be stored in a while_loop's
// TensorList. The output is otherwise not expected to be consumed by anything
// else.
class FakeParamOp : public OpKernel {
 public:
  explicit FakeParamOp(OpKernelConstruction* context) : OpKernel(context) {
    DataType dtype;
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype));

    // Set shape to the specified shape, setting unknown dimensions to empty.
    // If the specified shape is unknown, leave as an empty shape.
    TensorShape shape;
    PartialTensorShape partial_shape;
    OP_REQUIRES_OK(context, context->GetAttr("shape", &partial_shape));
    if (!partial_shape.unknown_rank()) {
      for (int64 d : partial_shape.dim_sizes()) {
        shape.AddDim(d == -1 ? 0 : d);
      }
    }

    // Create a persistent tensor that we can repeatedly return to save memory.
    // TODO(b/119612758): add optimization to prevent sending this across
    // devices on each Compute() call.
    OP_REQUIRES_OK(context, context->allocate_persistent(
                                dtype, shape, &value_handle_, nullptr));
  }

  void Compute(OpKernelContext* context) override {
    context->set_output(0, *value_handle_.AccessTensor(context));
  }

 private:
  PersistentTensor value_handle_;
};

REGISTER_KERNEL_BUILDER(Name("FakeParam").Device(DEVICE_CPU), FakeParamOp);
REGISTER_KERNEL_BUILDER(Name("FakeParam").Device(DEVICE_GPU), FakeParamOp);

}  // namespace
}  // namespace tensorflow
