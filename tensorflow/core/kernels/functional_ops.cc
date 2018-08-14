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
#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#endif
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/threadpool.h"

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

template <typename To, typename From>  // use like this: down_cast<T*>(foo);
inline To down_cast(From* f) {         // so we only accept pointers
  static_assert(
      (std::is_base_of<From, typename std::remove_pointer<To>::type>::value),
      "target type not derived from source type");

  // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
  // Uses RTTI in dbg and fastbuild. asserts are disabled in opt builds.
  assert(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

  return static_cast<To>(f);
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
        *v = !t[0].scalar<string>()().empty();
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
  opts->step_id = ctx->step_id();
  opts->rendezvous = ctx->rendezvous();
  opts->cancellation_manager = ctx->cancellation_manager();
  if (always_collect_stats) {
    opts->stats_collector = ctx->stats_collector();
  }
  opts->runner = ctx->runner();
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
    FHandle then_handle;
    FHandle else_handle;
    OP_REQUIRES_OK_ASYNC(ctx, Instantiate(lib, then_func_, &then_handle), done);
    OP_REQUIRES_OK_ASYNC(ctx, Instantiate(lib, else_func_, &else_handle), done);

    bool cond;
    OP_REQUIRES_OK(ctx, ToBool({ctx->input(0)}, &cond));
    (new State(this, ctx, cond, then_handle, else_handle, done))->Start();
  }

 private:
  NameAttrList then_func_;
  NameAttrList else_func_;

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
};

// TODO(drpng): remove this.
REGISTER_KERNEL_BUILDER(Name("_If").Device(DEVICE_CPU), IfOp);
REGISTER_KERNEL_BUILDER(Name("_If").Device(DEVICE_GPU).HostMemory("cond"),
                        IfOp);

REGISTER_KERNEL_BUILDER(Name("If").Device(DEVICE_CPU), IfOp);
REGISTER_KERNEL_BUILDER(Name("If").Device(DEVICE_GPU).HostMemory("cond"), IfOp);

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
    FHandle cond_handle;
    FHandle body_handle;
    OP_REQUIRES_OK_ASYNC(ctx, Instantiate(lib, cond_func_, &cond_handle), done);
    OP_REQUIRES_OK_ASYNC(ctx, Instantiate(lib, body_func_, &body_handle), done);
    (new State(this, ctx, cond_handle, body_handle, done))->Start();
  }

 private:
  NameAttrList cond_func_;
  NameAttrList body_func_;

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
      for (int i = 0; i < ctx_->num_inputs(); ++i) {
        args_.push_back(ctx_->input(i));
      }
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

    void EvalCond() {
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
      Tensor cond_t;
#if GOOGLE_CUDA
      const DeviceBase::GpuDeviceInfo* gpu_device_info =
          ctx_->device()->tensorflow_gpu_device_info();
      const bool is_hostmem_dtype =
          rets_[0].dtype() == DT_INT32 || rets_[0].dtype() == DT_INT64;
      if (!is_hostmem_dtype && gpu_device_info &&
          (opts_.rets_alloc_attrs.empty() ||
           !opts_.rets_alloc_attrs[0].on_host())) {
        // Copy the ret value to host if it's allocated on device.
        Device* device = down_cast<Device*>(ctx_->device());
        DeviceContext* device_ctx = ctx_->op_device_context();
        cond_t = Tensor(rets_[0].dtype(), rets_[0].shape());
        Notification done_copy;
        device_ctx->CopyDeviceTensorToCPU(
            &rets_[0], /*tensor_name=*/"", device, &cond_t,
            [&done_copy, &s](const Status& status) {
              s = status;
              done_copy.Notify();
            });
        done_copy.WaitForNotification();
        if (!s.ok()) {
          return Finish(s);
        }
      } else {
        cond_t = rets_[0];
      }
#else
      cond_t = rets_[0];
#endif
      bool cond;
      s = ToBool({cond_t}, &cond);

      if (!s.ok()) {
        return Finish(s);
      }
      if (!cond) {
        return Finish(Status::OK());
      }
      rets_.clear();
      lib_->Run(
          // Evaluate the body.
          opts_, body_handle_, args_, &rets_,
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
};
// TODO(drpng): remove these.
REGISTER_KERNEL_BUILDER(Name("_While").Device(DEVICE_CPU), WhileOp);
REGISTER_KERNEL_BUILDER(Name("_While").Device(DEVICE_GPU), WhileOp);

REGISTER_KERNEL_BUILDER(Name("While").Device(DEVICE_CPU), WhileOp);
REGISTER_KERNEL_BUILDER(Name("While").Device(DEVICE_GPU), WhileOp);

REGISTER_KERNEL_BUILDER(Name("StatelessWhile").Device(DEVICE_CPU), WhileOp);
REGISTER_KERNEL_BUILDER(Name("StatelessWhile").Device(DEVICE_GPU), WhileOp);

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

class FakeParamOp : public OpKernel {
 public:
  explicit FakeParamOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    // We must produce something (only Switch and Recvs are allowed to output
    // dead tensors). This output is not expected to be consumed by anything.
    Tensor output_tensor(dtype_, TensorShape({}));
    context->set_output(0, output_tensor);
  }

 private:
  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("FakeParam").Device(DEVICE_CPU), FakeParamOp);
REGISTER_KERNEL_BUILDER(Name("FakeParam").Device(DEVICE_GPU), FakeParamOp);

}  // namespace
}  // namespace tensorflow
