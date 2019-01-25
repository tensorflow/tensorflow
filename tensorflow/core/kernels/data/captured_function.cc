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
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"

namespace tensorflow {
namespace data {

namespace {

// Simplistic implementation of the `StepStatsCollectorInterface` that only
// cares about collecting the CPU time needed to execute a captured function.
class SimpleStepStatsCollector : public StepStatsCollectorInterface {
 public:
  void IncrementProcessingTime(int64 delta) {
    mutex_lock l(mu_);
    processing_time_ += delta;
  }

  NodeExecStatsInterface* CreateNodeExecStats(const Node* node) override {
    return new SimpleNodeExecStats(this);
  }

  string ReportAllocsOnResourceExhausted(const string& err) override {
    return "";
  }

  int64 processing_time() {
    tf_shared_lock l(mu_);
    return processing_time_;
  }

 private:
  class SimpleNodeExecStats : public NodeExecStatsInterface {
   public:
    explicit SimpleNodeExecStats(SimpleStepStatsCollector* step_stats_collector)
        : step_stats_collector_(step_stats_collector) {}

    void Done(const string& device) override {
      step_stats_collector_->IncrementProcessingTime(end_time_ns_ -
                                                     start_time_ns_);
      delete this;
    }

    void RecordExecutorStarted() override {
      start_time_ns_ = Env::Default()->NowNanos();
    }

    void RecordComputeStarted() override {}

    void RecordComputeEnded() override {}

    void RecordExecutorEnded() override {
      end_time_ns_ = Env::Default()->NowNanos();
    }

    bool TrackAllocations() const override { return false; }

    void SetMemory(OpKernelContext* ctx) override {}

    void SetOutput(int slot, const Tensor* tensor) override {}

    void SetReferencedTensors(const TensorReferenceVector& tensors) override {}

    void SetScheduled(int64 nanos) override {}

   private:
    int64 start_time_ns_ = 0;
    int64 end_time_ns_ = 0;
    SimpleStepStatsCollector* step_stats_collector_;  // Not owned.
  };

  mutex mu_;
  int64 processing_time_ GUARDED_BY(mu_) = 0;
};

}  // namespace

/* static */
Status CapturedFunction::Create(
    const NameAttrList& func, OpKernelContext* ctx, const string& argument,
    std::unique_ptr<CapturedFunction>* out_function) {
  return CapturedFunction::Create(func, ctx, argument, true, out_function);
}

Status CapturedFunction::Create(
    const NameAttrList& func, OpKernelContext* ctx, const string& argument,
    bool use_inter_op_parallelism,
    std::unique_ptr<CapturedFunction>* out_function) {
  OpInputList inputs;
  TF_RETURN_IF_ERROR(ctx->input_list(argument, &inputs));
  std::vector<Tensor> arguments(inputs.begin(), inputs.end());
  *out_function = absl::WrapUnique(new CapturedFunction(
      func, std::move(arguments), use_inter_op_parallelism));
  return Status::OK();
}

Status CapturedFunction::Instantiate(
    IteratorContext* ctx, std::unique_ptr<InstantiatedCapturedFunction>*
                              instantiated_captured_function) {
  // The context's runtime will be used for all subsequent calls.
  FunctionLibraryRuntime* lib = ctx->lib();
  FunctionLibraryRuntime::InstantiateOptions inst_opts;
  inst_opts.overlay_lib = ctx->function_library().get();
  inst_opts.create_kernels_eagerly = true;
  if (!use_inter_op_parallelism_) {
    inst_opts.executor_type = "SINGLE_THREADED_EXECUTOR";
  }

  FunctionLibraryRuntime::Handle f_handle;
  TF_RETURN_IF_ERROR(ctx->function_handle_cache()->Instantiate(
      func_.name(), AttrSlice(&func_.attr()), inst_opts, &f_handle));
  const FunctionBody* fbody = lib->GetFunctionBody(f_handle);
  if (fbody == nullptr) {
    return errors::Internal("Failed to instantiate function body.");
  }

  DataTypeVector ret_types;
  for (const auto& ret_type : fbody->ret_types) {
    ret_types.push_back(ret_type);
  }

  *instantiated_captured_function =
      absl::WrapUnique<InstantiatedCapturedFunction>(
          new InstantiatedCapturedFunction(lib, f_handle, std::move(ret_types),
                                           *ctx->runner(), this));
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

InstantiatedCapturedFunction::InstantiatedCapturedFunction(
    FunctionLibraryRuntime* lib, FunctionLibraryRuntime::Handle f_handle,
    DataTypeVector ret_types, std::function<void(std::function<void()>)> runner,
    CapturedFunction* captured_func)
    : lib_(lib),
      f_handle_(f_handle),
      ret_types_(std::move(ret_types)),
      captured_runner_(std::move(runner)),
      captured_func_(captured_func) {}

// NOTE: We don't release f_handle_ here and instead delegate the function
// handle releasing to the FunctionHandleCache. This is because in some cases
// (RepeatDatasetOp in particular), we want to keep the function state (e.g.
// random number generator) even after the Iterator is reset after going through
// one epoch.
InstantiatedCapturedFunction::~InstantiatedCapturedFunction() {}

Status InstantiatedCapturedFunction::Run(IteratorContext* ctx,
                                         std::vector<Tensor>&& args,
                                         std::vector<Tensor>* rets) const {
  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = InstantiatedCapturedFunction::generate_step_id();
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  if (lib_->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  OwnedArgsCallFrame frame(std::move(args), &captured_func_->captured_inputs(),
                           ret_types_);
  Notification n;
  Status s;
  lib_->Run(f_opts, f_handle_, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

Status InstantiatedCapturedFunction::RunWithBorrowedArgs(
    IteratorContext* ctx, const std::vector<Tensor>& args,
    std::vector<Tensor>* rets) const {
  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = InstantiatedCapturedFunction::generate_step_id();
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  if (lib_->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  BorrowedArgsCallFrame frame(args, &captured_func_->captured_inputs(),
                              ret_types_);
  Notification n;
  Status s;

  lib_->Run(f_opts, f_handle_, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

Status InstantiatedCapturedFunction::RunInstantiated(
    const std::vector<Tensor>& args, std::vector<Tensor>* rets) {
  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = InstantiatedCapturedFunction::generate_step_id();
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = &captured_runner_;
  if (lib_->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager c_mgr;
  f_opts.cancellation_manager = &c_mgr;

  BorrowedArgsCallFrame frame(args, &captured_func_->captured_inputs(),
                              ret_types_);
  Notification n;
  Status s;

  lib_->Run(f_opts, f_handle_, &frame, [&n, &s](Status func_status) {
    s.Update(func_status);
    n.Notify();
  });
  n.WaitForNotification();
  TF_RETURN_IF_ERROR(s);
  return frame.ConsumeRetvals(rets);
}

void InstantiatedCapturedFunction::RunAsync(
    IteratorContext* ctx, std::vector<Tensor>&& args, std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done, const string& prefix) const {
  // NOTE(mrry): This method does not transfer ownership of `ctx`, and it may
  // be deleted before `done` is called. Take care not to capture `ctx` in any
  // code that may execute asynchronously in this function.
  OwnedArgsCallFrame* frame = new OwnedArgsCallFrame(
      std::move(args), &captured_func_->captured_inputs(), ret_types_);

  FunctionLibraryRuntime::Options f_opts;
  f_opts.step_id = InstantiatedCapturedFunction::generate_step_id();
  ResourceMgr* resource_mgr = lib_->device()->resource_manager();
  ScopedStepContainer* step_container = new ScopedStepContainer(
      f_opts.step_id, [resource_mgr](const string& name) {
        resource_mgr->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = step_container;
  f_opts.runner = ctx->runner();
  if (lib_->device()->device_type() != DEVICE_CPU) {
    f_opts.create_rendezvous = true;
  }
  // TODO(mrry): Add cancellation manager support to IteratorContext
  // so that we can cancel running map functions. The local
  // cancellation manager here is created so that we can run kernels
  // (such as queue kernels) that depend on the non-nullness of
  // `OpKernelContext::cancellation_manager()`, but additional effort
  // will be required to plumb it through the `IteratorContext`.
  CancellationManager* c_mgr = new CancellationManager();
  f_opts.cancellation_manager = c_mgr;
  std::shared_ptr<SimpleStepStatsCollector> stats_collector;
  if (ctx->model() || ctx->stats_aggregator()) {
    stats_collector = absl::make_unique<SimpleStepStatsCollector>();
  }
  f_opts.stats_collector = stats_collector.get();

  auto callback = std::bind(
      [this, rets, step_container, c_mgr, frame](
          const FunctionLibraryRuntime::DoneCallback& done,
          const std::shared_ptr<model::Model>& model,
          const std::shared_ptr<StatsAggregator>& stats_aggregator,
          const string& prefix,
          const std::shared_ptr<SimpleStepStatsCollector>& stats_collector,
          // Begin unbound arguments.
          Status s) {
        delete step_container;
        delete c_mgr;
        if (s.ok()) {
          s = frame->ConsumeRetvals(rets);
        }
        delete frame;

        if (stats_aggregator) {
          stats_aggregator->AddToHistogram(
              strings::StrCat(
                  str_util::Split(prefix, "::", str_util::SkipEmpty()).back(),
                  "::", captured_func_->func().name(), "::execution_time"),
              {static_cast<float>(stats_collector->processing_time())});
        }
        if (model) {
          model->AddProcessingTime(prefix, stats_collector->processing_time());
          model->RecordStart(prefix, false /* stop_output */);
        }
        done(s);
        if (model) {
          model->RecordStop(prefix, false /* start_output */);
        }
      },
      std::move(done), ctx->model(), ctx->stats_aggregator(), prefix,
      std::move(stats_collector), std::placeholders::_1);

  lib_->Run(f_opts, f_handle_, frame, std::move(callback));
}

CapturedFunction::CapturedFunction(const NameAttrList& func,
                                   std::vector<Tensor> captured_inputs,
                                   bool use_inter_op_parallelism)
    : func_(func),
      captured_inputs_(std::move(captured_inputs)),
      use_inter_op_parallelism_(use_inter_op_parallelism) {}

}  // namespace data
}  // namespace tensorflow
