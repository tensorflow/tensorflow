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

#include "absl/time/clock.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/errors.h"
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
      start_time_ns_ = absl::GetCurrentTimeNanos();
    }

    void RecordComputeStarted() override {}

    void RecordComputeEnded() override {}

    void RecordExecutorEnded() override {
      end_time_ns_ = absl::GetCurrentTimeNanos();
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

Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const std::vector<Tensor>& input_element,
    int64 thread_index, const InstantiatedCapturedFunction& inst_captured_func,
    StringPiece prefix, std::unique_ptr<IteratorBase>* out_iterator) {
  std::vector<Tensor> return_values;

  TF_RETURN_IF_ERROR(inst_captured_func.RunWithBorrowedArgs(ctx, input_element,
                                                            &return_values));

  if (!(return_values.size() == 1 && return_values[0].dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(return_values[0].shape()))) {
    return errors::InvalidArgument(
        "Function must return a single scalar of dtype DT_VARIANT.");
  }

  // Retrieve the dataset that was created in `f`.
  DatasetBase* returned_dataset;
  TF_RETURN_IF_ERROR(
      GetDatasetFromVariantTensor(return_values[0], &returned_dataset));

  // Create an iterator for the dataset that was returned by `f`.
  return returned_dataset->MakeIterator(
      ctx, strings::StrCat(prefix, "[", thread_index, "]"), out_iterator);
}

/* static */
Status CapturedFunction::Create(
    const NameAttrList& func, OpKernelContext* ctx, const string& argument_name,
    Params params, std::unique_ptr<CapturedFunction>* out_function) {
  OpInputList inputs;
  TF_RETURN_IF_ERROR(ctx->input_list(argument_name, &inputs));
  std::vector<Tensor> captured_inputs(inputs.begin(), inputs.end());
  return Create(func, ctx, std::move(captured_inputs), std::move(params),
                out_function);
}

/* static */
Status CapturedFunction::Create(
    const NameAttrList& func, OpKernelContext* ctx,
    std::vector<Tensor>&& captured_inputs, Params params,
    std::unique_ptr<CapturedFunction>* out_function) {
  // TODO(b/130248232): Remove this code path once all users of CapturedFunction
  // are migrated to per-kernel FunctionLibraryDefinition.
  if (params.lib_def == nullptr) {
    TF_RETURN_IF_ERROR(CreateFunctionLibraryDefinition(
        ctx->function_library()->GetFunctionLibraryDefinition(), func.name(),
        &params.lib_def));
  }
  *out_function = absl::WrapUnique(new CapturedFunction(
      func, std::move(captured_inputs), std::move(params)));
  return Status::OK();
}

Status CapturedFunction::AddToGraph(
    SerializationContext* ctx, DatasetBase::DatasetGraphDefBuilder* b,
    std::vector<Node*>* other_arguments,
    DataTypeVector* other_arguments_types) const {
  other_arguments->reserve(captured_inputs_.size());
  other_arguments_types->reserve(captured_inputs_.size());
  for (const Tensor& t : captured_inputs_) {
    Node* node;
    DatasetBase* input;
    Status s = GetDatasetFromVariantTensor(t, &input);
    if (s.ok()) {
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &node));
    } else {
      TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
    }
    other_arguments->emplace_back(node);
    other_arguments_types->emplace_back(t.dtype());
  }

  TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name(), *lib_def_));

  return Status::OK();
}

Status CapturedFunction::Instantiate(
    IteratorContext* ctx, std::unique_ptr<InstantiatedCapturedFunction>*
                              instantiated_captured_function) {
  // The context's runtime will be used for all subsequent calls.
  FunctionLibraryRuntime* lib = ctx->lib();
  FunctionLibraryRuntime::InstantiateOptions inst_opts;
  inst_opts.lib_def = lib_def_.get();
  inst_opts.create_kernels_eagerly = true;
  if (!use_inter_op_parallelism_) {
    inst_opts.executor_type = "SINGLE_THREADED_EXECUTOR";
  }
  inst_opts.is_multi_device_function = is_multi_device_function_;

  // We infer the target device from the function library runtime.
  DCHECK(lib->device() != nullptr);
  inst_opts.target = lib->device()->name();

  if (is_multi_device_function_) {
    // Compute devices of non-captured inputs.
    //
    // We infer the number of non-captured inputs by subtracting the number
    // of captured inputs from the number of input arguments and we infer the
    // input devices from the function library runtime.
    const FunctionDef* fdef = lib_def_->Find(func_.name());
    if (fdef == nullptr) {
      return errors::InvalidArgument(
          "Failed to find function ", func_.name(),
          " in function library: ", lib->GetFunctionLibraryDefinition());
    }
    size_t num_non_captured_inputs =
        fdef->signature().input_arg_size() - captured_inputs_.size();
    for (size_t i = 0; i < num_non_captured_inputs; ++i) {
      inst_opts.input_devices.push_back(inst_opts.target);
    }
    // Compute devices of captured inputs.
    Device* cpu_device;
    TF_RETURN_IF_ERROR(lib->device_mgr()->LookupDevice("CPU:0", &cpu_device));
    for (auto& input : captured_inputs_) {
      DataType dtype = input.dtype();
      if (dtype == DT_RESOURCE) {
        const ResourceHandle& handle = input.flat<ResourceHandle>()(0);
        inst_opts.input_devices.push_back(handle.device());
      } else if (MTypeFromDType(dtype) == HOST_MEMORY) {
        // TODO(jsimsa): Correctly handle tensors on devices other than CPU:0.
        inst_opts.input_devices.push_back(cpu_device->name());
      } else {
        // Fall back to using the function library runtime device.
        inst_opts.input_devices.push_back(inst_opts.target);
      }
    }
  }

  FunctionLibraryRuntime::Handle f_handle;
  TF_RETURN_IF_ERROR(ctx->function_handle_cache()->Instantiate(
      func_.name(), AttrSlice(&func_.attr()), inst_opts, &f_handle));

  DataTypeVector ret_types;
  TF_RETURN_IF_ERROR(lib->GetRetTypes(f_handle, &ret_types));

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
    if (index < args_.size()) {
      // TODO(mrry): Consider making `CallFrameInterface::GetArg` non-const in
      // order to be able to `std::move(args_[index])` into `*val`.
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
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
    if (index < args_.size()) {
      *val = args_[index];
      return Status::OK();
    } else if (index < args_.size() + captured_inputs_->size()) {
      *val = (*captured_inputs_)[index - args_.size()];
      return Status::OK();
    } else {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
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
  if (lib_->device()->device_type() != DEVICE_CPU ||
      captured_func_->is_multi_device_function()) {
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

  if (captured_func_->is_multi_device_function()) {
    std::vector<Tensor> inputs = std::move(args);
    inputs.reserve(inputs.size() + captured_func_->captured_inputs().size());
    for (auto& input : captured_func_->captured_inputs()) {
      inputs.push_back(input);
    }

    Notification n;
    Status s;
    lib_->Run(f_opts, f_handle_, inputs, rets, [&n, &s](Status func_status) {
      s.Update(func_status);
      n.Notify();
    });
    n.WaitForNotification();
    return s;
  } else {
    OwnedArgsCallFrame frame(std::move(args),
                             &captured_func_->captured_inputs(), ret_types_);
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
        // TODO(b/129085499) Utilize the `node_name` which would be unique than
        // the prefix for the function execution time statistics.
        // prefix_with_func_name would then be node_name + func_name.
        if (stats_aggregator) {
          string prefix_end =
              str_util::Split(prefix, "::", str_util::SkipEmpty()).back();
          string prefix_with_func_name =
              strings::StrCat(prefix_end, stats_utils::kDelimiter,
                              captured_func_->func().name());
          stats_aggregator->AddToHistogram(
              stats_utils::ExecutionTimeHistogramName(prefix_with_func_name),
              {static_cast<float>(stats_collector->processing_time())},
              model->NumElements(prefix));
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
                                   Params params)
    : func_(func),
      captured_inputs_(std::move(captured_inputs)),
      use_inter_op_parallelism_(params.use_inter_op_parallelism),
      is_multi_device_function_(params.is_multi_device_function),
      lib_def_(std::move(params.lib_def)) {}

}  // namespace data
}  // namespace tensorflow
