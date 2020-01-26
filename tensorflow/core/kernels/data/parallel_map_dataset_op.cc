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
#include "tensorflow/core/kernels/data/parallel_map_dataset_op.h"

#include <deque>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ParallelMapDatasetOp::kDatasetType;
/* static */ constexpr const char* const ParallelMapDatasetOp::kInputDataset;
/* static */ constexpr const char* const ParallelMapDatasetOp::kOtherArguments;
/* static */ constexpr const char* const
    ParallelMapDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const ParallelMapDatasetOp::kFunc;
/* static */ constexpr const char* const ParallelMapDatasetOp::kTarguments;
/* static */ constexpr const char* const ParallelMapDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ParallelMapDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    ParallelMapDatasetOp::kUseInterOpParallelism;
/* static */ constexpr const char* const ParallelMapDatasetOp::kSloppy;
/* static */ constexpr const char* const
    ParallelMapDatasetOp::kPreserveCardinality;

// Period between reporting dataset statistics.
constexpr int kStatsReportingPeriodMillis = 1000;

class ParallelMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int32 num_parallel_calls, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes, bool sloppy,
          std::unique_ptr<CapturedFunction> captured_func,
          bool preserve_cardinality)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        num_parallel_calls_(num_parallel_calls),
        output_types_(output_types),
        output_shapes_(output_shapes),
        sloppy_(sloppy),
        preserve_cardinality_(preserve_cardinality),
        captured_func_(std::move(captured_func)) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    std::unique_ptr<ParallelMapFunctor> parallel_map_functor =
        absl::make_unique<ParallelMapDatasetFunctor>(this);
    return NewParallelMapIterator(
        {this, name_utils::IteratorPrefix(kDatasetType, prefix)}, input_,
        std::move(parallel_map_functor), num_parallel_calls_, sloppy_,
        preserve_cardinality_);
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(ParallelMapDatasetOp::kDatasetType);
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    // Input: input_dataset
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    // Input: other_arguments
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));

    // Input: num_parallel_calls
    Node* num_parallel_calls = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(num_parallel_calls_, &num_parallel_calls));

    // Attr: f
    AttrValue f_attr;
    b->BuildAttrValue(captured_func_->func(), &f_attr);

    // Attr: Targuments
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    // Attr: use_inter_op_parallelism
    AttrValue use_inter_op_parallelism_attr;
    b->BuildAttrValue(captured_func_->use_inter_op_parallelism(),
                      &use_inter_op_parallelism_attr);

    // Attr: sloppy
    AttrValue sloppy_attr;
    b->BuildAttrValue(sloppy_, &sloppy_attr);

    // Attr: preserve_cardinality
    AttrValue preserve_cardinality_attr;
    b->BuildAttrValue(preserve_cardinality_, &preserve_cardinality_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {std::make_pair(0, input_graph_node),
         std::make_pair(2, num_parallel_calls)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},     // Tensor list inputs.
        {std::make_pair(kFunc, f_attr),
         std::make_pair(kTarguments, other_arguments_types_attr),
         std::make_pair(kUseInterOpParallelism, use_inter_op_parallelism_attr),
         std::make_pair(kSloppy, sloppy_attr),
         std::make_pair(kPreserveCardinality,
                        preserve_cardinality_attr)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  class ParallelMapDatasetFunctor : public ParallelMapFunctor {
   public:
    explicit ParallelMapDatasetFunctor(const Dataset* dataset)
        : dataset_(dataset) {}

    Status InitFunc(IteratorContext* ctx) override {
      return dataset_->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    void MapFunc(IteratorContext* ctx, const string& prefix,
                 std::vector<Tensor> input_element, std::vector<Tensor>* result,
                 StatusCallback done) override {
      auto map_func = [this](IteratorContext* ctx, const string& prefix,
                             std::vector<Tensor> input_element,
                             std::vector<Tensor>* result, StatusCallback done) {
        instantiated_captured_func_->RunAsync(ctx, std::move(input_element),
                                              result, std::move(done), prefix);
      };
      if (!dataset_->captured_func_->use_inter_op_parallelism()) {
        (*ctx->runner())(std::bind(map_func, ctx, prefix,
                                   std::move(input_element), result,
                                   std::move(done)));
      } else {
        map_func(ctx, prefix, std::move(input_element), result,
                 std::move(done));
      }
    }

   private:
    const Dataset* const dataset_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const int32 num_parallel_calls_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const bool sloppy_;
  const bool preserve_cardinality_;
  const std::unique_ptr<CapturedFunction> captured_func_;
};

ParallelMapDatasetOp::ParallelMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseInterOpParallelism,
                                   &params.use_inter_op_parallelism));
  params.is_multi_device_function = true;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSloppy, &sloppy_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
}

void ParallelMapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                       DatasetBase** output) {
  int32 num_parallel_calls;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  OP_REQUIRES(
      ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
      errors::InvalidArgument("num_parallel_calls must be greater than zero."));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  if (num_parallel_calls == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output =
      new Dataset(ctx, input, num_parallel_calls, output_types_, output_shapes_,
                  sloppy_, std::move(captured_func), preserve_cardinality_);
}

namespace {

constexpr char kInvocationResults[] = "invocation_results";
constexpr char kSizeSuffix[] = ".size";
constexpr char kEndOfInputSuffix[] = ".end_of_input";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessage[] = ".error_message";

class ParallelMapIterator : public DatasetBaseIterator {
 public:
  struct Params {
    Params(std::unique_ptr<ParallelMapFunctor> parallel_map_functor,
           int32 num_parallel_calls, bool sloppy, bool preserve_cardinality)
        : parallel_map_functor(std::move(parallel_map_functor)),
          num_parallel_calls(num_parallel_calls),
          sloppy(sloppy),
          preserve_cardinality(preserve_cardinality) {}

    std::unique_ptr<ParallelMapFunctor> parallel_map_functor;
    int32 num_parallel_calls;
    bool sloppy;
    bool preserve_cardinality;
  };

  ParallelMapIterator(const DatasetBaseIterator::BaseParams& base_params,
                      const DatasetBase* input_dataset, Params params)
      : DatasetBaseIterator(base_params),
        input_dataset_(input_dataset),
        parallel_map_functor_(std::move(params.parallel_map_functor)),
        mu_(std::make_shared<mutex>()),
        cond_var_(std::make_shared<condition_variable>()),
        num_parallel_calls_(std::make_shared<model::SharedState>(
            params.num_parallel_calls, mu_, cond_var_)),
        sloppy_(params.sloppy),
        preserve_cardinality_(params.preserve_cardinality),
        autotune_(params.num_parallel_calls == model::kAutotune),
        key_prefix_(base_params.dataset->node_name()) {}

  ~ParallelMapIterator() override {
    CancelThreads(/*wait=*/true);
    if (deregister_fn_) deregister_fn_();
  }

  string BuildTraceMeName() override {
    int64 parallelism = -1;
    // NOTE: We only set the parallelism value if the lock can be acquired right
    // away to avoid introducing tracing overhead.
    if (mu_->try_lock()) {
      parallelism = num_parallel_calls_->value;
      mu_->unlock();
    }
    return strings::StrCat(this->prefix(), "#parallelism=", parallelism,
                           ",autotune=", autotune_, ",deterministic=", !sloppy_,
                           "#");
  }

  Status Initialize(IteratorContext* ctx) override {
    mutex_lock l(*mu_);
    if (num_parallel_calls_->value == model::kAutotune) {
      num_parallel_calls_->value = ctx->runner_threadpool_size();
    }
    TF_RETURN_IF_ERROR(RegisterCancellationCallback(
        ctx->cancellation_manager(),
        [this]() { CancelThreads(/*wait=*/false); }, &deregister_fn_));
    TF_RETURN_IF_ERROR(
        input_dataset_->MakeIterator(ctx, prefix(), &input_impl_));
    return parallel_map_functor_->InitFunc(ctx);
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    std::shared_ptr<InvocationResult> result;
    {
      mutex_lock l(*mu_);
      EnsureThreadsStarted(ctx);
      while (ShouldWait(&result)) {
        RecordStop(ctx);
        cond_var_->wait(l);
        RecordStart(ctx);
      }
      if (cancelled_) {
        return errors::Cancelled("Iterator was cancelled");
      }
    }
    RecordStop(ctx);
    result->notification.WaitForNotification();
    RecordStart(ctx);
    return ProcessResult(ctx, result, out_tensors, end_of_sequence);
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeAsyncKnownRatioNode(
        std::move(args),
        /*ratio=*/1,
        {model::MakeParameter("parallelism", num_parallel_calls_, /*min=*/1,
                              /*max=*/ctx->runner_threadpool_size())});
  }

  Status SaveInternal(IteratorStateWriter* writer) override {
    mutex_lock l(*mu_);
    // Wait for all in-flight calls to complete.
    while (num_calls_ > 0) {
      cond_var_->wait(l);
    }
    if (num_calls_ != 0) {
      return errors::FailedPrecondition(
          "Unexpected outstanding calls encountered.");
    }
    TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(
        full_name(strings::StrCat(kInvocationResults, kSizeSuffix)),
        invocation_results_.size()));
    for (size_t i = 0; i < invocation_results_.size(); i++) {
      const auto& result = *(invocation_results_[i]);
      TF_RETURN_IF_ERROR(WriteStatusLocked(writer, i, result.status));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(strings::StrCat(kInvocationResults, "[",
                                                        i, "]", kSizeSuffix)),
                              result.return_values.size()));
      for (size_t j = 0; j < result.return_values.size(); j++) {
        TF_RETURN_IF_ERROR(
            writer->WriteTensor(full_name(strings::StrCat(
                                    kInvocationResults, "[", i, "][", j, "]")),
                                result.return_values[j]));
      }
      if (result.end_of_input) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(kInvocationResults, "[", i, "]",
                                      kEndOfInputSuffix)),
            ""));
      }
    }
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    mutex_lock l(*mu_);
    TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
    int64 invocation_results_size;
    TF_RETURN_IF_ERROR(reader->ReadScalar(
        full_name(strings::StrCat(kInvocationResults, kSizeSuffix)),
        &invocation_results_size));
    if (!invocation_results_.empty()) invocation_results_.clear();
    for (size_t i = 0; i < invocation_results_size; i++) {
      invocation_results_.push_back(std::make_shared<InvocationResult>());
      auto& result = *invocation_results_.back();
      TF_RETURN_IF_ERROR(ReadStatusLocked(reader, i, &result.status));
      size_t num_return_values;
      {
        int64 size;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(
                strings::StrCat(kInvocationResults, "[", i, "]", kSizeSuffix)),
            &size));
        num_return_values = static_cast<size_t>(size);
        if (num_return_values != size) {
          return errors::InvalidArgument(strings::StrCat(
              full_name(strings::StrCat(kInvocationResults, "[", i, "]",
                                        kSizeSuffix)),
              ": ", size, " is not a valid value of type size_t."));
        }
      }
      result.return_values.reserve(num_return_values);
      for (size_t j = 0; j < num_return_values; j++) {
        result.return_values.emplace_back();
        TF_RETURN_IF_ERROR(
            reader->ReadTensor(full_name(strings::StrCat(kInvocationResults,
                                                         "[", i, "][", j, "]")),
                               &result.return_values.back()));
      }
      result.end_of_input = reader->Contains(full_name(
          strings::StrCat(kInvocationResults, "[", i, "]", kEndOfInputSuffix)));
      result.notification.Notify();
    }
    return Status::OK();
  }

 private:
  struct InvocationResult {
    Notification notification;
    Status status;
    std::vector<Tensor> return_values;
    bool end_of_input;
  };

  void CancelThreads(bool wait) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(*mu_);
    cancelled_ = true;
    cond_var_->notify_all();
    // Wait for all in-flight calls to complete.
    while (wait && num_calls_ > 0) {
      cond_var_->wait(l);
    }
  }

  void EnsureThreadsStarted(IteratorContext* ctx)
      EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    if (!runner_thread_) {
      auto ctx_copy = std::make_shared<IteratorContext>(*ctx);
      runner_thread_ = ctx->StartThread(
          "tf_data_parallel_map",
          std::bind(&ParallelMapIterator::RunnerThread, this, ctx_copy));
      if (ctx->stats_aggregator()) {
        stats_thread_ = ctx->StartThread(
            "tf_data_parallel_map_stats",
            std::bind(&ParallelMapIterator::StatsThread, this, ctx_copy));
      }
    }
  }

  void CallCompleted(const std::shared_ptr<IteratorContext>& ctx,
                     const std::shared_ptr<InvocationResult>& result)
      LOCKS_EXCLUDED(*mu_) {
    mutex_lock l(*mu_);
    num_calls_--;
    RecordBufferEnqueue(ctx.get(), result->return_values);
    result->notification.Notify();
    cond_var_->notify_all();
  }

  void CallFunction(const std::shared_ptr<IteratorContext>& ctx,
                    const std::shared_ptr<InvocationResult>& result)
      LOCKS_EXCLUDED(*mu_) {
    // Get the next input element.
    std::vector<Tensor> input_element;
    result->status =
        input_impl_->GetNext(ctx.get(), &input_element, &result->end_of_input);
    if (result->end_of_input || !result->status.ok()) {
      CallCompleted(ctx, result);
      return;
    }

    auto done = [this, ctx, result](Status status) {
      result->status.Update(status);
      CallCompleted(ctx, result);
    };

    // Apply the map function on `input_element`, storing the result in
    // `result->return_values`, and invoking `done` when finished.
    parallel_map_functor_->MapFunc(ctx.get(), prefix(),
                                   std::move(input_element),
                                   &result->return_values, std::move(done));
  }

  Status ProcessResult(IteratorContext* ctx,
                       const std::shared_ptr<InvocationResult>& result,
                       std::vector<Tensor>* out_tensors, bool* end_of_sequence)
      LOCKS_EXCLUDED(*mu_) {
    if (!result->end_of_input && result->status.ok()) {
      *out_tensors = std::move(result->return_values);
      RecordBufferDequeue(ctx, *out_tensors);
      *end_of_sequence = false;
      return Status::OK();
    }
    if (errors::IsOutOfRange(result->status)) {
      if (preserve_cardinality_) {
        // To guarantee that the transformation preserves the cardinality of the
        // dataset, we convert `OutOfRange` to `InvalidArgument` as the former
        // may be interpreted by a caller as the end of sequence.
        return errors::InvalidArgument(
            "Function invocation produced OutOfRangeError: ",
            result->status.error_message());
      } else {
        // `f` may deliberately raise `errors::OutOfRange` to indicate
        // that we should terminate the iteration early.
        *end_of_sequence = true;
        return Status::OK();
      }
    }
    *end_of_sequence = result->end_of_input;
    return result->status;
  }

  void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
      LOCKS_EXCLUDED(*mu_) {
    RecordStart(ctx.get());
    auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
    std::vector<std::shared_ptr<InvocationResult>> new_calls;
    {
      tf_shared_lock l(*mu_);  // mu_ == num_parallel_calls_->mu
      new_calls.reserve(num_parallel_calls_->value);
    }
    auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
      int64 num_parallel_calls = num_parallel_calls_->value;
      return num_calls_ >= num_parallel_calls ||
             invocation_results_.size() >= num_parallel_calls;
    };
    while (true) {
      {
        mutex_lock l(*mu_);
        while (!cancelled_ && busy()) {
          RecordStop(ctx.get());
          cond_var_->wait(l);
          RecordStart(ctx.get());
        }
        if (cancelled_) {
          return;
        }
        while (!busy()) {
          invocation_results_.push_back(std::make_shared<InvocationResult>());
          new_calls.push_back(invocation_results_.back());
          num_calls_++;
        }
        cond_var_->notify_all();
      }
      for (const auto& call : new_calls) {
        CallFunction(ctx, call);
      }
      new_calls.clear();
    }
  }

  // Determines whether the caller needs to wait for a result. Upon returning
  // false, `result` will point to the result.
  bool ShouldWait(std::shared_ptr<InvocationResult>* result)
      EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    if (cancelled_) {
      return false;
    }
    if (sloppy_) {
      for (auto it = invocation_results_.begin();
           it != invocation_results_.end(); ++it) {
        if ((*it)->notification.HasBeenNotified() &&
            (it == invocation_results_.begin() || !(*it)->end_of_input)) {
          std::swap(*result, *it);
          invocation_results_.erase(it);
          cond_var_->notify_all();
          return false;
        }
      }
    } else if (!invocation_results_.empty()) {
      std::swap(*result, invocation_results_.front());
      invocation_results_.pop_front();
      cond_var_->notify_all();
      return false;
    }
    return true;
  }

  void StatsThread(const std::shared_ptr<IteratorContext>& ctx) {
    for (int64 step = 0;; ++step) {
      int num_calls;
      int num_parallel_calls;
      {
        mutex_lock l(*mu_);
        if (step != 0 && !cancelled_) {
          cond_var_->wait_for(
              l, std::chrono::milliseconds(kStatsReportingPeriodMillis));
        }
        if (cancelled_) {
          return;
        }
        num_calls = num_calls_;
        num_parallel_calls = num_parallel_calls_->value;
      }
      if (num_parallel_calls == 0) {
        // Avoid division by zero.
        num_parallel_calls = 1;
      }
      ctx->stats_aggregator()->AddScalar(
          stats_utils::ThreadUtilizationScalarName(key_prefix_),
          static_cast<float>(num_calls) /
              static_cast<float>(num_parallel_calls),
          step);
    }
  }

  Status WriteStatusLocked(IteratorStateWriter* writer, size_t index,
                           const Status& status)
      EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(CodeKey(index), static_cast<int64>(status.code())));
    if (!status.ok()) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(ErrorMessageKey(index), status.error_message()));
    }
    return Status::OK();
  }

  Status ReadStatusLocked(IteratorStateReader* reader, size_t index,
                          Status* status) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
    int64 code_int;
    TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
    error::Code code = static_cast<error::Code>(code_int);

    if (code != error::Code::OK) {
      tstring error_message;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(ErrorMessageKey(index), &error_message));
      *status = Status(code, error_message);
    } else {
      *status = Status::OK();
    }
    return Status::OK();
  }

  string CodeKey(size_t index) {
    return full_name(
        strings::StrCat(kInvocationResults, "[", index, "]", kCodeSuffix));
  }

  string ErrorMessageKey(size_t index) {
    return full_name(
        strings::StrCat(kInvocationResults, "[", index, "]", kErrorMessage));
  }

  const DatasetBase* const input_dataset_;  // Not owned.
  std::unique_ptr<ParallelMapFunctor> parallel_map_functor_;
  // Used for coordination between the main thread and the runner thread.
  const std::shared_ptr<mutex> mu_;
  // Used for coordination between the main thread and the runner thread. In
  // particular, the runner thread should only schedule new calls when the
  // number of in-flight calls is less than the user specified level of
  // parallelism and there are slots available in the `invocation_results_`
  // buffer.
  const std::shared_ptr<condition_variable> cond_var_;
  // Identifies the maximum number of parallel calls.
  const std::shared_ptr<model::SharedState> num_parallel_calls_;
  // Determines whether outputs can be produced in non-deterministic order.
  const bool sloppy_;
  const bool preserve_cardinality_;
  const bool autotune_;
  const string key_prefix_;
  // Counts the number of outstanding calls.
  int64 num_calls_ GUARDED_BY(*mu_) = 0;
  std::unique_ptr<IteratorBase> input_impl_;
  // Buffer for storing the invocation results.
  std::deque<std::shared_ptr<InvocationResult>> invocation_results_
      GUARDED_BY(*mu_);

  std::unique_ptr<Thread> runner_thread_ GUARDED_BY(*mu_);
  std::unique_ptr<Thread> stats_thread_ GUARDED_BY(*mu_);
  bool cancelled_ GUARDED_BY(*mu_) = false;

  // Method for deregistering the cancellation callback.
  std::function<void()> deregister_fn_;
};

}  // namespace

std::unique_ptr<IteratorBase> NewParallelMapIterator(
    const DatasetBaseIterator::BaseParams& params,
    const DatasetBase* input_dataset,
    std::unique_ptr<ParallelMapFunctor> parallel_map_functor,
    int32 num_parallel_calls, bool sloppy, bool preserve_cardinality) {
  return absl::make_unique<ParallelMapIterator>(
      params, input_dataset,
      ParallelMapIterator::Params{std::move(parallel_map_functor),
                                  num_parallel_calls, sloppy,
                                  preserve_cardinality});
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ParallelMapDataset").Device(DEVICE_CPU),
                        ParallelMapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("ParallelMapDataset");
}  // namespace
}  // namespace data
}  // namespace tensorflow
