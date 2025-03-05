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

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/logging.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/data/unbounded_thread_pool.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
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
/* static */ constexpr const char* const ParallelMapDatasetOp::kDeterministic;
/* static */ constexpr const char* const ParallelMapDatasetOp::kSloppy;
/* static */ constexpr const char* const
    ParallelMapDatasetOp::kPreserveCardinality;

namespace {

constexpr char kParallelMapDatasetV1[] = "ParallelMapDataset";
constexpr char kParallelMapDatasetV2[] = "ParallelMapDatasetV2";

constexpr char kInvocationResults[] = "invocation_results";
constexpr char kSize[] = "size";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kErrorCode[] = "code";
constexpr char kErrorMessage[] = "error_message";

// Period between reporting dataset statistics.
constexpr int kStatsReportingPeriodMillis = 1000;

// Factor used to determine the autotune parallelism limit when using an
// unbounded threadpool. The limit is determined by multiplying this factor
// by the default threadpool size, which is typically based on the number of
// CPU cores. Without this limit, we see autotune sometimes choose unreasonably
// large values for the parallelism, e.g. creating 300k threads.
constexpr int kUnboundedThreadpoolAutotuningFactor = 10;

}  // namespace

class ParallelMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int64_t num_parallel_calls, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          DeterminismPolicy deterministic,
          std::unique_ptr<CapturedFunction> captured_func,
          bool preserve_cardinality, bool use_unbounded_threadpool,
          int op_version)
      : Dataset(DatasetContext(ctx), input, num_parallel_calls, output_types,
                output_shapes, deterministic, std::move(captured_func),
                preserve_cardinality, use_unbounded_threadpool, op_version) {}

  Dataset(DatasetContext dataset_context, const DatasetBase* input,
          int64_t num_parallel_calls, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          DeterminismPolicy deterministic,
          std::unique_ptr<CapturedFunction> captured_func,
          bool preserve_cardinality, bool use_unbounded_threadpool,
          int op_version)
      : DatasetBase(std::move(dataset_context)),
        input_(input),
        num_parallel_calls_(num_parallel_calls),
        output_types_(output_types),
        output_shapes_(output_shapes),
        deterministic_(deterministic),
        preserve_cardinality_(preserve_cardinality),
        use_unbounded_threadpool_(use_unbounded_threadpool),
        captured_func_(std::move(captured_func)),
        op_version_(op_version) {
    input_->Ref();
    random_indexing_compatible_ = absl::OkStatus();
    if (input_ != nullptr) {
      random_indexing_compatible_ = input_->RandomIndexingCompatible();
    }
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    return name_utils::DatasetDebugString(ParallelMapDatasetOp::kDatasetType,
                                          params);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    if (preserve_cardinality_) {
      return input_->Cardinality(options);
    } else {
      return kUnknownCardinality;
    }
  }

  absl::Status Get(OpKernelContext* ctx, int64 index,
                   std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    absl::call_once(instantiated_captured_func_once_, [this, ctx] {
      instantiated_captured_func_status_ = captured_func_->Instantiate(
          InstantiateCapturedFunctionParams(ctx), &instantiated_captured_func_);
    });
    TF_RETURN_IF_ERROR(instantiated_captured_func_status_);
    std::vector<Tensor> args;
    TF_RETURN_IF_ERROR(input_->Get(ctx, index, &args));
    return instantiated_captured_func_->RunInstantiated(args, out_tensors);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
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
    if (op_version_ == 1) {
      TF_RETURN_IF_ERROR(b->AddScalar(static_cast<int32>(num_parallel_calls_),
                                      &num_parallel_calls));
    } else {
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls));
    }
    std::vector<std::pair<absl::string_view, AttrValue>> attrs;

    // Attr: f
    AttrValue f_attr;
    b->BuildAttrValue(captured_func_->func(), &f_attr);
    attrs.emplace_back(kFunc, f_attr);

    // Attr: Targuments
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
    attrs.emplace_back(kTarguments, other_arguments_types_attr);

    // Attr: use_inter_op_parallelism
    AttrValue use_inter_op_parallelism_attr;
    b->BuildAttrValue(captured_func_->use_inter_op_parallelism(),
                      &use_inter_op_parallelism_attr);
    attrs.emplace_back(kUseInterOpParallelism, use_inter_op_parallelism_attr);

    if (op_version_ == 1) {
      // Attr: sloppy
      AttrValue sloppy_attr;
      b->BuildAttrValue(deterministic_.IsNondeterministic(), &sloppy_attr);
      attrs.emplace_back(kSloppy, sloppy_attr);
    }
    if (op_version_ == 2) {
      AttrValue deterministic_attr;
      b->BuildAttrValue(deterministic_.String(), &deterministic_attr);
      attrs.emplace_back(kDeterministic, deterministic_attr);
    }

    // Attr: preserve_cardinality
    AttrValue preserve_cardinality_attr;
    b->BuildAttrValue(preserve_cardinality_, &preserve_cardinality_attr);
    attrs.emplace_back(kPreserveCardinality, preserve_cardinality_attr);

    // Attr: use_unbounded_threadpool
    AttrValue use_unbounded_threadpool_attr;
    b->BuildAttrValue(use_unbounded_threadpool_,
                      &use_unbounded_threadpool_attr);
    attrs.emplace_back(kUseUnboundedThreadpool, use_unbounded_threadpool_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {std::make_pair(0, input_graph_node),
         std::make_pair(2, num_parallel_calls)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},     // Tensor list inputs.
        attrs, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          cond_var_(std::make_shared<condition_variable>()),
          num_parallel_calls_(std::make_shared<model::SharedState>(
              params.dataset->num_parallel_calls_, mu_, cond_var_)),
          deterministic_(params.dataset->deterministic_.IsDeterministic() ||
                         params.dataset->deterministic_.IsDefault()),
          preserve_cardinality_(params.dataset->preserve_cardinality_),
          use_unbounded_threadpool_(params.dataset->use_unbounded_threadpool_),
          autotune_(params.dataset->num_parallel_calls_ == model::kAutotune) {}

    ~Iterator() override {
      CancelThreads(/*wait=*/true);
      input_impl_.reset();
      if (deregister_fn_) deregister_fn_();
    }

    bool SymbolicCheckpointCompatible() const override {
      return deterministic_;
    }

    absl::Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);
      interleave_depth_ = ctx->interleave_depth();
      if (use_unbounded_threadpool_) {
        unbounded_thread_pool_ = std::make_unique<UnboundedThreadPool>(
            ctx->env(), "tf_data_map_unbounded_thread_pool");
      }
      if (num_parallel_calls_->value == model::kAutotune) {
        num_parallel_calls_->value = GetAutotuneDefaultParallelism(ctx);
      }
      cancellation_manager_ = std::make_unique<CancellationManager>();
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(),
          [this]() { CancelThreads(/*wait=*/false); }, &deregister_fn_));
      auto params = std::make_unique<IteratorContext::Params>(ctx);
      params->cancellation_manager = cancellation_manager_.get();
      auto iter_ctx = std::make_unique<IteratorContext>(*params);
      TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
          iter_ctx.get(), this, prefix(), &input_impl_));
      ctx->MergeCheckpoint(iter_ctx->checkpoint());
      TF_RETURN_IF_ERROR(dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_));
      if (ctx->warm_start() && !ctx->is_restoring()) {
        EnsureThreadsStarted(ctx);
      }
      return absl::OkStatus();
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
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
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode("ParallelMapConsume",
                                            {{"element_id", result->uid}});
      });
      return ProcessResult(ctx, result, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      std::shared_ptr<model::Parameter> parameter;
      double max_parallelism_value = ctx->runner_threadpool_size();
      if (use_unbounded_threadpool_) {
        max_parallelism_value *= kUnboundedThreadpoolAutotuningFactor;
      }

      int64_t min_parallelism =
          std::min(static_cast<double>(GetAutotuneMinParallelism(ctx)),
                   max_parallelism_value);

      LOG(INFO) << "Parallel map: " << prefix() << " min parallelism is set to "
                << min_parallelism;
      if (num_parallel_calls_ &&
          dataset()->num_parallel_calls_ == model::kAutotune) {
        parameter = model::MakeParameter(
            "parallelism", num_parallel_calls_, /*min=*/min_parallelism,
            /*max=*/max_parallelism_value,
            // This is to ensure before this op has seen its first element,
            // `MaximumBufferedBytes()` can use the correct `parameter->value`
            // to estimate the maximum buffer bytes.
            GetAutotuneDefaultParallelism(ctx));
      } else {
        parameter = model::MakeParameter("parallelism", num_parallel_calls_,
                                         /*min=*/min_parallelism,
                                         /*max=*/max_parallelism_value);
      }
      std::optional<int64_t> estimated_element_size =
          dataset()->GetEstimatedElementSize();
      if (!estimated_element_size) {
        VLOG(2) << absl::StrFormat(
            "Cannot estimate the size of the output tensor because the "
            "output shape of node %s(id:%d) is only partially known.",
            args.name, args.id);
      }

      return model::MakeAsyncKnownRatioNode(
          std::move(args),
          /*ratio=*/1, {std::move(parameter)},
          /*is_legacy_prefetch_autotuned=*/false, estimated_element_size);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      if (ctx->symbolic_checkpoint()) {
        return absl::OkStatus();
      }
      mutex_lock l(*mu_);
      // Wait for all in-flight calls to complete.
      while (num_calls_ > 0) {
        cond_var_->wait(l);
      }
      if (num_calls_ != 0) {
        return errors::FailedPrecondition(
            "Unexpected outstanding calls encountered.");
      }
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), absl::StrCat(kInvocationResults, "_", kSize),
          invocation_results_.size()));
      for (size_t i = 0; i < invocation_results_.size(); i++) {
        const auto& result = *(invocation_results_[i]);
        std::string element_prefix =
            absl::StrCat(prefix(), "_", kInvocationResults, "[", i, "]");
        TF_RETURN_IF_ERROR(
            WriteStatusLocked(writer, element_prefix, result.status));
        TF_RETURN_IF_ERROR(writer->WriteScalar(element_prefix, kSize,
                                               result.return_values.size()));
        for (size_t j = 0; j < result.return_values.size(); j++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(element_prefix,
                                                 absl::StrCat("[", j, "]"),
                                                 result.return_values[j]));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(element_prefix, kEndOfInput,
                                static_cast<int64_t>(result.end_of_input)));
      }
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      DCHECK(invocation_results_.empty());
      if (ctx->symbolic_checkpoint()) {
        return absl::OkStatus();
      }
      int64_t invocation_results_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          prefix(), absl::StrCat(kInvocationResults, "_", kSize),
          &invocation_results_size));
      for (size_t i = 0; i < invocation_results_size; i++) {
        invocation_results_.push_back(std::make_shared<InvocationResult>(ctx));
        auto& result = *invocation_results_.back();
        std::string element_prefix =
            absl::StrCat(prefix(), "_", kInvocationResults, "[", i, "]");
        TF_RETURN_IF_ERROR(
            ReadStatusLocked(reader, element_prefix, &result.status));
        size_t num_return_values;
        {
          int64_t size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(element_prefix, kSize, &size));
          num_return_values = static_cast<size_t>(size);
          if (num_return_values != size) {
            return errors::InvalidArgument(
                element_prefix, ",", kSize, ": ", size,
                " is not a valid value of type size_t.");
          }
        }
        result.return_values.reserve(num_return_values);
        for (size_t j = 0; j < num_return_values; j++) {
          result.return_values.emplace_back();
          TF_RETURN_IF_ERROR(reader->ReadTensor(ctx->flr(), element_prefix,
                                                absl::StrCat("[", j, "]"),
                                                &result.return_values.back()));
        }
        int64_t end_of_input;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(element_prefix, kEndOfInput, &end_of_input));
        result.end_of_input = static_cast<bool>(end_of_input);
        RecordBufferEnqueue(ctx, result.return_values);
        result.notification.Notify();
      }
      return absl::OkStatus();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      int64_t parallelism = -1;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        parallelism = num_parallel_calls_->value;
        mu_->unlock();
      }
      data::TraceMeMetadata result;
      result.push_back(
          std::make_pair("autotune", autotune_ ? "true" : "false"));
      result.push_back(
          std::make_pair("deterministic", deterministic_ ? "true" : "false"));
      result.push_back(
          std::make_pair("use_unbounded_threadpool",
                         use_unbounded_threadpool_ ? "true" : "false"));
      result.push_back(std::make_pair(
          "parallelism",
          parallelism == -1
              ? kTraceInfoUnavailable
              : strings::Printf("%lld", static_cast<long long>(parallelism))));
      result.push_back(std::make_pair(
          "interleave_depth",
          strings::Printf("%lld", static_cast<long long>(interleave_depth_))));
      return result;
    }

   private:
    struct InvocationResult {
      explicit InvocationResult(IteratorContext* ctx)
          : uid(tensorflow::EnvTime::NowNanos()),
            checkpoint(MemoryCheckpoint{ctx->id_registry()}) {}

      Notification notification;
      absl::Status status;
      std::vector<Tensor> return_values;
      bool end_of_input = false;
      const int64_t uid;
      MemoryCheckpoint checkpoint;
    };

    void CancelThreads(bool wait) TF_LOCKS_EXCLUDED(mu_) {
      cancellation_manager_->StartCancel();
      mutex_lock l(*mu_);
      cancelled_ = true;
      cond_var_->notify_all();
      // Wait for all in-flight calls to complete.
      while (wait && num_calls_ > 0) {
        cond_var_->wait(l);
      }
    }

    void EnsureThreadsStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!runner_thread_) {
        auto ctx_copy = std::make_shared<IteratorContext>(*ctx);
        runner_thread_ = ctx->StartThread(
            "tf_data_parallel_map",
            std::bind(&Iterator::RunnerThread, this, ctx_copy));
        if (ctx->stats_aggregator()) {
          stats_thread_ = ctx->StartThread(
              "tf_data_parallel_map_stats",
              std::bind(&Iterator::StatsThread, this, ctx_copy));
        }
      }
    }

    void CallCompleted(const std::shared_ptr<IteratorContext>& ctx,
                       const std::shared_ptr<InvocationResult>& result)
        TF_LOCKS_EXCLUDED(*mu_) {
      mutex_lock l(*mu_);
      num_calls_--;
      result->notification.Notify();
      cond_var_->notify_all();
    }

    void CallFunction(const std::shared_ptr<IteratorContext>& ctx,
                      const std::shared_ptr<InvocationResult>& result)
        TF_LOCKS_EXCLUDED(*mu_) {
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode("ParallelMapProduce",
                                            {{"element_id", result->uid}});
      });
      // Get the next input element.
      std::vector<Tensor> input_element;
      result->status = input_impl_->GetNext(ctx.get(), &input_element,
                                            &result->end_of_input);
      result->checkpoint.Merge(ctx->checkpoint());
      if (result->end_of_input || !result->status.ok()) {
        CallCompleted(ctx, result);
        return;
      }

      auto done = [this, ctx, result](absl::Status status) {
        if (!status.ok()) {
          result->status = AddErrorContext(status);
        }
        RecordBufferEnqueue(ctx.get(), result->return_values);
        CallCompleted(ctx, result);
      };

      // Apply the map function on `input_element`, storing the result in
      // `result->return_values`, and invoking `done` when finished.
      if (use_unbounded_threadpool_) {
        auto runner_fn = [this](std::function<void()> fn) {
          this->unbounded_thread_pool_->Schedule(fn);
        };
        instantiated_captured_func_->RunAsync(
            runner_fn, ctx->cancellation_manager(), ctx->collective_executor(),
            std::move(input_element), &result->return_values, done,
            model_node());
      } else if (dataset()->captured_func_->use_inter_op_parallelism()) {
        instantiated_captured_func_->RunAsync(
            ctx.get(), std::move(input_element), &result->return_values,
            std::move(done), model_node());
      } else {
        // In this case, the function will be executed using single-threaded
        // executor. We schedule it using `ctx->runner()` to enable concurrent
        // application of the function over different input elements.
        auto fn = std::bind(
            [this, ctx, result](std::vector<Tensor> input_element) {
              return instantiated_captured_func_->Run(
                  ctx.get(), std::move(input_element), &result->return_values,
                  model_node());
            },
            std::move(input_element));
        (*ctx->runner())(
            [this, ctx, fn = std::move(fn), done = std::move(done)]() {
              absl::Status s;
              // Check whether we are already recording to prevent invalid
              // nesting of `RecordStart` calls.
              if (IsRecording(ctx.get())) {
                s = fn();
              } else {
                RecordStart(ctx.get());
                s = fn();
                RecordStop(ctx.get());
              }
              done(s);
            });
      }
    }

    absl::Status ProcessResult(IteratorContext* ctx,
                               const std::shared_ptr<InvocationResult>& result,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) TF_LOCKS_EXCLUDED(*mu_) {
      ctx->MergeCheckpoint(&result->checkpoint);
      if (!result->end_of_input && result->status.ok()) {
        *out_tensors = std::move(result->return_values);
        RecordBufferDequeue(ctx, *out_tensors);
        *end_of_sequence = false;
        return absl::OkStatus();
      }
      if (errors::IsOutOfRange(result->status)) {
        if (preserve_cardinality_) {
          // To guarantee that the transformation preserves the cardinality of
          // the dataset, we convert `OutOfRange` to `InvalidArgument` as the
          // former may be interpreted by a caller as the end of sequence.
          return errors::InvalidArgument(
              "Function invocation produced OutOfRangeError: ",
              result->status.message());
        } else {
          // `f` may deliberately raise `errors::OutOfRange` to indicate
          // that we should terminate the iteration early.
          *end_of_sequence = true;
          return absl::OkStatus();
        }
      }
      *end_of_sequence = result->end_of_input;
      return result->status;
    }

    void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
        TF_LOCKS_EXCLUDED(*mu_) {
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      std::vector<std::shared_ptr<InvocationResult>> new_calls;
      {
        tf_shared_lock l(*mu_);  // mu_ == num_parallel_calls_->mu
        new_calls.reserve(num_parallel_calls_->value);
      }
      auto busy = [this]() TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
        int64_t num_parallel_calls = num_parallel_calls_->value;
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
            invocation_results_.push_back(
                std::make_shared<InvocationResult>(ctx.get()));
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
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (cancelled_) {
        return false;
      }
      if (!deterministic_) {
        // Iterate through in-flight results and return the first one that is
        // found to be available and not end-of-input. If the first result (in
        // order) is end-of-input, we know that all earlier iterations have
        // already been completed, so it is safe to return that result for the
        // caller to process end of iteration.
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

    void StatsThread(const std::shared_ptr<IteratorContext>& ctx)
        TF_LOCKS_EXCLUDED(*mu_) {
      for (int64_t step = 0;; ++step) {
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
            stats_utils::ThreadUtilizationScalarName(dataset()->node_name()),
            static_cast<float>(num_calls) /
                static_cast<float>(num_parallel_calls),
            step);
      }
    }

    absl::Status WriteStatusLocked(IteratorStateWriter* writer,
                                   const std::string& prefix,
                                   const absl::Status& status)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix, absl::StrCat("_", kErrorCode),
                              static_cast<int64_t>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(prefix,
                                               absl::StrCat("_", kErrorMessage),
                                               std::string(status.message())));
      }
      return absl::OkStatus();
    }

    absl::Status ReadStatusLocked(IteratorStateReader* reader,
                                  const std::string& prefix,
                                  absl::Status* status)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64_t code_int;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix, absl::StrCat("_", kErrorCode), &code_int));
      absl::StatusCode code = static_cast<absl::StatusCode>(code_int);

      if (code != absl::StatusCode::kOk) {
        tstring error_message;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            prefix, absl::StrCat("_", kErrorMessage), &error_message));
        *status = absl::Status(code, error_message);
      } else {
        *status = absl::OkStatus();
      }
      return absl::OkStatus();
    }

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
    const bool deterministic_;
    const bool preserve_cardinality_;
    const bool use_unbounded_threadpool_;
    const bool autotune_;
    // Counts the number of outstanding calls.
    int64_t num_calls_ TF_GUARDED_BY(*mu_) = 0;
    // Controls cancellation of `input_impl_`. Must be ordered before
    // `input_impl_` so that `input_impl_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    // Must be ordered after `cancellation_manager_` so that `input_impl_` is
    // destroyed first.
    std::unique_ptr<IteratorBase> input_impl_;
    // Buffer for storing the invocation results.
    std::deque<std::shared_ptr<InvocationResult>> invocation_results_
        TF_GUARDED_BY(*mu_);
    bool cancelled_ TF_GUARDED_BY(*mu_) = false;
    std::unique_ptr<Thread> runner_thread_ TF_GUARDED_BY(*mu_);
    std::unique_ptr<Thread> stats_thread_ TF_GUARDED_BY(*mu_);
    std::unique_ptr<UnboundedThreadPool> unbounded_thread_pool_;

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;

    // Records the number of ParallelInterleave operations in the path from the
    // root node to this node (not including this node) in the input pipeline
    // tree. We record the interleave depth so that it can be included in the
    // trace metadata.
    int64 interleave_depth_ = -1;
  };

  const DatasetBase* const input_;
  const int64_t num_parallel_calls_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const DeterminismPolicy deterministic_;
  const bool preserve_cardinality_;
  const bool use_unbounded_threadpool_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const int op_version_;
  // This is used for random access provided by Get().
  mutable absl::once_flag instantiated_captured_func_once_;
  mutable absl::Status instantiated_captured_func_status_;
  mutable std::unique_ptr<InstantiatedCapturedFunction>
      instantiated_captured_func_;
  absl::Status random_indexing_compatible_;
};

ParallelMapDatasetOp::ParallelMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), op_version_(ctx->HasAttr(kSloppy) ? 1 : 2) {
  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseInterOpParallelism,
                                   &params.use_inter_op_parallelism));
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  if (op_version_ == 1) {
    bool sloppy;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kSloppy, &sloppy));
    if (sloppy) {
      deterministic_ =
          DeterminismPolicy(DeterminismPolicy::Type::kNondeterministic);
    } else {
      deterministic_ = DeterminismPolicy(DeterminismPolicy::Type::kDefault);
    }
    use_unbounded_threadpool_ = false;
  }
  if (op_version_ == 2) {
    std::string deterministic;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDeterministic, &deterministic));
    OP_REQUIRES_OK(
        ctx, DeterminismPolicy::FromString(deterministic, &deterministic_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kUseUnboundedThreadpool, &use_unbounded_threadpool_));
  }
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
}

void ParallelMapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                       DatasetBase** output) {
  int64_t num_parallel_calls;
  if (op_version_ == 1) {
    int32_t parallel_calls;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument(ctx, kNumParallelCalls, &parallel_calls));
    num_parallel_calls = parallel_calls;
  }
  if (op_version_ == 2) {
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  }
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

  *output = new Dataset(ctx, input, num_parallel_calls, output_types_,
                        output_shapes_, deterministic_,
                        std::move(captured_func), preserve_cardinality_,
                        use_unbounded_threadpool_, op_version_);
}

std::unique_ptr<DatasetBase> MakeDataServiceUncompressDataset(
    DatasetBase* input, std::unique_ptr<CapturedFunction> captured_function,
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes) {
  DatasetContext::Params param;
  param.type_string = kParallelMapDatasetV2;
  param.node_name = kParallelMapDatasetV2;
  return std::make_unique<ParallelMapDatasetOp::Dataset>(
      DatasetContext(std::move(param)), input,
      /*num_parallel_calls=*/model::kAutotune, output_types, output_shapes,
      DeterminismPolicy(DeterminismPolicy::Type::kDefault),
      std::move(captured_function),
      /*preserve_cardinality=*/true,
      /*use_unbounded_threadpool=*/false, /*op_version=*/2);
}

namespace {
REGISTER_KERNEL_BUILDER(Name(kParallelMapDatasetV1).Device(DEVICE_CPU),
                        ParallelMapDatasetOp);
REGISTER_KERNEL_BUILDER(Name(kParallelMapDatasetV2).Device(DEVICE_CPU),
                        ParallelMapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION(kParallelMapDatasetV1);
REGISTER_INPUT_COLOCATION_EXEMPTION(kParallelMapDatasetV2);
}  // namespace
}  // namespace data
}  // namespace tensorflow
