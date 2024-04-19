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
#include "tensorflow/core/kernels/data/experimental/map_and_batch_dataset_op.h"

#include <atomic>
#include <functional>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const MapAndBatchDatasetOp::kDatasetType;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kInputDataset;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kOtherArguments;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kBatchSize;
/* static */ constexpr const char* const
    MapAndBatchDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kDropRemainder;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kFunc;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kTarguments;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kOutputTypes;
/* static */ constexpr const char* const MapAndBatchDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    MapAndBatchDatasetOp::kPreserveCardinality;

// Maximum number of batch results to buffer.

namespace {

constexpr int64_t kMaxBatchResults = 16;
constexpr char kParallelism[] = "parallelism";
constexpr char kCallCounter[] = "call_counter";
constexpr char kBatchResultsSize[] = "batch_results_size";
constexpr char kTFDataMapAndBatch[] = "tf_data_map_and_batch";
constexpr char kBatchResults[] = "batch_results";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kNumCalls[] = "num_calls";
constexpr char kNumElements[] = "num_elements";
constexpr char kOutputAllocated[] = "output_allocated";
constexpr char kStatus[] = "status";

// Computes ceil(x / y).
inline int64_t CeilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

}  // namespace

class MapAndBatchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t batch_size,
          int64_t num_parallel_calls, bool drop_remainder,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          std::unique_ptr<CapturedFunction> captured_func,
          bool preserve_cardinality)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        batch_size_(batch_size),
        num_parallel_calls_(num_parallel_calls),
        drop_remainder_(drop_remainder),
        output_types_(output_types),
        output_shapes_(output_shapes),
        captured_func_(std::move(captured_func)),
        preserve_cardinality_(preserve_cardinality),
        traceme_metadata_(
            {{"autotune",
              num_parallel_calls == model::kAutotune ? "true" : "false"},
             {"batch_size",
              strings::Printf("%lld", static_cast<long long>(batch_size))},
             {"drop_remainder", drop_remainder ? "true" : "false"}}) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    if (!preserve_cardinality_) {
      return kUnknownCardinality;
    }
    int64_t n = input_->Cardinality(options);
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / batch_size_ + (n % batch_size_ == 0 || drop_remainder_ ? 0 : 1);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* batch_size_node;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size_node));
    Node* num_parallel_calls_node;
    TF_RETURN_IF_ERROR(
        b->AddScalar(num_parallel_calls_, &num_parallel_calls_node));
    Node* drop_remainder_node;
    TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
    AttrValue preserve_cardinality_attr;
    b->BuildAttrValue(preserve_cardinality_, &preserve_cardinality_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {std::make_pair(0, input_graph_node),
         std::make_pair(2, batch_size_node),
         std::make_pair(3, num_parallel_calls_node),
         std::make_pair(4, drop_remainder_node)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},      // Tensor list inputs.
        {std::make_pair(kFunc, f),
         std::make_pair(kTarguments, other_arguments_types_attr),
         std::make_pair(kPreserveCardinality,
                        preserve_cardinality_attr)},  // Attrs
        output));
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
              params.dataset->num_parallel_calls_, mu_, cond_var_)) {
      // To mitigate the effect of stragglers (i.e. map invocations that take
      // much longer than others), we allow the kernel to pre-compute batches
      // ahead of time and store them in an internal buffer. The maximum number
      // of batches to buffer is a trade-off between performance and memory and
      // we derive it from the degree of parallelism and the batch size.
      //
      // TODO(b/178059273): If we handle RAM budget correctly, the upper bound
      // should be removed.
      max_batch_results_ = std::min(
          kMaxBatchResults,
          CeilDiv(params.dataset->num_parallel_calls_ == model::kAutotune
                      ? GetCpuBudget()  // maximum parallelism
                      : params.dataset->num_parallel_calls_,
                  params.dataset->batch_size_));
    }

    ~Iterator() override {
      CancelThreads(/*wait=*/true);
      if (deregister_fn_) deregister_fn_();
    }

    bool SymbolicCheckpointCompatible() const override { return true; }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);
      interleave_depth_ = ctx->interleave_depth();

      if (num_parallel_calls_->value == model::kAutotune) {
        num_parallel_calls_->value = GetAutotuneDefaultParallelism(ctx);
      }
      cancellation_manager_ = std::make_unique<CancellationManager>();
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(),
          [this]() { CancelThreads(/*wait=*/false); }, &deregister_fn_));
      IteratorContext::Params params(ctx);
      params.cancellation_manager = cancellation_manager_.get();
      IteratorContext iter_ctx(params);
      TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
          &iter_ctx, this, prefix(), &input_impl_));
      ctx->MergeCheckpoint(iter_ctx.checkpoint());
      TF_RETURN_IF_ERROR(dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_));
      if (ctx->warm_start() && !ctx->is_restoring()) {
        EnsureThreadsStarted(ctx);
      }
      return absl::OkStatus();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      std::shared_ptr<BatchResult> result;
      {
        mutex_lock l(*mu_);
        EnsureThreadsStarted(ctx);
        while (!cancelled_ && (batch_results_.empty() ||
                               batch_results_.front()->num_calls > 0)) {
          ++waiting_;
          RecordStop(ctx);
          cond_var_->wait(l);
          RecordStart(ctx);
          --waiting_;
        }
        if (cancelled_) {
          return errors::Cancelled("Iterator was cancelled");
        }
        std::swap(result, batch_results_.front());
        batch_results_.pop_front();
        cond_var_->notify_all();
      }
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode("MapAndBatchConsume",
                                            {{"element_id", result->uid}});
      });
      // Deallocate tensors allocated for the output.
      auto cleanup = gtl::MakeCleanup([result] { result->output.clear(); });
      mutex_lock l(result->mu);
      if (result->output_allocated) {
        RecordBufferDequeue(ctx, result->output);
      }
      ctx->MergeCheckpoint(&result->checkpoint);
      TF_RETURN_IF_ERROR(
          ProcessBatch(dataset()->batch_size_, result->num_elements,
                       dataset()->drop_remainder_, result->status, ctx,
                       out_tensors, end_of_sequence, &result->output));
      return absl::OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(
          std::move(args), dataset()->batch_size_,
          {model::MakeParameter(kParallelism, num_parallel_calls_, /*min=*/1,
                                /*max=*/ctx->runner_threadpool_size())});
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      if (ctx->symbolic_checkpoint()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kCallCounter, 0));
        TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kBatchResultsSize, 0));
        return absl::OkStatus();
      }
      mutex_lock l(*mu_);
      // Wait for all in-flight calls to complete.
      while (num_calls_ > 0) {
        cond_var_->wait(l);
      }
      DCHECK_EQ(num_calls_, 0);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kCallCounter, call_counter_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kBatchResultsSize,
                                             batch_results_.size()));
      for (size_t i = 0; i < batch_results_.size(); ++i) {
        TF_RETURN_IF_ERROR(WriteBatchResult(writer, i));
      }
      return absl::OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(*mu_);
      DCHECK(!runner_thread_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kCallCounter, &call_counter_));
      int64_t batch_results_size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kBatchResultsSize, &batch_results_size));
      DCHECK(batch_results_.empty());
      for (int i = 0; i < batch_results_size; ++i) {
        TF_RETURN_IF_ERROR(ReadBatchResult(ctx, reader, i));
      }
      if (ctx->warm_start()) {
        EnsureThreadsStarted(ctx);
      }
      return absl::OkStatus();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      int64_t parallelism = -1;
      int64_t max_batch_results = -1;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        parallelism = num_parallel_calls_->value;
        max_batch_results = max_batch_results_;
        mu_->unlock();
      }
      auto result = dataset()->traceme_metadata_;
      result.push_back(std::make_pair(
          "max_batch_results",
          strings::Printf("%lld", static_cast<long long>(max_batch_results))));
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
    // BatchResult encapsulates the output batch, as well as ancillary
    // metadata required to execute the fused map-and-batch operation.
    struct BatchResult {
      explicit BatchResult(int64_t batch_size, IteratorContext* ctx)
          : end_of_input(false),
            num_elements(0),
            output_allocated(false),
            status(absl::OkStatus()),
            status_offset(-1),
            num_calls(batch_size),
            checkpoint(MemoryCheckpoint{ctx->id_registry()}),
            uid(tensorflow::EnvTime::NowNanos()) {}

      // UpdateStatus updates the batch's aggregate Status.
      //
      // In order to ensure that exactly the first non-OK status is returned
      // (required to make the behavior is observably identical to a
      // sequential execution of map followed by batch), we must also keep
      // track of the offset into the batch that produced `s`.
      void UpdateStatus(const Status& s, int64_t offset) {
        if (TF_PREDICT_FALSE(!s.ok())) {
          mutex_lock l(mu);
          if (status.ok() || offset < status_offset) {
            status = s;
            status_offset = offset;
          }
        }
      }

      mutex mu;
      bool end_of_input TF_GUARDED_BY(mu);
      int64_t num_elements TF_GUARDED_BY(mu);
      std::vector<Tensor> output;
      bool output_allocated TF_GUARDED_BY(mu);
      Status status TF_GUARDED_BY(mu);
      int64_t status_offset TF_GUARDED_BY(mu);
      // Counts the number of outstanding calls for this batch.
      int64_t num_calls TF_GUARDED_BY(&Iterator::mu_);
      MemoryCheckpoint checkpoint TF_GUARDED_BY(mu);
      const uint64 uid = -1;
    };

    void CallCompleted(const std::shared_ptr<IteratorContext>& ctx,
                       const std::shared_ptr<BatchResult>& result)
        TF_LOCKS_EXCLUDED(*mu_) {
      mutex_lock l(*mu_);
      num_calls_--;
      result->num_calls--;
      const auto& stats_aggregator = ctx->stats_aggregator();
      if (stats_aggregator) {
        stats_aggregator->AddScalar(
            stats_utils::ThreadUtilizationScalarName(dataset()->node_name()),
            static_cast<float>(num_calls_) /
                static_cast<float>(num_parallel_calls_->value),
            num_elements());
      }
      cond_var_->notify_all();
    }

    void CallFunction(std::shared_ptr<IteratorContext> ctx,
                      const std::shared_ptr<BatchResult>& result,
                      int64_t offset) TF_LOCKS_EXCLUDED(*mu_) {
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode("MapAndBatchProduce",
                                            {{"element_id", result->uid}});
      });
      // Get the next input element.
      std::vector<Tensor> input_element;
      bool end_of_input = false;
      Status status =
          input_impl_->GetNext(ctx.get(), &input_element, &end_of_input);
      bool return_early;
      {
        mutex_lock l(result->mu);
        result->checkpoint.Merge(ctx->checkpoint());
        result->end_of_input = result->end_of_input || end_of_input;
        result->status.Update(status);
        return_early = result->end_of_input || !result->status.ok();
      }
      if (return_early) {
        CallCompleted(ctx, result);
        return;
      }

      std::shared_ptr<std::vector<Tensor>> return_values =
          std::make_shared<std::vector<Tensor>>();
      auto done = [this, ctx, result, return_values, offset](Status status) {
        if (dataset()->preserve_cardinality_ && errors::IsOutOfRange(status)) {
          // To guarantee that the transformation preserves the cardinality of
          // the dataset, we convert `OutOfRange` to `InvalidArgument` as the
          // former may be interpreted by a caller as the end of sequence.
          status = errors::InvalidArgument(
              "Function invocation produced OutOfRangeError: ",
              status.message());
        }
        result->UpdateStatus(status, offset);
        if (status.ok()) {
          Status allocate_status =
              EnsureOutputAllocated(ctx, result, return_values);
          if (!allocate_status.ok()) {
            result->UpdateStatus(allocate_status, offset);
          } else {
            for (size_t i = 0; i < return_values->size(); ++i) {
              Tensor& tensor = return_values->at(i);
              Tensor* batch = &(result->output)[i];
              if (tensor.NumElements() !=
                  (batch->NumElements() / batch->dim_size(0))) {
                TensorShape batch_shape = batch->shape();
                batch_shape.RemoveDim(0);
                result->UpdateStatus(
                    errors::InvalidArgument(
                        "Cannot add tensor to the batch: number of elements "
                        "does not match. Shapes are: [tensor]: ",
                        tensor.shape().DebugString(),
                        ", [batch]: ", batch_shape.DebugString()),
                    offset);
                break;
              }
              // TODO(mrry): Add a version of DoParallelConcat that allows us
              // to move `tensor` where possible, to speed up string tensor
              // batching.
              Status copy_status = batch_util::CopyElementToSlice(
                  std::move(tensor), batch, offset);
              if (!copy_status.ok()) {
                result->UpdateStatus(copy_status, offset);
                break;
              }
            }
          }
          {
            mutex_lock l(result->mu);
            result->num_elements++;
          }
        }
        CallCompleted(ctx, result);
      };

      // Apply the map function on `input_element`, storing the result in
      // `return_values`, and invoking `done` when finished.
      instantiated_captured_func_->RunAsync(ctx.get(), std::move(input_element),
                                            return_values.get(),
                                            std::move(done), model_node());
    }

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
        auto new_ctx = std::make_shared<IteratorContext>(*ctx);
        runner_thread_ =
            ctx->StartThread(kTFDataMapAndBatch,
                             std::bind(&Iterator::RunnerThread, this, new_ctx));
      }
    }

    Status EnsureOutputAllocated(
        const std::shared_ptr<IteratorContext>& ctx,
        const std::shared_ptr<BatchResult>& result,
        const std::shared_ptr<std::vector<Tensor>>& return_values) {
      mutex_lock l(result->mu);
      if (result->output_allocated) {
        return absl::OkStatus();
      }
      const size_t num_components = return_values->size();
      result->output.reserve(num_components);
      for (size_t i = 0; i < num_components; ++i) {
        TensorShape component_shape({dataset()->batch_size_});
        component_shape.AppendShape(return_values->at(i).shape());
        AllocatorAttributes attr;
        attr.set_gpu_compatible(true);
        result->output.emplace_back(ctx->allocator(attr),
                                    return_values->at(i).dtype(),
                                    component_shape);
        if (!result->output.back().IsInitialized()) {
          return errors::ResourceExhausted(
              "Failed to allocate memory for the batch of component ", i);
        }
      }
      RecordBufferEnqueue(ctx.get(), result->output);
      result->output_allocated = true;
      return absl::OkStatus();
    }

    void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
        TF_LOCKS_EXCLUDED(*mu_) {
      std::vector<std::pair<std::shared_ptr<BatchResult>, int64_t>> new_calls;
      RecordStart(ctx.get());
      auto stop_cleanup =
          gtl::MakeCleanup([this, &ctx]() { RecordStop(ctx.get()); });
      {
        tf_shared_lock l(*mu_);  // mu_ == num_parallel_calls_->mu
        new_calls.reserve(num_parallel_calls_->value);
      }
      auto busy = [this]() TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
        int64_t num_parallel_calls = num_parallel_calls_->value;
        return num_calls_ >= num_parallel_calls ||
               (batch_results_.size() > max_batch_results_ ||
                (batch_results_.size() == max_batch_results_ &&
                 call_counter_ % dataset()->batch_size_ == 0));
      };
      while (true) {
        {
          mutex_lock l(*mu_);
          while (!cancelled_ && busy()) {
            if (waiting_ > 0 && num_calls_ < num_parallel_calls_->value &&
                max_batch_results_ < kMaxBatchResults) {
              // If there is a caller waiting for a batch and the number of
              // outstanding calls is not maxed out, it means we are out of
              // `batch_results_` slots. Instead of waiting for a slot to open
              // up, we create a new one to utilize CPU efficiently.
              max_batch_results_++;
              continue;
            }
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
          }

          if (cancelled_) {
            return;
          }

          while (!busy()) {
            if (call_counter_ % dataset()->batch_size_ == 0) {
              batch_results_.push_back(std::make_shared<BatchResult>(
                  dataset()->batch_size_, ctx.get()));
            }
            int64_t offset = call_counter_++ % dataset()->batch_size_;
            new_calls.emplace_back(batch_results_.back(), offset);
            num_calls_++;
          }
        }
        const auto& stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator) {
          mutex_lock l(*mu_);
          stats_aggregator->AddScalar(
              stats_utils::ThreadUtilizationScalarName(dataset()->node_name()),
              static_cast<float>(num_calls_) /
                  static_cast<float>(num_parallel_calls_->value),
              num_elements());
        }
        for (const auto& call : new_calls) {
          CallFunction(ctx, call.first, call.second);
        }
        new_calls.clear();
      }
    }

    Status ReadBatchResult(IteratorContext* ctx, IteratorStateReader* reader,
                           size_t index) TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      batch_results_.push_back(
          std::make_shared<BatchResult>(dataset()->batch_size_, ctx));
      std::shared_ptr<BatchResult> result = batch_results_.back();
      string batch_prefix = strings::StrCat(kBatchResults, "_", index);
      mutex_lock l(result->mu);
      result->end_of_input = reader->Contains(
          prefix(), strings::StrCat(batch_prefix, "_", kEndOfInput));
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          prefix(), strings::StrCat(batch_prefix, "_", kNumCalls),
          &result->num_calls));
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          prefix(), strings::StrCat(batch_prefix, "_", kNumElements),
          &result->num_elements));
      result->output_allocated = reader->Contains(
          prefix(), strings::StrCat(batch_prefix, "_", kOutputAllocated));

      TF_RETURN_IF_ERROR(ReadBatch(ctx, reader, dataset()->batch_size_,
                                   prefix(), batch_prefix, &result->output));
      TF_RETURN_IF_ERROR(ReadStatus(prefix(),
                                    strings::StrCat(batch_prefix, "_", kStatus),
                                    reader, &result->status));
      if (result->output_allocated) {
        RecordBufferEnqueue(ctx, result->output);
      }
      return absl::OkStatus();
    }

    Status WriteBatchResult(IteratorStateWriter* writer, size_t index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      std::shared_ptr<BatchResult> result = batch_results_[index];
      string batch_prefix = strings::StrCat(kBatchResults, "_", index);
      mutex_lock l(result->mu);
      if (result->end_of_input) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            prefix(), strings::StrCat(batch_prefix, "_", kEndOfInput), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), strings::StrCat(batch_prefix, "_", kNumCalls),
          result->num_calls));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), strings::StrCat(batch_prefix, "_", kNumElements),
          result->num_elements));
      if (result->output_allocated) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            prefix(), strings::StrCat(batch_prefix, "_", kOutputAllocated),
            ""));
      }

      TF_RETURN_IF_ERROR(WriteBatch(dataset()->batch_size_,
                                    result->num_elements, prefix(),
                                    batch_prefix, writer, &result->output));
      TF_RETURN_IF_ERROR(
          WriteStatus(prefix(), strings::StrCat(batch_prefix, "_", kStatus),
                      result->status, writer));
      return absl::OkStatus();
    }

    // Used for coordination between the main thread, the runner thread, and
    // the callback threads.
    const std::shared_ptr<mutex> mu_;
    // Used for coordination between the main thread, the runner thread, and
    // the callback threads. In particular, the runner thread should only
    // schedule new calls when the number of in-flight calls is less than
    // `num_parallel_calls_->value` and there are slots available in the
    // `batch_results_` buffer.
    const std::shared_ptr<condition_variable> cond_var_;
    // Identifies the maximum number of parallel calls.
    const std::shared_ptr<model::SharedState> num_parallel_calls_;

    // Controls cancellation of `input_impl_`. Must be ordered before
    // `input_impl_` so that `input_impl_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;
    // Counts the number of outstanding calls for this batch.
    int64_t num_calls_ TF_GUARDED_BY(*mu_) = 0;
    // Counts the total number of calls.
    int64_t call_counter_ TF_GUARDED_BY(*mu_) = 0;
    std::unique_ptr<IteratorBase> input_impl_;
    // Buffer for storing the (intermediate) batch results. Whenever an
    // output-allocated batch result is added to or removed from
    // `batch_results_`, call `RecordBufferEnqueue` or `RecordBufferDequeue`
    // respectively.
    std::deque<std::shared_ptr<BatchResult>> batch_results_ TF_GUARDED_BY(*mu_);
    // Determines whether the transformation has been cancelled.
    bool cancelled_ TF_GUARDED_BY(*mu_) = false;
    // Identifies the number of callers currently waiting for a batch result.
    int64_t waiting_ TF_GUARDED_BY(*mu_) = 0;
    // Identifies the maximum number of batch results to store.
    int64_t max_batch_results_ TF_GUARDED_BY(*mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;

    // Records the number of ParallelInterleave operations in the path from the
    // root node to this node (not including this node) in the input pipeline
    // tree. We record the interleave depth so that it can be included in the
    // trace metadata.
    int64 interleave_depth_ = -1;
    // Background thread used for coordinating input processing. The thread
    // should be destroyed before the variables it accesses are destroyed.
    std::unique_ptr<Thread> runner_thread_ TF_GUARDED_BY(*mu_);
  };

  const DatasetBase* const input_;
  const int64_t batch_size_;
  const int64_t num_parallel_calls_;
  const bool drop_remainder_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const bool preserve_cardinality_;
  const TraceMeMetadata traceme_metadata_;
};

MapAndBatchDatasetOp::MapAndBatchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kFunc, /*params=*/{},
                                               &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
}

void MapAndBatchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                       DatasetBase** output) {
  int64_t batch_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBatchSize, &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("batch_size must be greater than zero."));

  int64_t num_parallel_calls = 0;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  OP_REQUIRES(
      ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
      errors::InvalidArgument("num_parallel_calls must be greater than zero."));

  bool drop_remainder;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument(ctx, kDropRemainder, &drop_remainder));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  if (num_parallel_calls == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output = new Dataset(ctx, input, batch_size, num_parallel_calls,
                        drop_remainder, output_types_, output_shapes_,
                        std::move(captured_func), preserve_cardinality_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("MapAndBatchDataset").Device(DEVICE_CPU),
                        MapAndBatchDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalMapAndBatchDataset").Device(DEVICE_CPU),
    MapAndBatchDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("MapAndBatchDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalMapAndBatchDataset");
}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
