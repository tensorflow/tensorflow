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
#include <utility>

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
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/tracing.h"

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
constexpr int64 kMaxBatchResults = 16;
constexpr char kParallelism[] = "parallelism";
constexpr char kCallCounter[] = "call_counter";
constexpr char kBatchResultsSize[] = "batch_results_size";
constexpr char kTFDataMapAndBatch[] = "tf_data_map_and_batch";
constexpr char kBatchResults[] = "batch_results";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kNumCalls[] = "num_calls";
constexpr char kNumElements[] = "num_elements";
constexpr char kOutputAllocated[] = "output_allocated";
constexpr char kOutputSize[] = "output_size";
constexpr char kOutput[] = "output";
constexpr char kStatus[] = "status";
constexpr char kCode[] = "code";
constexpr char kMessage[] = "msg";

// Period between reporting dataset statistics.
constexpr int kStatsReportingPeriodMillis = 1000;

class MapAndBatchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 batch_size,
          int64 num_parallel_calls, bool drop_remainder,
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
        preserve_cardinality_(preserve_cardinality) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64 Cardinality() const override {
    int64 n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / batch_size_ + (n % batch_size_ == 0 || drop_remainder_ ? 0 : 1);
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
    return Status::OK();
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
          max_batch_results_(
              params.dataset->num_parallel_calls_ == model::kAutotune
                  ? kMaxBatchResults
                  : std::min(kMaxBatchResults,
                             (params.dataset->num_parallel_calls_ +
                              params.dataset->batch_size_ - 1) /
                                 params.dataset->batch_size_)) {}

    ~Iterator() override {
      CancelThreads(/*wait=*/true);
      if (deregister_fn_) deregister_fn_();
    }

    string BuildTraceMeName() override {
      int64 parallelism = -1;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        parallelism = num_parallel_calls_->value;
        mu_->unlock();
      }
      return strings::StrCat(
          prefix(), "#parallelism=", parallelism,
          ",autotune=", dataset()->num_parallel_calls_ == model::kAutotune,
          ",batch_size=", dataset()->batch_size_,
          ",drop_remainder=", dataset()->drop_remainder_, "#");
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
          dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
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
      return ProcessResult(ctx, result, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(
          std::move(args), dataset()->batch_size_,
          {model::MakeParameter(kParallelism, num_parallel_calls_, /*min=*/1,
                                /*max=*/ctx->runner_threadpool_size())});
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(*mu_);
      // Wait for all in-flight calls to complete.
      while (num_calls_ > 0) {
        cond_var_->wait(l);
      }
      DCHECK_EQ(num_calls_, 0);
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCallCounter), call_counter_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kBatchResultsSize),
                                             batch_results_.size()));
      for (size_t i = 0; i < batch_results_.size(); ++i) {
        TF_RETURN_IF_ERROR(WriteBatchResult(writer, i));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCallCounter), &call_counter_));
      int64 batch_results_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kBatchResultsSize),
                                            &batch_results_size));
      for (int i = 0; i < batch_results_size; ++i) {
        TF_RETURN_IF_ERROR(ReadBatchResult(ctx, reader, i));
      }
      return Status::OK();
    }

   private:
    // BatchResult encapsulates the output batch, as well as ancillary
    // metadata required to execute the fused map-and-batch operation.
    struct BatchResult {
      explicit BatchResult(int64 batch_size) {
        end_of_input = false;
        num_calls = batch_size;
        num_elements = 0;
        output_allocated = false;
        status = Status::OK();
        status_offset = -1;
      }

      // UpdateStatus updates the batch's aggregate Status.
      //
      // In order to ensure that exactly the first non-OK status is returned
      // (required to make the behavior is observably identical to a
      // sequential execution of map followed by batch), we must also keep
      // track of the offset into the batch that produced `s`.
      void UpdateStatus(const Status& s, int64 offset) {
        if (TF_PREDICT_FALSE(!s.ok())) {
          mutex_lock l(mu);
          if (status.ok() || offset < status_offset) {
            status = s;
            status_offset = offset;
          }
        }
      }

      mutex mu;
      bool end_of_input GUARDED_BY(mu);
      int64 num_elements GUARDED_BY(mu);
      std::vector<Tensor> output;
      bool output_allocated GUARDED_BY(mu);
      Status status GUARDED_BY(mu);
      int64 status_offset GUARDED_BY(mu);
      // Counts the number of outstanding calls for this batch.
      int64 num_calls;  // access guarded by owner's mutex
    };

    void CallCompleted(const std::shared_ptr<IteratorContext>& ctx,
                       const std::shared_ptr<BatchResult>& result)
        LOCKS_EXCLUDED(*mu_) {
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
                      const std::shared_ptr<BatchResult>& result, int64 offset)
        LOCKS_EXCLUDED(*mu_) {
      // Get the next input element.
      std::vector<Tensor> input_element;
      bool end_of_input = false;
      Status status =
          input_impl_->GetNext(ctx.get(), &input_element, &end_of_input);
      bool return_early;
      {
        mutex_lock l(result->mu);
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
              status.error_message());
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
                                            std::move(done), prefix());
    }

    void CancelThreads(bool wait) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(*mu_);
      cancelled_ = true;
      cond_var_->notify_all();
      // Wait for all in-flight calls to complete.
      while (wait && num_calls_ > 0) {
        cond_var_->wait(l);
      }
    }

    Status CopyPartialBatch(Tensor* output, const Tensor& value,
                            int64 num_elements) {
      switch (value.dtype()) {
#define HANDLE_TYPE(type)                                         \
  case DataTypeToEnum<type>::value: {                             \
    auto output_t = output->flat_outer_dims<type>();              \
    auto value_t = value.flat_outer_dims<type>();                 \
    for (size_t i = 0; i < num_elements; i++) {                   \
      output_t.template chip<0>(i) = value_t.template chip<0>(i); \
    }                                                             \
    return Status::OK();                                          \
  }
        TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
        default:
          return errors::InvalidArgument("Unsupported data type: ",
                                         DataTypeString(value.dtype()));
      }
      return Status::OK();
    }

    void EnsureThreadsStarted(IteratorContext* ctx)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!runner_thread_) {
        auto ctx_copy = std::make_shared<IteratorContext>(*ctx);
        runner_thread_ = ctx->StartThread(
            kTFDataMapAndBatch,
            std::bind(&Iterator::RunnerThread, this, ctx_copy));
        if (ctx->stats_aggregator()) {
          stats_thread_ = ctx->StartThread(
              "tf_data_map_and_batch_stats",
              std::bind(&Iterator::StatsThread, this, ctx_copy));
        }
      }
    }

    Status EnsureOutputAllocated(
        const std::shared_ptr<IteratorContext>& ctx,
        const std::shared_ptr<BatchResult>& result,
        const std::shared_ptr<std::vector<Tensor>>& return_values) {
      mutex_lock l(result->mu);
      if (result->output_allocated) {
        return Status::OK();
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
      result->output_allocated = true;
      return Status::OK();
    }

    Status ProcessResult(IteratorContext* ctx,
                         const std::shared_ptr<BatchResult>& result,
                         std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) {
      mutex_lock l(result->mu);
      if (result->num_elements == 0) {
        if (result->status.ok() || errors::IsOutOfRange(result->status)) {
          *end_of_sequence = true;
          return Status::OK();
        } else {
          *end_of_sequence = false;
          return result->status;
        }
      }
      if (!result->status.ok() && !errors::IsOutOfRange(result->status)) {
        // Deallocate tensors allocated for the output.
        result->output.clear();
        *end_of_sequence = false;
        return result->status;
      }
      if (result->num_elements < dataset()->batch_size_) {
        if (dataset()->drop_remainder_) {
          // Deallocate tensors allocated for the output.
          result->output.clear();
          *end_of_sequence = true;
          return Status::OK();
        }
        const std::vector<Tensor>& output = result->output;
        for (size_t i = 0; i < output.size(); ++i) {
          TensorShape component_shape(result->output[i].shape());
          component_shape.set_dim(0, result->num_elements);
          AllocatorAttributes attr;
          attr.set_gpu_compatible(true);
          out_tensors->emplace_back(ctx->allocator(attr), output[i].dtype(),
                                    component_shape);
          TF_RETURN_IF_ERROR(CopyPartialBatch(&out_tensors->back(), output[i],
                                              result->num_elements));
        }
        // Deallocate tensors allocated for the output.
        result->output.clear();
      } else {
        *out_tensors = std::move(result->output);
      }
      *end_of_sequence = false;
      return Status::OK();
    }

    void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
        LOCKS_EXCLUDED(*mu_) {
      std::vector<std::pair<std::shared_ptr<BatchResult>, int64>> new_calls;
      RecordStart(ctx.get());
      auto stop_cleanup =
          gtl::MakeCleanup([this, &ctx]() { RecordStop(ctx.get()); });
      {
        tf_shared_lock l(*mu_);  // mu_ == num_parallel_calls_->mu
        new_calls.reserve(num_parallel_calls_->value);
      }
      auto busy = [this]() EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
        int64 num_parallel_calls = num_parallel_calls_->value;
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
              batch_results_.push_back(
                  std::make_shared<BatchResult>(dataset()->batch_size_));
            }
            int64 offset = call_counter_++ % dataset()->batch_size_;
            new_calls.emplace_back(batch_results_.back(), offset);
            num_calls_++;
          }
        }
        for (const auto& call : new_calls) {
          CallFunction(ctx, call.first, call.second);
        }
        new_calls.clear();
      }
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
            stats_utils::ThreadUtilizationScalarName(dataset()->node_name()),
            static_cast<float>(num_calls) /
                static_cast<float>(num_parallel_calls),
            step);
      }
    }

    Status ReadBatchResult(IteratorContext* ctx, IteratorStateReader* reader,
                           size_t index) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      batch_results_.push_back(
          std::make_shared<BatchResult>(dataset()->batch_size_));
      std::shared_ptr<BatchResult> result = batch_results_.back();
      string prefix = strings::StrCat(kBatchResults, "_", index);
      mutex_lock l(result->mu);
      result->end_of_input = reader->Contains(
          full_name(strings::StrCat(prefix, "_", kEndOfInput)));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(strings::StrCat(prefix, "_", kNumCalls)),
                             &result->num_calls));
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(prefix, "_", kNumElements)),
          &result->num_elements));
      result->output_allocated = reader->Contains(
          full_name(strings::StrCat(prefix, "_", kOutputAllocated)));
      int64 output_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(prefix, "_", kOutputSize)), &output_size));
      result->output.reserve(output_size);
      for (int i = 0; i < output_size; i++) {
        Tensor t;
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            full_name(strings::StrCat(prefix, "_", kOutput, "_", i)), &t));
        // If the batch was not full, we may have stored only the relevant
        // slice. Since tensors in `BatchResult.output` are expected to
        // have the leading dimension of size batch_size, we build a larger
        // tensor and copy the slice read from the checkpoint into it.
        if (t.dim_size(0) < dataset()->batch_size_) {
          TensorShape component_shape(t.shape());
          component_shape.set_dim(0, dataset()->batch_size_);
          AllocatorAttributes attr;
          attr.set_gpu_compatible(true);
          Tensor new_t(ctx->allocator(attr), t.dtype(), component_shape);
          TF_RETURN_IF_ERROR(CopyPartialBatch(&new_t, t, t.dim_size(0)));
          result->output.emplace_back(std::move(new_t));
        } else {
          result->output.emplace_back(std::move(t));
        }
      }
      TF_RETURN_IF_ERROR(ReadStatus(
          reader, strings::StrCat(prefix, "_", kStatus), &result->status));
      return Status::OK();
    }

    Status ReadStatus(IteratorStateReader* reader, const string& prefix,
                      Status* status) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64 code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(prefix, "_", kCode)), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        tstring error_message;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(prefix, "_", kMessage)), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    Status WriteBatchResult(IteratorStateWriter* writer, size_t index)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      std::shared_ptr<BatchResult> result = batch_results_[index];
      string prefix = strings::StrCat(kBatchResults, "_", index);
      mutex_lock l(result->mu);
      if (result->end_of_input) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_", kEndOfInput)), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat(prefix, "_", kNumCalls)),
          result->num_calls));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat(prefix, "_", kNumElements)),
          result->num_elements));
      if (result->output_allocated) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_", kOutputAllocated)), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat(prefix, "_", kOutputSize)),
          result->output.size()));
      for (int i = 0; i < result->output.size(); i++) {
        // If the batch is not full, we only store the first `num_elements`
        // values. The rest of the batch tensor is *uninitialized* and
        // accessing that will raise msan errors.
        if (result->num_elements < dataset()->batch_size_) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(prefix, "_", kOutput, "_", i)),
              result->output[i].Slice(0, result->num_elements)));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(prefix, "_", kOutput, "_", i)),
              result->output[i]));
        }
      }
      TF_RETURN_IF_ERROR(WriteStatus(
          writer, strings::StrCat(prefix, "_", kStatus), result->status));
      return Status::OK();
    }

    Status WriteStatus(IteratorStateWriter* writer, const string& prefix,
                       const Status& status) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(strings::StrCat(prefix, "_", kCode)),
                              static_cast<int64>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(prefix, "_", kMessage)),
            status.error_message()));
      }
      return Status::OK();
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

    // Counts the number of outstanding calls for this batch.
    int64 num_calls_ GUARDED_BY(*mu_) = 0;
    // Counts the total number of calls.
    int64 call_counter_ GUARDED_BY(*mu_) = 0;
    std::unique_ptr<IteratorBase> input_impl_;
    // Buffer for storing the (intermediate) batch results.
    std::deque<std::shared_ptr<BatchResult>> batch_results_ GUARDED_BY(*mu_);
    // Background thread used for coordinating input processing.
    std::unique_ptr<Thread> runner_thread_ GUARDED_BY(*mu_);
    std::unique_ptr<Thread> stats_thread_ GUARDED_BY(*mu_);
    // Determines whether the transformation has been cancelled.
    bool cancelled_ GUARDED_BY(*mu_) = false;
    // Identifies the number of callers currently waiting for a batch result.
    int64 waiting_ GUARDED_BY(*mu_) = 0;
    // Identifies the maximum number of batch results to store.
    int64 max_batch_results_ GUARDED_BY(*mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;
  };

  const DatasetBase* const input_;
  const int64 batch_size_;
  const int64 num_parallel_calls_;
  const bool drop_remainder_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const bool preserve_cardinality_;
};

MapAndBatchDatasetOp::MapAndBatchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  FunctionMetadata::Params params;
  params.is_multi_device_function = true;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
}

void MapAndBatchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                       DatasetBase** output) {
  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBatchSize, &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("batch_size must be greater than zero."));

  int64 num_parallel_calls = 0;
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
