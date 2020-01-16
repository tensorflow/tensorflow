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
#include "tensorflow/core/kernels/data/prefetch_dataset_op.h"

#include <deque>

#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const PrefetchDatasetOp::kDatasetType;
/* static */ constexpr const char* const PrefetchDatasetOp::kInputDataset;
/* static */ constexpr const char* const PrefetchDatasetOp::kBufferSize;
/* static */ constexpr const char* const PrefetchDatasetOp::kOutputTypes;
/* static */ constexpr const char* const PrefetchDatasetOp::kOutputShapes;
/* static */ constexpr const char* const PrefetchDatasetOp::kSlackPeriod;
/* static */ constexpr const char* const PrefetchDatasetOp::kLegacyAutotune;

// Determines the fraction of slack time by which to delay prefetching of data.
constexpr double kSleepFactor = 0.2;
constexpr char kBuffer[] = "buffer";
constexpr char kStatus[] = "status";
constexpr char kSizeSuffix[] = ".size";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessageSuffix[] = ".error_message";

class PrefetchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 buffer_size,
          int64 slack_period, bool legacy_autotune)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        slack_period_(slack_period),
        legacy_autotune_(legacy_autotune) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    AttrValue slack_period_attr;
    b->BuildAttrValue(slack_period_, &slack_period_attr);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size},
        {std::make_pair(kSlackPeriod, slack_period_attr)}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          cond_var_(std::make_shared<condition_variable>()),
          auto_tuner_(params.dataset->buffer_size_),
          legacy_autotune_(params.dataset->legacy_autotune_),
          buffer_size_(std::make_shared<model::SharedState>(
              legacy_autotune_ ? 0 : params.dataset->buffer_size_, mu_,
              cond_var_)) {
      slack_us_ = 0;
    }

    ~Iterator() override {
      CancelThreads();
      if (deregister_fn_) deregister_fn_();
    }

    string BuildTraceMeName() override {
      int64 limit = -1;
      // NOTE: We only set the buffer limit value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        limit = buffer_limit();
        mu_->unlock();
      }
      string prefetch_with_slack_trace = "";
      if (dataset()->slack_period_ > 0) {
        int64 slack_us = slack_us_;
        prefetch_with_slack_trace = strings::StrCat(",slack=", slack_us);
      }
      return strings::StrCat(prefix(), "#buffer_limit=", limit,
                             prefetch_with_slack_trace, "#");
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);
      if (buffer_size_->value == model::kAutotune) {
        buffer_size_->value = 0;
      }
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(), [this]() { CancelThreads(); },
          &deregister_fn_));
      return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      const auto& stats_aggregator = ctx->stats_aggregator();
      {
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(EnsurePrefetchThreadStarted(ctx));
        // Wait until the next element in the buffer has been
        // produced, or we are shutting down.
        if (legacy_autotune_) {
          while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_ &&
                 auto_tuner_.buffer_limit() != 0) {
            auto_tuner_.RecordEmpty();
            buffer_size_->value = auto_tuner_.buffer_limit();
            RecordStop(ctx);
            cond_var_->wait(l);
            RecordStart(ctx);
          }
        } else {
          while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_ &&
                 buffer_size_->value != 0) {
            RecordStop(ctx);
            cond_var_->wait(l);
            RecordStart(ctx);
          }
        }

        if (cancelled_) {
          return errors::Cancelled("Iterator was cancelled");
        }

        if (!buffer_.empty()) {
          return Consume(ctx, out_tensors, end_of_sequence);
        }

        if (prefetch_thread_finished_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        DCHECK_EQ(buffer_limit(), 0);
      }

      mutex_lock input_l(input_mu_);
      {
        mutex_lock l(*mu_);
        if (stats_aggregator) {
          stats_aggregator->AddScalar(
              stats_utils::BufferSizeScalarName(dataset()->node_name()),
              static_cast<float>(buffer_.size()), num_elements());
          stats_aggregator->AddScalar(
              stats_utils::BufferCapacityScalarName(dataset()->node_name()),
              static_cast<float>(buffer_limit()), num_elements());
        }
        // Release mu_
      }
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(
          std::move(args),
          /*ratio=*/1,
          {model::MakeParameter(kBufferSize, buffer_size_, /*min=*/0,
                                /*max=*/std::numeric_limits<int64>::max())});
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      // Acquire both locks to ensure that the prefetch thread and
      // all GetNext threads are blocked.
      mutex_lock input_l(input_mu_);
      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kBufferSize, buffer_.size()));
      for (size_t i = 0; i < buffer_.size(); i++) {
        auto& buffer_element = buffer_[i];
        TF_RETURN_IF_ERROR(WriteStatus(writer, i, buffer_element.status));
        if (buffer_element.status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              absl::StrCat(prefix(), "::", i),
              absl::StrCat(kBuffer, kSizeSuffix), buffer_element.value.size()));
          for (size_t j = 0; j < buffer_element.value.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                absl::StrCat(prefix(), "::", i),
                absl::StrCat(kBuffer, "[", j, "]"), buffer_element.value[j]));
          }
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock input_l(input_mu_);
      mutex_lock l(*mu_);
      buffer_.clear();
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      size_t buffer_size;
      {
        int64 temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kBufferSize, &temp));
        buffer_size = static_cast<size_t>(temp);
      }
      for (size_t i = 0; i < buffer_size; i++) {
        buffer_.emplace_back();
        auto& buffer_element = buffer_.back();
        TF_RETURN_IF_ERROR(ReadStatus(reader, i, &buffer_element.status));
        if (buffer_element.status.ok()) {
          size_t value_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(absl::StrCat(prefix(), "::", i),
                                   absl::StrCat(kBuffer, kSizeSuffix), &temp));
            value_size = static_cast<size_t>(temp);
          }
          buffer_element.value.reserve(value_size);
          for (size_t j = 0; j < value_size; j++) {
            buffer_element.value.emplace_back();
            TF_RETURN_IF_ERROR(
                reader->ReadTensor(absl::StrCat(prefix(), "::", i),
                                   absl::StrCat(kBuffer, "[", j, "]"),
                                   &buffer_element.value.back()));
          }
        }
      }
      return Status::OK();
    }

   private:
    // A buffer element comprises a status and (if that status is
    // OK) a vector of tensors, representing an element of the input dataset.
    struct BufferElement {
      // The producer sets `status` if getting the input element fails.
      Status status;
      // The buffered data element.
      std::vector<Tensor> value;
      int64 created_us;
    };

    inline int64 buffer_limit() EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (legacy_autotune_) {
        return auto_tuner_.buffer_limit();
      }
      return buffer_size_->value;
    }

    void CancelThreads() LOCKS_EXCLUDED(mu_) {
      mutex_lock l(*mu_);
      cancelled_ = true;
      cond_var_->notify_all();
    }

    Status Consume(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                   bool* end_of_sequence) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      const auto& stats_aggregator = ctx->stats_aggregator();
      if (stats_aggregator) {
        double buffer_limit_ = buffer_limit();
        stats_aggregator->AddToHistogram(
            stats_utils::BufferUtilizationHistogramName(dataset()->node_name()),
            {static_cast<float>(buffer_.size()) /
             static_cast<float>(buffer_limit_)},
            num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferSizeScalarName(dataset()->node_name()),
            static_cast<float>(buffer_.size()), num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferCapacityScalarName(dataset()->node_name()),
            static_cast<float>(buffer_limit_), num_elements());
      }
      // A new element is available. Forward the status from computing it, and
      // (if we successfully got an element) the output values.
      Status s = buffer_.front().status;
      if (s.ok()) {
        if (dataset()->slack_period_ > 0 &&
            (num_elements() + 1) % dataset()->slack_period_ == 0) {
          // TODO(rachelim): Consider doing something more sophisticated
          // to decide how long to sleep for; e.g. using a kalman filter.
          int64 slack_us = EnvTime::NowMicros() - buffer_.front().created_us;
          // Every slack_period_-th element, update the most recent slack time,
          // measured by the duration between when the element is prefetched
          // and when it is consumed. We add kSleepFactor * slack_us_ to the
          // measurement because we slept for that duration before prefetching
          // the element.
          slack_us_ = kSleepFactor * slack_us_ + slack_us;
          VLOG(2) << "Setting slack_us_: " << slack_us_;
        }
        *out_tensors = std::move(buffer_.front().value);
        RecordBufferDequeue(ctx, *out_tensors);
      }
      if (legacy_autotune_) {
        auto_tuner_.RecordConsumption(buffer_.size());
        buffer_size_->value = auto_tuner_.buffer_limit();
      }
      buffer_.pop_front();
      *end_of_sequence = false;

      // Wake the prefetch thread, in case it has been waiting for space
      // in the buffer. Also wake up threads from other calls to GetNext.
      //
      // TODO(mrry): Consider using different condition variables for
      // GetNext and Prefetch.
      cond_var_->notify_all();
      return s;
    }

    Status EnsurePrefetchThreadStarted(IteratorContext* ctx)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!prefetch_thread_) {
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        prefetch_thread_ = ctx->StartThread(
            "tf_data_prefetch", [this, new_ctx]() { PrefetchThread(new_ctx); });
      }
      return Status::OK();
    }

    // Prefetches elements of the input, storing results in an internal buffer.
    //
    // It owns the iterator context passed to it.
    void PrefetchThread(const std::shared_ptr<IteratorContext>& ctx) {
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      // Keep track of where we are in an iteration "burst"
      int num_produced = 0;
      while (true) {
        // 1. Wait for a slot in the buffer.
        {
          mutex_lock l(*mu_);
          while (!cancelled_ && buffer_.size() >= buffer_limit()) {
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
          }

          if (cancelled_) {
            prefetch_thread_finished_ = true;
            cond_var_->notify_all();
            return;
          }
        }

        if (dataset()->slack_period_ > 0 &&
            num_produced % dataset()->slack_period_ == 0) {
          // For the first element in the "burst", sleep for a bit if there is
          // slack.
          VLOG(2) << "Sleeping for: " << slack_us_ * kSleepFactor;
          ctx->env()->SleepForMicroseconds(slack_us_ * kSleepFactor);
        }

        // 2. Read the next element.
        // Acquire the input mutex since we will be reading an element from the
        // input iterator. Note that we do not wish to release this mutex till
        // we have added the fetched element to the `buffer_` else there will be
        // local state that may be missed by SaveInternal.
        mutex_lock input_l(input_mu_);
        bool end_of_sequence;
        BufferElement buffer_element;
        buffer_element.status = input_impl_->GetNext(
            ctx.get(), &buffer_element.value, &end_of_sequence);
        if (buffer_element.status.ok() && end_of_sequence) {
          mutex_lock l(*mu_);
          prefetch_thread_finished_ = true;
          cond_var_->notify_all();
          return;
        }

        // 3. Signal that the element has been produced.
        {
          mutex_lock l(*mu_);
          RecordBufferEnqueue(ctx.get(), buffer_element.value);
          buffer_element.created_us = EnvTime::NowMicros();
          buffer_.push_back(std::move(buffer_element));
          cond_var_->notify_all();
        }
        ++num_produced;
      }
    }

    Status WriteStatus(IteratorStateWriter* writer, size_t index,
                       const Status& status) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(absl::StrCat(prefix(), "::", index), CodeKey(),
                              static_cast<int64>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(absl::StrCat(prefix(), "::", index),
                                ErrorMessageKey(), status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatus(IteratorStateReader* reader, size_t index, Status* status)
        EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64 code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(absl::StrCat(prefix(), "::", index),
                                            CodeKey(), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        tstring error_message;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(absl::StrCat(prefix(), "::", index),
                               ErrorMessageKey(), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    string CodeKey() { return absl::StrCat(kStatus, kCodeSuffix); }

    string ErrorMessageKey() {
      return absl::StrCat(kStatus, kErrorMessageSuffix);
    }

    // This mutex is used to ensure exclusivity between multiple threads
    // reading/writing this iterator's local state.
    //
    // NOTE: We should never call GetNext on the input while holding this mutex.
    const std::shared_ptr<mutex> mu_;
    // This mutex is used to ensure exclusivity between multiple threads
    // accessing the input iterator. We keep this separate from `mu_` to allow
    // prefetching to run in parallel with GetNext calls.
    mutex input_mu_ ACQUIRED_BEFORE(*mu_);
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(input_mu_);
    const std::shared_ptr<condition_variable> cond_var_;
    PrefetchAutotuner auto_tuner_ GUARDED_BY(*mu_);
    std::deque<BufferElement> buffer_ GUARDED_BY(*mu_);
    std::unique_ptr<Thread> prefetch_thread_ GUARDED_BY(*mu_);
    bool cancelled_ GUARDED_BY(*mu_) = false;
    bool prefetch_thread_finished_ GUARDED_BY(*mu_) = false;
    const bool legacy_autotune_;

    std::atomic<int64> slack_us_;

    // If legacy_autotune_ is false, identifies the maximum size of the buffer.
    const std::shared_ptr<model::SharedState> buffer_size_;

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;
  };
  const DatasetBase* const input_;
  const int64 buffer_size_;

  // If non-zero, determines the period between injecting "slack" into the
  // execution.
  const int64 slack_period_;

  // Determines whether legacy autotuning should be used.
  const bool legacy_autotune_ = true;
};

PrefetchDatasetOp::PrefetchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  if (ctx->HasAttr(kSlackPeriod)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kSlackPeriod, &slack_period_));
  }
  if (ctx->HasAttr(kLegacyAutotune)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kLegacyAutotune, &legacy_autotune_));
  }
}

void PrefetchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  int64 buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx, buffer_size >= 0 || buffer_size == model::kAutotune,
              errors::InvalidArgument("buffer_size must be >= 0 or set "
                                      "buffer_size to be ",
                                      model::kAutotune, " for auto-tuning"));

  if (buffer_size == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output =
      new Dataset(ctx, input, buffer_size, slack_period_, legacy_autotune_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("PrefetchDataset").Device(DEVICE_CPU).Priority(2),
                        PrefetchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("PrefetchDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("buffer_size")
                            .HostMemory("input_dataset")
                            .HostMemory("handle")
                            .Priority(1),
                        PrefetchDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
