/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/data_service_dataset_op.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "xla/tsl/framework/allocator.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/data/service/client/data_service_client.h"
#include "tensorflow/core/data/service/client/utils.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/data/parallel_map_dataset_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const DataServiceDatasetOp::kDatasetType;
/* static */ constexpr const char* const DataServiceDatasetOp::kDatasetId;
/* static */ constexpr const char* const DataServiceDatasetOp::kProcessingMode;
/* static */ constexpr const char* const DataServiceDatasetOp::kAddress;
/* static */ constexpr const char* const DataServiceDatasetOp::kProtocol;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kDataTransferProtocol;
/* static */ constexpr const char* const DataServiceDatasetOp::kJobName;
/* static */ constexpr const char* const DataServiceDatasetOp::kConsumerIndex;
/* static */ constexpr const char* const DataServiceDatasetOp::kNumConsumers;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kMaxOutstandingRequests;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kTaskRefreshIntervalHintMs;
/* static */ constexpr const char* const DataServiceDatasetOp::kTargetWorkers;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kIterationCounter;
/* static */ constexpr const char* const DataServiceDatasetOp::kOutputTypes;
/* static */ constexpr const char* const DataServiceDatasetOp::kOutputShapes;
/* static */ constexpr const char* const DataServiceDatasetOp::kUncompress;
/* static */ constexpr const char* const DataServiceDatasetOp::kUncompressFn;
/* static */ constexpr const char* const
    DataServiceDatasetOp::kCrossTrainerCacheOptions;

namespace {
constexpr char kDataServiceDatasetV1[] = "DataServiceDataset";
constexpr char kDataServiceDatasetV2[] = "DataServiceDatasetV2";
constexpr char kDataServiceDatasetV3[] = "DataServiceDatasetV3";
constexpr char kDataServiceDatasetV4[] = "DataServiceDatasetV4";

constexpr const char kParallelEpochs[] = "parallel_epochs";
constexpr const char kDistributedEpoch[] = "distributed_epoch";

// Default interval between task list refreshes.
constexpr absl::Duration kDefaultTaskRefreshInterval = absl::Seconds(1);

// Default starting `max_outstanding_requests` when it is autotuned.
constexpr int64_t kStartingMaxOutstandingRequests = 16;

}  // namespace

// Dataset for reading data from the tf.data service.
class DataServiceDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(
      OpKernelContext* ctx, int op_version, const std::string& dataset_id,
      const ProcessingModeDef& processing_mode, const std::string& address,
      const std::string& protocol, const std::string& data_transfer_protocol,
      const std::string& job_name, std::optional<int64_t> consumer_index,
      std::optional<int64_t> num_consumers, int64_t max_outstanding_requests,
      absl::Duration task_refresh_interval, const TargetWorkers target_workers,
      const DataServiceMetadata& metadata, IterationCounter* iteration_counter,
      bool owns_resource, ResourceHandle iteration_counter_handle,
      std::unique_ptr<CapturedFunction> captured_uncompress_func,
      const std::optional<CrossTrainerCacheOptions>&
          cross_trainer_cache_options,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        op_version_(op_version),
        dataset_id_(dataset_id),
        processing_mode_(processing_mode),
        address_(address),
        protocol_(protocol),
        data_transfer_protocol_(data_transfer_protocol),
        job_name_(job_name),
        is_coordinated_read_(consumer_index.has_value()),
        consumer_index_(consumer_index),
        num_consumers_(num_consumers),
        max_outstanding_requests_(max_outstanding_requests),
        task_refresh_interval_(task_refresh_interval),
        target_workers_(target_workers),
        metadata_(metadata),
        iteration_counter_(iteration_counter),
        owns_resource_(owns_resource),
        iteration_counter_handle_(iteration_counter_handle),
        resource_mgr_(ctx->resource_manager()),
        captured_uncompress_func_(std::move(captured_uncompress_func)),
        cross_trainer_cache_options_(cross_trainer_cache_options),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  ~Dataset() override {
    iteration_counter_->Unref();
    if (owns_resource_) {
      absl::Status s = resource_mgr_->Delete<IterationCounter>(
          iteration_counter_handle_.container(),
          iteration_counter_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete iteration counter resource: " << s;
      }
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(
        Iterator::Params{this,
                         name_utils::IteratorPrefix(kDatasetType, prefix)},
        DataServiceParams{dataset_id_, processing_mode_, address_, protocol_,
                          data_transfer_protocol_, job_name_,
                          /*repetition=*/iteration_counter_->GetAndIncrement(),
                          num_consumers_, consumer_index_,
                          max_outstanding_requests_, task_refresh_interval_,
                          target_workers_, metadata_,
                          cross_trainer_cache_options_});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return EstimateCardinality(processing_mode_, metadata_,
                               is_coordinated_read_);
  }

  absl::Status CheckExternalState() const override {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        strings::StrCat(DebugString(), " does not yet support serialization."));
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    return absl::OkStatus();
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    // Inputs
    std::vector<Node*> inputs;

    if (op_version_ >= 4) {
      Node* dataset_id;
      TF_RETURN_IF_ERROR(b->AddScalar(dataset_id_, &dataset_id));
      inputs.push_back(dataset_id);
    } else {
      int64_t dataset_id_int;
      if (!absl::SimpleAtoi(dataset_id_, &dataset_id_int)) {
        return errors::Internal("Failed to parse dataset ID: ", dataset_id_,
                                ". Expect integers.");
      }
      Node* dataset_id;
      TF_RETURN_IF_ERROR(b->AddScalar(dataset_id_int, &dataset_id));
      inputs.push_back(dataset_id);
    }

    Node* processing_mode;
    tstring processing_mode_str = processing_mode_.SerializeAsString();
    TF_RETURN_IF_ERROR(b->AddScalar(processing_mode_str, &processing_mode));
    inputs.push_back(processing_mode);

    Node* address;
    TF_RETURN_IF_ERROR(b->AddScalar(address_, &address));
    inputs.push_back(address);

    Node* protocol;
    TF_RETURN_IF_ERROR(b->AddScalar(protocol_, &protocol));
    inputs.push_back(protocol);

    Node* job_name;
    TF_RETURN_IF_ERROR(b->AddScalar(job_name_, &job_name));
    inputs.push_back(job_name);

    if (op_version_ >= 2) {
      Node* consumer_index;
      TF_RETURN_IF_ERROR(
          b->AddScalar(consumer_index_.value_or(-1), &consumer_index));
      inputs.push_back(consumer_index);

      Node* num_consumers;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_consumers_.value_or(-1), &num_consumers));
      inputs.push_back(num_consumers);
    }

    Node* max_outstanding_requests;
    TF_RETURN_IF_ERROR(
        b->AddScalar(max_outstanding_requests_, &max_outstanding_requests));
    inputs.push_back(max_outstanding_requests);

    Node* iteration_counter_handle = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = iteration_counter_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &iteration_counter_handle));
    inputs.push_back(iteration_counter_handle);

    // Attributes
    std::vector<std::pair<StringPiece, AttrValue>> attrs;
    AttrValue task_refresh_interval_hint_ms;
    b->BuildAttrValue(absl::ToInt64Milliseconds(task_refresh_interval_),
                      &task_refresh_interval_hint_ms);
    attrs.push_back(
        {kTaskRefreshIntervalHintMs, task_refresh_interval_hint_ms});

    AttrValue data_transfer_protocol;
    b->BuildAttrValue(data_transfer_protocol_, &data_transfer_protocol);
    attrs.push_back({kDataTransferProtocol, data_transfer_protocol});

    AttrValue target_workers;
    b->BuildAttrValue(TargetWorkersToString(target_workers_), &target_workers);
    attrs.push_back({kTargetWorkers, target_workers});

    if (op_version_ >= 3) {
      // Attr: uncompress is true for the first time the graph is built, when a
      // ParallelMap dataset is inserted for uncompression. Subsequent
      // serialization always sets it to false to avoid inserting repeated map
      // datasets for uncompression.
      AttrValue uncompress_attr;
      b->BuildAttrValue(false, &uncompress_attr);
      attrs.push_back({kUncompress, uncompress_attr});

      // Attr: uncompress_fn
      AttrValue uncompress_fn_attr;
      b->BuildAttrValue(captured_uncompress_func_->func(), &uncompress_fn_attr);
      attrs.push_back({kUncompressFn, uncompress_fn_attr});

      std::vector<Node*> uncompress_arguments;
      DataTypeVector uncompress_arguments_types;
      TF_RETURN_IF_ERROR(captured_uncompress_func_->AddToGraph(
          ctx, b, &uncompress_arguments, &uncompress_arguments_types));
    }

    // Attr: cross_trainer_cache_options
    AttrValue cross_trainer_cache_options_attr;
    std::string serialized_cross_trainer_cache_options;
    if (cross_trainer_cache_options_.has_value()) {
      serialized_cross_trainer_cache_options =
          cross_trainer_cache_options_->SerializeAsString();
    }
    b->BuildAttrValue(serialized_cross_trainer_cache_options,
                      &cross_trainer_cache_options_attr);
    attrs.push_back(
        {kCrossTrainerCacheOptions, cross_trainer_cache_options_attr});
    return b->AddDataset(this, inputs, attrs, output);
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params,
                      const DataServiceParams& data_service_params)
        : DatasetIterator<Dataset>(params),
          data_service_client_(data_service_params),
          buffer_size_(std::make_shared<model::SharedState>(
              // Give it a value of 1 if it is autotuned to make the parameter
              // not tunable by Autotune because it will be directly set by the
              // `data_service_client_` when number of tasks changes.
              params.dataset->max_outstanding_requests_ == model::kAutotune
                  ? 1
                  : params.dataset->max_outstanding_requests_,
              std::make_shared<mutex>(),
              std::make_shared<condition_variable>())) {}

    ~Iterator() override {
      data_service_client_.Cancel();
      if (deregister_fn_) {
        deregister_fn_();
      }
    }

    absl::Status Initialize(IteratorContext* ctx) override {
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(),
          [this]() { data_service_client_.Cancel(); }, &deregister_fn_));
      tsl::AllocatorAttributes attrs;
      if (ctx->options() != nullptr) {
        attrs.set_gpu_compatible(ctx->options()->service_options().pinned());
      }
      return data_service_client_.Initialize(ctx->accelerator_device_info(),
                                             ctx->allocator(attrs));
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      auto ctx_factory = [ctx, this]() {
        return std::make_unique<DataServiceIteratorContext>(
            ctx, this, buffer_size_, model_node());
      };
      TF_ASSIGN_OR_RETURN(GetNextResult result,
                          data_service_client_.GetNext(ctx_factory));
      *out_tensors = std::move(result.tensors);
      *end_of_sequence = result.end_of_sequence;
      return absl::OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(
          std::move(args),
          /*ratio=*/1,
          {model::MakeParameter(model::kBufferSize, buffer_size_,
                                /*min=*/1,
                                /*max=*/std::numeric_limits<double>::max())});
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal is not yet supported");
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      return errors::Unimplemented("RestoreInternal is not yet supported");
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return data_service_client_.GetTraceMeMetadata();
    }

   private:
    class DataServiceIteratorContext : public DataServiceContext {
     public:
      DataServiceIteratorContext(
          IteratorContext* ctx, Iterator* iterator,
          std::shared_ptr<model::SharedState> buffer_size,
          std::shared_ptr<model::Node> node)
          : ctx_(*ctx),
            iterator_(iterator),
            node_(node),
            buffer_size_(buffer_size) {}
      ~DataServiceIteratorContext() override = default;
      DataServiceIteratorContext(const DataServiceIteratorContext&) = delete;
      DataServiceIteratorContext& operator=(const DataServiceIteratorContext&) =
          delete;

      std::unique_ptr<Thread> StartThread(const string& name,
                                          std::function<void()> fn) override {
        return ctx_.StartThread(name, std::move(fn));
      }

      void RecordBufferEnqueue(const std::vector<Tensor>& element) override {
        iterator_->RecordBufferEnqueue(&ctx_, element);
      }

      void RecordBufferDequeue(const std::vector<Tensor>& element) override {
        iterator_->RecordBufferDequeue(&ctx_, element);
      }

      double GetTargetProcessingTimeNsec() const override {
        if (ctx_.model() == nullptr) {
          VLOG(1) << "tf.data Model is null in DataServiceIteratorContext";
          return 0.0;
        }

        double target_time_nsec =
            ctx_.model()->ComputeExperimentalTargetTimeNsec();
        if (target_time_nsec == 0.0) return 0.0;

        model::ModelTiming model_timing(ctx_.model()->output());
        const model::ModelTiming::NodeTiming* data_service_node_timing =
            model_timing.GetTiming(iterator_->model_node().get());

        return target_time_nsec / data_service_node_timing->pipeline_ratio;
      }

      // TODO(yangchen): Move this code to `DataServiceClient` and implement it
      // around `UpdateBufferSize()`.
      int64_t UpdateMaxOutstandingRequests(
          int64_t max_outstanding_requests,
          int64_t requested_outstanding_requests) override {
        if (node_ == nullptr ||
            max_outstanding_requests == requested_outstanding_requests) {
          return requested_outstanding_requests;
        }
        if (element_size_cache_ == 0.0) {
          element_size_cache_ = node_->AverageBufferedElementSize();
          VLOG(3) << "Average DataService element size is "
                  << element_size_cache_;
          if (element_size_cache_ == 0) {
            int64_t new_outstanding_requests = std::max(
                max_outstanding_requests, kStartingMaxOutstandingRequests);
            VLOG(3) << "The average element size of `DataService` is 0. The "
                       "`max_outstanding_requests` value "
                    << max_outstanding_requests
                    << (max_outstanding_requests == new_outstanding_requests
                            ? " is kept at "
                            : " is changed to the default value of ")
                    << new_outstanding_requests << ".";
            return new_outstanding_requests;
          }
        }
        const int64_t delta_outstanding_requests =
            ctx_.ram_budget_manager()->RequestModelBytes(
                requested_outstanding_requests -
                    std::max(static_cast<int64_t>(0), max_outstanding_requests),
                element_size_cache_);
        if (delta_outstanding_requests == 0) {
          VLOG(3) << "Request to change `max_outstanding_requests` from "
                  << max_outstanding_requests << " to "
                  << requested_outstanding_requests
                  << " failed due to low available memory. It is kept at "
                  << max_outstanding_requests;
          return max_outstanding_requests;
        }
        element_size_cache_ = node_->AverageBufferedElementSize();
        const int64_t new_outstanding_requests =
            max_outstanding_requests + delta_outstanding_requests;
        VLOG(3) << "The `max_outstanding_requests` changed from "
                << max_outstanding_requests << " to "
                << new_outstanding_requests << ". Requested value is "
                << requested_outstanding_requests;
        mutex_lock l(*buffer_size_->mu);
        buffer_size_->value = new_outstanding_requests;
        return new_outstanding_requests;
      }

     private:
      IteratorContext ctx_;
      Iterator* iterator_ = nullptr;
      const std::shared_ptr<model::Node> node_;
      const std::shared_ptr<model::SharedState> buffer_size_;
      double element_size_cache_ = 0.0;
    };

    DataServiceClient data_service_client_;
    const std::shared_ptr<model::SharedState> buffer_size_;
    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;
    friend class DataServiceIteratorContext;
  };

  const int op_version_;
  const tstring dataset_id_;
  const ProcessingModeDef processing_mode_;
  const tstring address_;
  const tstring protocol_;
  const tstring data_transfer_protocol_;
  const tstring job_name_;
  const bool is_coordinated_read_;
  const std::optional<int64_t> consumer_index_;
  const std::optional<int64_t> num_consumers_;
  const int64_t max_outstanding_requests_;
  const absl::Duration task_refresh_interval_;
  const TargetWorkers target_workers_;
  const DataServiceMetadata metadata_;
  IterationCounter* const iteration_counter_;  // Owned
  const bool owns_resource_;
  const ResourceHandle iteration_counter_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned
  const std::unique_ptr<CapturedFunction> captured_uncompress_func_;
  const std::optional<CrossTrainerCacheOptions> cross_trainer_cache_options_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

DataServiceDatasetOp::DataServiceDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  const auto& op_name = ctx->def().op();
  if (op_name == kDataServiceDatasetV1) {
    op_version_ = 1;
  } else if (op_name == kDataServiceDatasetV2) {
    op_version_ = 2;
  } else if (op_name == kDataServiceDatasetV3) {
    op_version_ = 3;
  } else if (op_name == kDataServiceDatasetV4) {
    op_version_ = 4;
  } else {
    ctx->CtxFailure(errors::FailedPrecondition(
        "Unrecognized data service dataset op name: ", op_name));
    return;
  }

  int64_t task_refresh_interval_hint_ms = 0;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kTaskRefreshIntervalHintMs,
                                   &task_refresh_interval_hint_ms));
  if (task_refresh_interval_hint_ms == model::kAutotune) {
    task_refresh_interval_hint_ = kDefaultTaskRefreshInterval;
  } else {
    task_refresh_interval_hint_ =
        absl::Milliseconds(task_refresh_interval_hint_ms);
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  if (ctx->HasAttr(kDataTransferProtocol)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kDataTransferProtocol, &data_transfer_protocol_));
  }

  std::string target_workers_str = "AUTO";
  if (ctx->HasAttr(kTargetWorkers)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kTargetWorkers, &target_workers_str));
  }
  absl::StatusOr<TargetWorkers> status_or_target_workers =
      ParseTargetWorkers(target_workers_str);
  OP_REQUIRES_OK(ctx, status_or_target_workers.status());
  target_workers_ = *status_or_target_workers;

  if (op_version_ >= 3) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kUncompress, &uncompress_));
    FunctionMetadata::Params params;
    params.use_inter_op_parallelism = true;
    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kUncompressFn, params,
                                                 &uncompress_fn_));
  }

  if (ctx->HasAttr(kCrossTrainerCacheOptions)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kCrossTrainerCacheOptions,
                                     &seriazlied_cross_trainer_cache_options_));
  }
}

void DataServiceDatasetOp::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  tstring dataset_id;
  if (op_version_ >= 4) {
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kDatasetId, &dataset_id));
  } else {
    int64_t dataset_id_int = 0;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kDatasetId, &dataset_id_int));
    dataset_id = absl::StrCat(dataset_id_int);
  }

  tstring processing_mode_str;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kProcessingMode, &processing_mode_str));
  ProcessingModeDef processing_mode;
  if (processing_mode_str == kParallelEpochs) {
    processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  } else if (processing_mode_str == kDistributedEpoch) {
    processing_mode.set_sharding_policy(ProcessingModeDef::DYNAMIC);
  } else {
    OP_REQUIRES(ctx, processing_mode.ParseFromString(processing_mode_str),
                errors::InvalidArgument(absl::Substitute(
                    "Failed to parse ProcessingModeDef from string: $0",
                    std::string(processing_mode_str))));
  }

  tstring address;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kAddress, &address));
  OP_REQUIRES(ctx, !address.empty(),
              errors::InvalidArgument(kAddress, " must be non-empty."));

  tstring protocol;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kProtocol, &protocol));
  OP_REQUIRES(ctx, !protocol.empty(),
              errors::InvalidArgument(kProtocol, " must be non-empty."));

  tstring job_name;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kJobName, &job_name));

  absl::StatusOr<DataServiceConfig> config =
      GetDataServiceConfig(address, protocol);
  OP_REQUIRES_OK(ctx, config.status());

  if (IsStaticShard(processing_mode) &&
      config->deployment_mode() == DEPLOYMENT_MODE_COLOCATED &&
      target_workers_ == TARGET_WORKERS_AUTO) {
    VLOG(1) << "Using LOCAL target workers for static sharding mode: "
            << processing_mode.ShortDebugString();
    target_workers_ = TARGET_WORKERS_LOCAL;
  }
  if (target_workers_ == TARGET_WORKERS_LOCAL) {
    data_transfer_protocol_ = kLocalTransferProtocol;
  }

  std::optional<int64_t> consumer_index;
  std::optional<int64_t> num_consumers;
  if (op_version_ >= 2) {
    int64_t consumer_index_int;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument(ctx, kConsumerIndex, &consumer_index_int));
    if (consumer_index_int >= 0) {
      consumer_index = consumer_index_int;
    }

    int64_t num_consumers_int;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, kNumConsumers, &num_consumers_int));
    if (num_consumers_int >= 0) {
      num_consumers = num_consumers_int;
    }
  }

  int64_t max_outstanding_requests;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kMaxOutstandingRequests,
                                          &max_outstanding_requests));

  ResourceHandle iteration_counter_handle;
  OP_REQUIRES_OK(
      ctx, HandleFromInput(ctx, kIterationCounter, &iteration_counter_handle));
  IterationCounter* iteration_counter = nullptr;
  absl::Status s = ctx->resource_manager()->Lookup<IterationCounter>(
      iteration_counter_handle.container(), iteration_counter_handle.name(),
      &iteration_counter);
  bool owns_resource = false;
  if (errors::IsNotFound(s)) {
    owns_resource = true;
    static std::atomic<int64_t> resource_id_counter(0);
    const std::string& container = ctx->resource_manager()->default_container();
    std::string name =
        strings::StrCat(ctx->op_kernel().name(), "/", kIterationCounter, "_",
                        resource_id_counter.fetch_add(1));
    OP_REQUIRES_OK(ctx,
                   ctx->resource_manager()->LookupOrCreate<IterationCounter>(
                       container, name, &iteration_counter,
                       [](IterationCounter** counter) {
                         *counter = new IterationCounter();
                         return absl::OkStatus();
                       }));
    iteration_counter_handle =
        MakeResourceHandle<IterationCounter>(ctx, container, name);
  } else {
    OP_REQUIRES_OK(ctx, s);
  }

  OP_REQUIRES(
      ctx,
      max_outstanding_requests == model::kAutotune ||
          max_outstanding_requests > 0,
      errors::InvalidArgument(kMaxOutstandingRequests, " must be positive or ",
                              model::kAutotune));

  absl::StatusOr<DataServiceMetadata> metadata =
      GetDataServiceMetadata(dataset_id, address, protocol);
  OP_REQUIRES_OK(ctx, metadata.status());

  bool should_uncompress = op_version_ >= 3 && uncompress_;
  if (should_uncompress) {
    absl::StatusOr<DataServiceMetadata::Compression> compression =
        GetValidatedCompression(dataset_id, *metadata);
    OP_REQUIRES_OK(ctx, compression.status());
    should_uncompress =
        should_uncompress &&
        (*compression == DataServiceMetadata::COMPRESSION_SNAPPY);
  }
  if (should_uncompress) {
    absl::StatusOr<bool> disable_compression_at_runtime =
        DisableCompressionAtRuntime(data_transfer_protocol_,
                                    config->deployment_mode());
    OP_REQUIRES_OK(ctx, disable_compression_at_runtime.status());
    absl::StatusOr<bool> compression_disabled_at_runtime =
        CompressionDisabledAtRuntime(dataset_id, address, protocol,
                                     *disable_compression_at_runtime);
    OP_REQUIRES_OK(ctx, compression_disabled_at_runtime.status());
    metrics::RecordTFDataServiceRuntimeCompressionDecision(
        *compression_disabled_at_runtime);
    should_uncompress = should_uncompress && !*compression_disabled_at_runtime;
  }

  DataTypeVector data_service_output_types = output_types_;
  std::vector<PartialTensorShape> data_service_output_shapes = output_shapes_;
  if (should_uncompress) {
    data_service_output_types = {DT_VARIANT};
    data_service_output_shapes = {TensorShape({})};
  }

  std::unique_ptr<CapturedFunction> captured_uncompress_func;
  if (op_version_ >= 3) {
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(ctx, uncompress_fn_,
                                      /*captured_inputs=*/std::vector<Tensor>{},
                                      &captured_uncompress_func));
  }

  std::optional<CrossTrainerCacheOptions> cross_trainer_cache_options;
  if (!seriazlied_cross_trainer_cache_options_.empty()) {
    cross_trainer_cache_options.emplace();
    cross_trainer_cache_options->ParseFromString(
        seriazlied_cross_trainer_cache_options_);
  }
  DatasetBase* dataset = new Dataset(
      ctx, op_version_, dataset_id, processing_mode, address, protocol,
      data_transfer_protocol_, job_name, consumer_index, num_consumers,
      max_outstanding_requests, task_refresh_interval_hint_, target_workers_,
      *metadata, iteration_counter, owns_resource, iteration_counter_handle,
      std::move(captured_uncompress_func), cross_trainer_cache_options,
      data_service_output_types, data_service_output_shapes);
  if (should_uncompress) {
    VLOG(2) << "Inserting a ParallelMap dataset to uncompress tf.data service "
            << "dataset " << dataset_id << ".";
    dataset->Initialize(/*metadata=*/{});
    captured_uncompress_func.reset();
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(ctx, uncompress_fn_,
                                      /*captured_inputs=*/std::vector<Tensor>{},
                                      &captured_uncompress_func));

    // Release the ownership of `dataset` and transfer it to the ParallelMap
    // dataset for uncompression.
    core::ScopedUnref unref(dataset);
    dataset = MakeDataServiceUncompressDataset(
                  /*input=*/dataset, std::move(captured_uncompress_func),
                  output_types_, output_shapes_)
                  .release();
  }
  *output = dataset;
}

REGISTER_KERNEL_BUILDER(Name(kDataServiceDatasetV1).Device(DEVICE_CPU),
                        DataServiceDatasetOp);
REGISTER_KERNEL_BUILDER(Name(kDataServiceDatasetV2).Device(DEVICE_CPU),
                        DataServiceDatasetOp);
REGISTER_KERNEL_BUILDER(Name(kDataServiceDatasetV3).Device(DEVICE_CPU),
                        DataServiceDatasetOp);
REGISTER_KERNEL_BUILDER(Name(kDataServiceDatasetV4).Device(DEVICE_CPU),
                        DataServiceDatasetOp);
REGISTER_KERNEL_BUILDER(Name("DummyIterationCounter").Device(DEVICE_CPU),
                        DummyResourceOp<IterationCounter>);

REGISTER_KERNEL_BUILDER(Name(kDataServiceDatasetV4)
                            .Device(DEVICE_GPU)
                            .HostMemory("dataset_id")
                            .HostMemory("processing_mode")
                            .HostMemory("address")
                            .HostMemory("protocol")
                            .HostMemory("job_name")
                            .HostMemory("consumer_index")
                            .HostMemory("num_consumers")
                            .HostMemory("max_outstanding_requests")
                            .HostMemory("iteration_counter")
                            .HostMemory("handle"),
                        DataServiceDatasetOp);
REGISTER_KERNEL_BUILDER(Name("DummyIterationCounter").Device(DEVICE_GPU),
                        DummyResourceOp<IterationCounter>);

}  // namespace data
}  // namespace tensorflow
