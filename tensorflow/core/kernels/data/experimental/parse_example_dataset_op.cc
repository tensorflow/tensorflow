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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/kernels/data/parallel_map_dataset_op.h"
#include "tensorflow/core/kernels/ragged_tensor_variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/util/example_proto_fast_parsing.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kInvocationResults[] = "invocation_results";
constexpr char kSizeSuffix[] = ".size";
constexpr char kEndOfInputSuffix[] = ".end_of_input";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessage[] = ".error_message";

// Period between reporting dataset statistics.
constexpr int kStatsReportingPeriodMillis = 1000;

class ParseExampleDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "ParseExample";

  explicit ParseExampleDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()),
        op_version_(ctx->HasAttr("deterministic") ? 2 : 1) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_keys", &sparse_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_keys", &dense_keys_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_types", &sparse_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tdense", &dense_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_shapes", &dense_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));

    if (op_version_ == 1) {
      bool sloppy;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("sloppy", &sloppy));
      if (sloppy) {
        deterministic_ =
            DeterminismPolicy(DeterminismPolicy::Type::kNondeterministic);
      } else {
        deterministic_ = DeterminismPolicy(DeterminismPolicy::Type::kDefault);
      }
    }
    if (op_version_ == 2) {
      std::string deterministic;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("deterministic", &deterministic));
      OP_REQUIRES_OK(
          ctx, DeterminismPolicy::FromString(deterministic, &deterministic_));
    }

    has_ragged_keys_ = ctx->HasAttr("ragged_keys");
    if (has_ragged_keys_) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ragged_keys", &ragged_keys_));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("ragged_value_types", &ragged_value_types_));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("ragged_split_types", &ragged_split_types_));
    }
    for (int i = 0; i < dense_shapes_.size(); ++i) {
      bool shape_ok = true;
      if (dense_shapes_[i].dims() == -1) {
        shape_ok = false;
      } else {
        for (int d = 1; d < dense_shapes_[i].dims(); ++d) {
          if (dense_shapes_[i].dim_size(d) == -1) {
            shape_ok = false;
          }
        }
      }
      OP_REQUIRES(ctx, shape_ok,
                  errors::InvalidArgument(
                      "dense_shapes[", i,
                      "] has unknown rank or unknown inner dimensions: ",
                      dense_shapes_[i].DebugString()));
      TensorShape dense_shape;
      if (dense_shapes_[i].dims() > 0 && dense_shapes_[i].dim_size(0) == -1) {
        variable_length_.push_back(true);
        for (int d = 1; d < dense_shapes_[i].dims(); ++d) {
          dense_shape.AddDim(dense_shapes_[i].dim_size(d));
        }
      } else {
        variable_length_.push_back(false);
        dense_shapes_[i].AsTensorShape(&dense_shape);
      }
      elements_per_stride_.push_back(dense_shape.num_elements());
    }
    metrics::RecordParseDenseFeature(dense_keys_.size());
    metrics::RecordParseSparseFeature(sparse_keys_.size());
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64_t num_parallel_calls = 0;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(
        ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
        errors::InvalidArgument(
            "num_parallel_calls must be greater than zero."));

    OpInputList dense_default_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("dense_defaults", &dense_default_tensors));

    OP_REQUIRES(ctx, dense_default_tensors.size() == dense_keys_.size(),
                errors::InvalidArgument(
                    "Expected len(dense_defaults) == len(dense_keys) but got: ",
                    dense_default_tensors.size(), " vs. ", dense_keys_.size()));

    std::vector<Tensor> dense_defaults(dense_default_tensors.begin(),
                                       dense_default_tensors.end());

    for (int d = 0; d < dense_keys_.size(); ++d) {
      const Tensor& def_value = dense_defaults[d];
      if (variable_length_[d]) {
        OP_REQUIRES(ctx, def_value.NumElements() == 1,
                    errors::InvalidArgument(
                        "dense_shape[", d, "] is a variable length shape: ",
                        dense_shapes_[d].DebugString(),
                        ", therefore "
                        "def_value[",
                        d,
                        "] must contain a single element ("
                        "the padding element).  But its shape is: ",
                        def_value.shape().DebugString()));
      } else if (def_value.NumElements() > 0) {
        OP_REQUIRES(ctx, dense_shapes_[d].IsCompatibleWith(def_value.shape()),
                    errors::InvalidArgument(
                        "def_value[", d,
                        "].shape() == ", def_value.shape().DebugString(),
                        " is not compatible with dense_shapes_[", d,
                        "] == ", dense_shapes_[d].DebugString()));
      }
      OP_REQUIRES(ctx, def_value.dtype() == dense_types_[d],
                  errors::InvalidArgument(
                      "dense_defaults[", d, "].dtype() == ",
                      DataTypeString(def_value.dtype()), " != dense_types_[", d,
                      "] == ", DataTypeString(dense_types_[d])));
    }

    example::FastParseExampleConfig config;
    std::map<string, int> key_to_output_index;
    for (int d = 0; d < dense_keys_.size(); ++d) {
      config.dense.push_back({dense_keys_[d], dense_types_[d], dense_shapes_[d],
                              dense_default_tensors[d], variable_length_[d],
                              elements_per_stride_[d]});
      auto result = key_to_output_index.insert({dense_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          dense_keys_[d]));
    }
    for (int d = 0; d < sparse_keys_.size(); ++d) {
      config.sparse.push_back({sparse_keys_[d], sparse_types_[d]});
      auto result = key_to_output_index.insert({sparse_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          sparse_keys_[d]));
    }
    for (int d = 0; d < ragged_keys_.size(); ++d) {
      config.ragged.push_back(
          {ragged_keys_[d], ragged_value_types_[d], ragged_split_types_[d]});
      auto result = key_to_output_index.insert({ragged_keys_[d], 0});
      OP_REQUIRES(ctx, result.second,
                  errors::InvalidArgument("Duplicate key not allowed: ",
                                          ragged_keys_[d]));
    }
    int i = 0;
    for (auto it = key_to_output_index.begin(); it != key_to_output_index.end();
         it++) {
      it->second = i++;
    }

    *output = new Dataset(
        ctx, input, dense_defaults, sparse_keys_, dense_keys_,
        std::move(key_to_output_index), std::move(config), num_parallel_calls,
        sparse_types_, dense_types_, dense_shapes_, output_types_,
        output_shapes_, deterministic_, has_ragged_keys_, ragged_keys_,
        ragged_value_types_, ragged_split_types_, op_version_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::vector<Tensor> dense_defaults, std::vector<string> sparse_keys,
            std::vector<string> dense_keys,
            std::map<string, int> key_to_output_index,
            example::FastParseExampleConfig config, int32_t num_parallel_calls,
            const DataTypeVector& sparse_types,
            const DataTypeVector& dense_types,
            const std::vector<PartialTensorShape>& dense_shapes,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            const DeterminismPolicy& deterministic, bool has_ragged_keys,
            std::vector<string> ragged_keys,
            const DataTypeVector& ragged_value_types,
            const DataTypeVector& ragged_split_types, int op_version)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          dense_defaults_(std::move(dense_defaults)),
          sparse_keys_(std::move(sparse_keys)),
          dense_keys_(std::move(dense_keys)),
          ragged_keys_(std::move(ragged_keys)),
          key_to_output_index_(std::move(key_to_output_index)),
          config_(std::move(config)),
          num_parallel_calls_(num_parallel_calls),
          sparse_types_(sparse_types),
          dense_types_(dense_types),
          ragged_value_types_(ragged_value_types),
          ragged_split_types_(ragged_split_types),
          dense_shapes_(dense_shapes),
          output_types_(output_types),
          output_shapes_(output_shapes),
          deterministic_(deterministic),
          has_ragged_keys_(has_ragged_keys),
          op_version_(op_version) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      name_utils::IteratorPrefixParams params;
      params.op_version = op_version_;
      return std::make_unique<Iterator>(Iterator::Params{
          this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      name_utils::DatasetDebugStringParams params;
      params.op_version = op_version_;
      return name_utils::DatasetDebugString(kDatasetType, params);
    }

    int64_t CardinalityInternal(CardinalityOptions options) const override {
      return input_->Cardinality(options);
    }

    absl::Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return absl::OkStatus();
    }

    absl::Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* num_parallel_calls_node;
      std::vector<Node*> dense_defaults_nodes;
      dense_defaults_nodes.reserve(dense_defaults_.size());

      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls_node));

      for (const Tensor& dense_default : dense_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(dense_default, &node));
        dense_defaults_nodes.emplace_back(node);
      }

      std::vector<std::pair<absl::string_view, AttrValue>> attrs;

      AttrValue sparse_keys_attr;
      b->BuildAttrValue(sparse_keys_, &sparse_keys_attr);
      attrs.emplace_back("sparse_keys", sparse_keys_attr);

      AttrValue dense_keys_attr;
      b->BuildAttrValue(dense_keys_, &dense_keys_attr);
      attrs.emplace_back("dense_keys", dense_keys_attr);

      AttrValue sparse_types_attr;
      b->BuildAttrValue(sparse_types_, &sparse_types_attr);
      attrs.emplace_back("sparse_types", sparse_types_attr);

      AttrValue dense_attr;
      b->BuildAttrValue(dense_types_, &dense_attr);
      attrs.emplace_back("Tdense", dense_attr);

      AttrValue dense_shapes_attr;
      b->BuildAttrValue(dense_shapes_, &dense_shapes_attr);
      attrs.emplace_back("dense_shapes", dense_shapes_attr);

      if (op_version_ == 1) {
        AttrValue sloppy_attr;
        b->BuildAttrValue(deterministic_.IsNondeterministic(), &sloppy_attr);
        attrs.emplace_back("sloppy", sloppy_attr);
      }
      if (op_version_ == 2) {
        AttrValue deterministic_attr;
        b->BuildAttrValue(deterministic_.String(), &deterministic_attr);
        attrs.emplace_back("deterministic", deterministic_attr);
      }

      if (has_ragged_keys_) {
        AttrValue ragged_keys_attr;
        b->BuildAttrValue(ragged_keys_, &ragged_keys_attr);
        attrs.emplace_back("ragged_keys", ragged_keys_attr);

        AttrValue ragged_value_types_attr;
        b->BuildAttrValue(ragged_value_types_, &ragged_value_types_attr);
        attrs.emplace_back("ragged_value_types", ragged_value_types_attr);

        AttrValue ragged_split_types_attr;
        b->BuildAttrValue(ragged_split_types_, &ragged_split_types_attr);
        attrs.emplace_back("ragged_split_types", ragged_split_types_attr);
      }

      TF_RETURN_IF_ERROR(b->AddDataset(this,
                                       {
                                           {0, input_graph_node},
                                           {1, num_parallel_calls_node},
                                       },
                                       {{2, dense_defaults_nodes}}, attrs,
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
                params.dataset->num_parallel_calls_, mu_, cond_var_)),
            deterministic_(params.dataset->deterministic_.IsDeterministic() ||
                           params.dataset->deterministic_.IsDefault()),
            autotune_(params.dataset->num_parallel_calls_ == model::kAutotune) {
      }

      ~Iterator() override {
        CancelThreads(/*wait=*/true);
        if (deregister_fn_) deregister_fn_();
      }

      absl::Status Initialize(IteratorContext* ctx) override {
        mutex_lock l(*mu_);
        if (num_parallel_calls_->value == model::kAutotune) {
          num_parallel_calls_->value = GetAutotuneDefaultParallelism(ctx);
        }
        TF_RETURN_IF_ERROR(RegisterCancellationCallback(
            ctx->cancellation_manager(),
            [this]() { CancelThreads(/*wait=*/false); }, &deregister_fn_));
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
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
          return tsl::profiler::TraceMeEncode("ParseExampleConsume",
                                              {{"element_id", result->id}});
        });
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

      absl::Status SaveInternal(SerializationContext* ctx,
                                IteratorStateWriter* writer) override {
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
            full_name(strings::StrCat(kInvocationResults, kSizeSuffix)),
            invocation_results_.size()));
        for (size_t i = 0; i < invocation_results_.size(); i++) {
          const auto& result = *(invocation_results_[i]);
          TF_RETURN_IF_ERROR(WriteStatusLocked(writer, i, result.status));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kInvocationResults, "[", i, "]",
                                        kSizeSuffix)),
              result.return_values.size()));
          for (size_t j = 0; j < result.return_values.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(
                    strings::StrCat(kInvocationResults, "[", i, "][", j, "]")),
                result.return_values[j]));
          }
          if (result.end_of_input) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kInvocationResults, "[", i, "]",
                                          kEndOfInputSuffix)),
                ""));
          }
        }
        return absl::OkStatus();
      }

      absl::Status RestoreInternal(IteratorContext* ctx,
                                   IteratorStateReader* reader) override {
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        int64_t invocation_results_size;
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
            int64_t size;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kInvocationResults, "[", i, "]",
                                          kSizeSuffix)),
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
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(),
                full_name(
                    strings::StrCat(kInvocationResults, "[", i, "][", j, "]")),
                &result.return_values.back()));
          }
          result.end_of_input = reader->Contains(full_name(strings::StrCat(
              kInvocationResults, "[", i, "]", kEndOfInputSuffix)));
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
        result.push_back(std::make_pair(
            "parallelism", parallelism == -1
                               ? kTraceInfoUnavailable
                               : strings::Printf("%lld", static_cast<long long>(
                                                             parallelism))));
        return result;
      }

     private:
      struct InvocationResult {
        InvocationResult() = default;
        explicit InvocationResult(int64_t id) : id(id) {}

        Notification notification;
        absl::Status status;
        std::vector<Tensor> return_values;
        bool end_of_input = false;
        int64_t id = -1;
      };

      void CancelThreads(bool wait) TF_LOCKS_EXCLUDED(mu_) {
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
        RecordBufferEnqueue(ctx.get(), result->return_values);
        result->notification.Notify();
        cond_var_->notify_all();
      }

      void CallFunction(const std::shared_ptr<IteratorContext>& ctx,
                        const std::shared_ptr<InvocationResult>& result)
          TF_LOCKS_EXCLUDED(*mu_) {
        tsl::profiler::TraceMe traceme([&] {
          return tsl::profiler::TraceMeEncode("ParseExampleProduce",
                                              {{"element_id", result->id}});
        });
        // Get the next input element.
        std::vector<Tensor> input_element;
        result->status = input_impl_->GetNext(ctx.get(), &input_element,
                                              &result->end_of_input);
        if (result->end_of_input || !result->status.ok()) {
          CallCompleted(ctx, result);
          return;
        }

        auto done = [this, ctx, result](absl::Status status) {
          result->status.Update(status);
          CallCompleted(ctx, result);
        };

        // We schedule the `ParseExample` function using `ctx->runner()` to
        // enable applying it concurrently over different input elements.
        auto fn = std::bind(
            [this, ctx, result](std::vector<Tensor> input_element) {
              return ParseExample(ctx.get(), std::move(input_element),
                                  &result->return_values);
            },
            std::move(input_element));
        auto node = model_node();
        const bool collect_usage = node && ctx->model();
        // `ctx->runner()` may execute its logic synchronous so we wrap it in
        // `RecordStop` and `RecordStart` to prevent invalid nesting of
        // `RecordStart` calls.
        RecordStop(ctx.get());
        (*ctx->runner())([node, collect_usage, fn = std::move(fn),
                          done = std::move(done)]() {
          if (collect_usage) {
            node->record_start(EnvTime::NowNanos());
          }
          done(fn());
          if (collect_usage) {
            node->record_stop(EnvTime::NowNanos());
          }
        });
        RecordStart(ctx.get());
      }

      absl::Status CheckOutputTensor(const Tensor& tensor, size_t value_index,
                                     size_t output_index) const {
        if (tensor.dtype() != dataset()->output_dtypes()[output_index]) {
          return errors::InvalidArgument(
              "Got wrong type for FastParseExample return value ", value_index,
              " (expected ",
              DataTypeString(dataset()->output_dtypes()[output_index]),
              ", got ", DataTypeString(tensor.dtype()), ").");
        }
        if (!dataset()->output_shapes()[output_index].IsCompatibleWith(
                tensor.shape())) {
          return errors::InvalidArgument(
              "Got wrong shape for FastParseExample return value ", value_index,
              " (expected ",
              dataset()->output_shapes()[output_index].DebugString(), ", got ",
              tensor.shape().DebugString(), ").");
        }
        return absl::OkStatus();
      }

      absl::Status ParseExample(IteratorContext* ctx, std::vector<Tensor> input,
                                std::vector<Tensor>* output) {
        thread::ThreadPool* device_threadpool =
            ctx->flr()->device()->tensorflow_cpu_worker_threads()->workers;
        std::vector<tstring> slice_vec;
        for (const Tensor& t : input) {
          auto serialized_t = t.flat<tstring>();
          absl::Span<const tstring> slice(serialized_t.data(),
                                          serialized_t.size());
          for (auto it = slice.begin(); it != slice.end(); it++)
            slice_vec.push_back(*it);
        }
        example::FastParseExampleConfig config = dataset()->config_;
        // local copy of config_ for modification.
        auto stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator) {
          config.collect_feature_stats = true;
        }
        example::Result example_result;
        TF_RETURN_IF_ERROR(FastParseExample(
            config, slice_vec, {}, device_threadpool, &example_result));
        (*output).resize(dataset()->key_to_output_index_.size());
        for (int d = 0; d < dataset()->dense_keys_.size(); ++d) {
          int output_index =
              dataset()->key_to_output_index_.at(dataset()->dense_keys_[d]);
          TF_RETURN_IF_ERROR(CheckOutputTensor(example_result.dense_values[d],
                                               d, output_index));
          (*output)[output_index] = example_result.dense_values[d];
        }
        for (int d = 0; d < dataset()->sparse_keys_.size(); ++d) {
          int output_index =
              dataset()->key_to_output_index_.at(dataset()->sparse_keys_[d]);
          (*output)[output_index] = Tensor(ctx->allocator({}), DT_VARIANT, {3});
          Tensor& serialized_sparse = (*output)[output_index];
          auto serialized_sparse_t = serialized_sparse.vec<Variant>();
          serialized_sparse_t(0) = example_result.sparse_indices[d];
          serialized_sparse_t(1) = example_result.sparse_values[d];
          serialized_sparse_t(2) = example_result.sparse_shapes[d];
          TF_RETURN_IF_ERROR(
              CheckOutputTensor(serialized_sparse, d, output_index));
        }
        for (int d = 0; d < dataset()->ragged_keys_.size(); ++d) {
          int output_index =
              dataset()->key_to_output_index_.at(dataset()->ragged_keys_[d]);
          RaggedTensorVariant serialized_ragged;
          serialized_ragged.append_splits(example_result.ragged_splits[d]);
          serialized_ragged.set_values(example_result.ragged_values[d]);
          (*output)[output_index] = Tensor(ctx->allocator({}), DT_VARIANT, {});
          Tensor& ragged_wrapper = (*output)[output_index];
          ragged_wrapper.scalar<Variant>()() = serialized_ragged;
          TF_RETURN_IF_ERROR(
              CheckOutputTensor(ragged_wrapper, d, output_index));
        }
        if (stats_aggregator) {
          stats_aggregator->IncrementCounter(
              stats_utils::kExamplesCount, "trainer",
              example_result.feature_stats.size());
          for (example::PerExampleFeatureStats feature_stats :
               example_result.feature_stats) {
            stats_aggregator->IncrementCounter(stats_utils::kFeaturesCount,
                                               "trainer",
                                               feature_stats.features_count);
            stats_aggregator->IncrementCounter(
                stats_utils::kFeatureValuesCount, "trainer",
                feature_stats.feature_values_count);
            int64_t steps = model_node() ? model_node()->num_elements() : 0;
            stats_aggregator->AddToHistogram(
                stats_utils::FeatureHistogramName(dataset()->node_name()),
                {static_cast<double>(feature_stats.features_count)}, steps);

            stats_aggregator->AddToHistogram(
                stats_utils::FeatureValueHistogramName(dataset()->node_name()),
                {static_cast<double>(feature_stats.feature_values_count)},
                steps);
          }
        }
        return absl::OkStatus();
      }

      absl::Status ProcessResult(
          IteratorContext* ctx, const std::shared_ptr<InvocationResult>& result,
          std::vector<Tensor>* out_tensors, bool* end_of_sequence)
          TF_LOCKS_EXCLUDED(*mu_) {
        if (!result->end_of_input && result->status.ok()) {
          *out_tensors = std::move(result->return_values);
          RecordBufferDequeue(ctx, *out_tensors);
          *end_of_sequence = false;
          return absl::OkStatus();
        }
        if (absl::IsOutOfRange(result->status)) {
          // To guarantee that the transformation preserves the cardinality of
          // the dataset, we convert `OutOfRange` to `InvalidArgument` as the
          // former may be interpreted by a caller as the end of sequence.
          return errors::InvalidArgument(
              "Function invocation produced OutOfRangeError: ",
              result->status.message());
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
        // Counts the total number of calls to use as an id of InvocationResult.
        int64_t num_total_calls = 0;
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
                  std::make_shared<InvocationResult>(num_total_calls++));
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

      // Determines whether the caller needs to wait for a result. Upon
      // returning false, `result` will point to the result.
      bool ShouldWait(std::shared_ptr<InvocationResult>* result)
          TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (cancelled_) {
          return false;
        }
        if (!deterministic_) {
          // Iterate through in-flight results and returns the first one that is
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

      void StatsThread(const std::shared_ptr<IteratorContext>& ctx) {
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

      absl::Status WriteStatusLocked(IteratorStateWriter* writer, size_t index,
                                     const absl::Status& status)
          TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            CodeKey(index), static_cast<int64_t>(status.code())));
        if (!status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              ErrorMessageKey(index), std::string(status.message())));
        }
        return absl::OkStatus();
      }

      absl::Status ReadStatusLocked(IteratorStateReader* reader, size_t index,
                                    absl::Status* status)
          TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        int64_t code_int;
        TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
        absl::StatusCode code = static_cast<absl::StatusCode>(code_int);

        if (code != absl::StatusCode::kOk) {
          tstring error_message;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(ErrorMessageKey(index), &error_message));
          *status = absl::Status(code, error_message);
        } else {
          *status = absl::OkStatus();
        }
        return absl::OkStatus();
      }

      string CodeKey(size_t index) {
        return full_name(
            strings::StrCat(kInvocationResults, "[", index, "]", kCodeSuffix));
      }

      string ErrorMessageKey(size_t index) {
        return full_name(strings::StrCat(kInvocationResults, "[", index, "]",
                                         kErrorMessage));
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
      const bool autotune_;
      // Counts the number of outstanding calls.
      int64_t num_calls_ TF_GUARDED_BY(*mu_) = 0;
      std::unique_ptr<IteratorBase> input_impl_;
      // Buffer for storing the invocation results.
      std::deque<std::shared_ptr<InvocationResult>> invocation_results_
          TF_GUARDED_BY(*mu_);
      bool cancelled_ TF_GUARDED_BY(*mu_) = false;

      std::unique_ptr<Thread> runner_thread_ TF_GUARDED_BY(*mu_);
      std::unique_ptr<Thread> stats_thread_ TF_GUARDED_BY(*mu_);

      // Method for deregistering the cancellation callback.
      std::function<void()> deregister_fn_;
    };

    const DatasetBase* const input_;
    const std::vector<Tensor> dense_defaults_;
    const std::vector<string> sparse_keys_;
    const std::vector<string> dense_keys_;
    const std::vector<string> ragged_keys_;
    const std::map<string, int> key_to_output_index_;
    const example::FastParseExampleConfig config_;
    const int64_t num_parallel_calls_;
    const DataTypeVector sparse_types_;
    const DataTypeVector dense_types_;
    const DataTypeVector ragged_value_types_;
    const DataTypeVector ragged_split_types_;
    const std::vector<PartialTensorShape> dense_shapes_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const DeterminismPolicy deterministic_;
    const bool has_ragged_keys_;
    const int op_version_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  DeterminismPolicy deterministic_;
  std::vector<string> sparse_keys_;
  std::vector<string> dense_keys_;
  std::vector<string> ragged_keys_;
  DataTypeVector sparse_types_;
  DataTypeVector dense_types_;
  DataTypeVector ragged_value_types_;
  DataTypeVector ragged_split_types_;
  std::vector<PartialTensorShape> dense_shapes_;
  std::vector<bool> variable_length_;
  std::vector<std::size_t> elements_per_stride_;
  bool has_ragged_keys_;
  const int op_version_;
};

REGISTER_KERNEL_BUILDER(Name("ParseExampleDataset").Device(DEVICE_CPU),
                        ParseExampleDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ParseExampleDatasetV2").Device(DEVICE_CPU),
                        ParseExampleDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalParseExampleDataset").Device(DEVICE_CPU),
    ParseExampleDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
