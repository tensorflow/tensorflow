/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/root_dataset.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tsl/platform/host_info.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDatasetType[] = "Root";

constexpr char kAlgorithm[] = "algorithm";
constexpr char kCpuBudget[] = "cpu_budget";
constexpr char kExperiments[] = "experiments";
constexpr char kReadRoundtripLatency[] = "read_latency_usec";
constexpr char kReadResponseBytes[] = "read_bytes";
constexpr char kIntraOpParallelism[] = "intra_op_parallelism";
constexpr char kMemBandwidth[] = "mem_bw_used_megabytes_per_sec";
constexpr char kPrivateThreadpoolSize[] = "threadpool_size";
constexpr char kRamBudget[] = "ram_budget_megabytes";
constexpr char kRamUsage[] = "ram_usage_megabytes";
constexpr char kMaxBufferBytes[] = "max_buffered_megabytes";
constexpr char kWarmStart[] = "warm_start";

// If value `x` matches `y`, returns default value `z`. Otherwise, return `x`.
inline int64_t value_or_default(int64_t x, int64_t y, int64_t z) {
  return x == y ? z : x;
}

void SetRootDatasetParams(const Options& options, RootDataset::Params* params) {
  if (ShouldConfigureMaxIntraOpParallelism(options)) {
    params->max_intra_op_parallelism =
        options.threading_options().max_intra_op_parallelism();
  }
  if (ShouldUsePrivateThreadPool(options)) {
    params->private_threadpool_size =
        options.threading_options().private_threadpool_size();
  }
  params->autotune = ShouldUseAutotuning(options);
  params->autotune_algorithm = model::AutotuneAlgorithm::DEFAULT;
  auto experiments = GetExperiments();
  if (experiments.contains("stage_based_autotune") ||
      experiments.contains("stage_based_autotune_v2")) {
    params->autotune_algorithm = model::AutotuneAlgorithm::STAGE_BASED;
  }
  if (options.autotune_options().optional_autotune_algorithm_case() ==
      AutotuneOptions::kAutotuneAlgorithm) {
    params->autotune_algorithm =
        options.autotune_options().autotune_algorithm();
  }
  int64_t cpu_budget_from_options = options.autotune_options().cpu_budget();
  if (cpu_budget_from_options == 0) {
    params->autotune_cpu_budget_func = [] { return GetCpuBudget(); };
  } else {
    params->autotune_cpu_budget_func = [cpu_budget_from_options] {
      return cpu_budget_from_options;
    };
  }
  params->autotune_ram_budget_from_options =
      options.autotune_options().ram_budget();
  double ram_budget_share;
  if (experiments.contains("autotune_buffer_optimization")) {
    // When running this experiment, increase the ram_budget since it already
    // takes into account the ram usage in buffer sizing, which is not the
    // case for prefetch autotuner. Without this, we see degradation in some
    // jobs for lack of buffers while ram usage is low.
    ram_budget_share = 0.9;
  } else {
    ram_budget_share = model::kRamBudgetShare;
  }
  params->ram_budget_share = ram_budget_share;
}

void AddTraceMetadata(const RootDataset::Params& params, const Options& options,
                      TraceMeMetadata* trace_metadata) {
  if (params.autotune) {
    trace_metadata->push_back(std::make_pair(
        kAlgorithm, model::AutotuneAlgorithm_Name(params.autotune_algorithm)));
    trace_metadata->push_back(std::make_pair(
        kCpuBudget,
        strings::Printf("%lld", static_cast<long long>(
                                    params.autotune_cpu_budget_func()))));
    int64_t ram_budget = params.ComputeInitialAutotuneRamBudget();
    trace_metadata->push_back(std::make_pair(
        kRamBudget,
        strings::Printf("%lld", static_cast<long long>(ram_budget / 1.0e6))));
  }
  if (params.max_intra_op_parallelism >= 0) {
    trace_metadata->push_back(std::make_pair(
        kIntraOpParallelism,
        strings::Printf("%lld", static_cast<long long>(value_or_default(
                                    params.max_intra_op_parallelism, 0,
                                    port::MaxParallelism())))));
  }
  if (params.private_threadpool_size >= 0) {
    trace_metadata->push_back(std::make_pair(
        kPrivateThreadpoolSize,
        strings::Printf("%lld", static_cast<long long>(value_or_default(
                                    params.private_threadpool_size, 0,
                                    port::MaxParallelism())))));
  }
  auto experiments = GetExperiments();
  if (!experiments.empty()) {
    trace_metadata->push_back(
        std::make_pair(kExperiments, absl::StrJoin(experiments, " ")));
  }
  trace_metadata->push_back(
      std::make_pair(kWarmStart, options.warm_start() ? "true" : "false"));
}
}  // namespace

// static
Status RootDataset::FromOptions(const DatasetBase* input,
                                DatasetBase** output) {
  Params params;
  SetRootDatasetParams(input->options(), &params);
  *output = new RootDataset(input, params);
  (*output)->Initialize(/*metadata=*/{});
  for (const auto& framework : input->options().framework_type()) {
    metrics::RecordTFDataFrameworkType(framework);
  }
  return absl::OkStatus();
}

Status RootDataset::FromOptions(core::RefCountPtr<DatasetBase> input,
                                DatasetBase** output) {
  Params params;
  for (const auto& framework : input->options().framework_type()) {
    metrics::RecordTFDataFrameworkType(framework);
  }
  SetRootDatasetParams(input->options(), &params);
  *output = new RootDataset(std::move(input), params);
  (*output)->Initialize(/*metadata=*/{});
  return absl::OkStatus();
}

class RootDataset::Iterator : public DatasetIterator<RootDataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<RootDataset>(params) {
    if (dataset()->params_.max_intra_op_parallelism >= 0) {
      max_intra_op_parallelism_ =
          value_or_default(dataset()->params_.max_intra_op_parallelism, 0,
                           port::MaxParallelism());
    }
    if (dataset()->params_.private_threadpool_size >= 0) {
      threadpool_size_ =
          value_or_default(dataset()->params_.private_threadpool_size, 0,
                           port::MaxParallelism());
      thread_pool_ = std::make_unique<thread::ThreadPool>(
          Env::Default(), ThreadOptions{}, "data_private_threadpool",
          threadpool_size_);
    }
    cancellation_manager_ = std::make_unique<CancellationManager>();
  }

  ~Iterator() override { cancellation_manager_->StartCancel(); }

  bool SymbolicCheckpointCompatible() const override { return true; }

  Status Initialize(IteratorContext* ctx) override {
    // prefetch_autotuner.h currently disregards `autotune` parameter
    // so no matter whether dataset()->params_.autotune is on or not
    // we need to pass ram_budget_manager_ to the downstream dataset operations
    ram_budget_manager_ = std::make_shared<model::RamBudgetManager>(
        dataset()->params_.ComputeInitialAutotuneRamBudget());

    if (dataset()->params_.autotune) {
      if (ctx->model() != nullptr) {
        model_ = ctx->model();
      } else {
        model_ = std::make_shared<model::Model>();
        ctx->SetModel(model_);
      }

      absl::flat_hash_set<string> experiments = GetExperiments();
      if (experiments.contains("stage_based_autotune_v2")) {
        model_->AddExperiment("stage_based_autotune_v2");
      }
      if (experiments.contains("autotune_buffer_optimization")) {
        model_->AddExperiment("autotune_buffer_optimization");
      }
    }
    IteratorContext iter_ctx(CreateParams(ctx));
    if (model_) {
      auto factory = [&iter_ctx, this](model::Node::Args args) {
        return CreateNode(&iter_ctx, std::move(args));
      };
      model_->AddNode(std::move(factory), prefix(), nullptr, &node_);
    }
    TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(&iter_ctx, this,
                                                       prefix(), &input_impl_));
    ctx->MergeCheckpoint(iter_ctx.checkpoint());
    return absl::OkStatus();
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    {
      tf_shared_lock l(mu_);
      if (model_ != nullptr && end_time_usec_ > 0) {
        model_->RecordIteratorGapTime(ctx->env()->NowMicros() - end_time_usec_);
      }
    }
    if (dataset()->params_.autotune) {
      TF_RETURN_IF_ERROR(EnsureModelThreadStarted(ctx));
    }
    IteratorContext iter_ctx(CreateParams(ctx));
    TF_RETURN_IF_ERROR(
        input_impl_->GetNext(&iter_ctx, out_tensors, end_of_sequence));
    ctx->MergeCheckpoint(iter_ctx.checkpoint());
    {
      mutex_lock l(mu_);
      end_time_usec_ = std::max(ctx->env()->NowMicros(), end_time_usec_);
    }
    return absl::OkStatus();
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
  }

  Status SaveInternal(SerializationContext* ctx,
                      IteratorStateWriter* writer) override {
    TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
    return absl::OkStatus();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    IteratorContext iter_ctx(CreateParams(ctx));
    TF_RETURN_IF_ERROR(RestoreInput(&iter_ctx, reader, input_impl_));
    ctx->MergeCheckpoint(iter_ctx.checkpoint());
    return absl::OkStatus();
  }

  TraceMeMetadata GetTraceMeMetadata() const override {
    tensorflow::data::TraceMeMetadata traceme_metadata =
        dataset()->traceme_metadata_;
    const int64_t mem_bw = port::GetMemoryBandwidthInfo().bw_used;
    if (mem_bw != INT64_MAX) {
      traceme_metadata.push_back(std::make_pair(
          kMemBandwidth,
          strings::Printf("%lld", static_cast<long long>(mem_bw))));
    }
    const auto memory_info = port::GetMemoryInfo();
    const auto memory_usage = memory_info.total - memory_info.free;
    traceme_metadata.push_back(std::make_pair(
        kRamUsage,
        strings::Printf("%lld out of %lld (%.2f%%)",
                        static_cast<long long>(memory_usage / 1.0e6),
                        static_cast<long long>(memory_info.total / 1.0e6),
                        static_cast<double>(100 * memory_usage) /
                            static_cast<double>(memory_info.total))));
    const auto io_statistics = tsl::port::GetIOStatistics();
    if (io_statistics.roundtrip_latency_usec.count > 0) {
      traceme_metadata.push_back(std::make_pair(
          kReadRoundtripLatency,
          strings::Printf(
              "(count: %lld, mean: %lld, std dev: %lld)",
              static_cast<long long>(
                  io_statistics.roundtrip_latency_usec.count),
              static_cast<long long>(io_statistics.roundtrip_latency_usec.mean),
              static_cast<long long>(
                  io_statistics.roundtrip_latency_usec.std_dev))));
    }
    if (io_statistics.response_bytes.count > 0) {
      traceme_metadata.push_back(std::make_pair(
          kReadResponseBytes,
          strings::Printf(
              "(count: %lld, mean: %lld, std dev: %lld)",
              static_cast<long long>(io_statistics.response_bytes.count),
              static_cast<long long>(io_statistics.response_bytes.mean),
              static_cast<long long>(io_statistics.response_bytes.std_dev))));
    }
    return traceme_metadata;
  }

 private:
  IteratorContext::Params CreateParams(IteratorContext* ctx) {
    IteratorContext::Params params(ctx);
    // prefetch_autotuner.h currently disregards `autotune` parameter
    // so no matter whether dataset()->params_.autotune is on or not
    // we need to pass ram_budget_manager_ to the downstream dataset operations
    params.ram_budget_manager = ram_budget_manager_;
    // After cl/548201925, `ctx->model()` may not be `nullptr` even when
    // autotuning is off, but `model_` should be `nullptr` and it should have
    // been set to a valid model in `Initialize()` if autotuning is on. We
    // should simply set `params.model` to `model_` here.
    params.model = model_;
    if (dataset()->params_.private_threadpool_size >= 0) {
      params.runner = [pool = thread_pool_.get()](std::function<void()> c) {
        pool->Schedule(std::move(c));
      };
      params.runner_threadpool_size = threadpool_size_;
    }
    if (dataset()->params_.max_intra_op_parallelism >= 0) {
      params.runner =
          RunnerWithMaxParallelism(params.runner, max_intra_op_parallelism_);
    }
    params.options = &dataset()->options();
    return params;
  }

  Status EnsureModelThreadStarted(IteratorContext* ctx) {
    mutex_lock l(mu_);
    if (!model_thread_) {
      RunMode run_mode = ctx->run_mode();
      model_thread_ = ctx->StartThread("tf_data_model", [this, run_mode]() {
        RootDataset::Params params = dataset()->params_;
        std::function<int64_t(int64_t)> ram_budget_func;
        std::optional<int64_t> raw_ram_budget;
        if (params.autotune_ram_budget_from_options > 0) {
          raw_ram_budget = params.autotune_ram_budget_from_options;
        } else if (run_mode != RunMode::STANDALONE) {
          // Dynamic RAM budget should only apply to tf.data service.
          raw_ram_budget = params.ComputeInitialAutotuneRamBudget();
        }
        Status status = model_->OptimizeLoop(
            params.autotune_algorithm, params.autotune_cpu_budget_func,
            params.ram_budget_share, raw_ram_budget, *ram_budget_manager_,
            cancellation_manager_.get());
        if (!status.ok()) {
          LOG(WARNING) << "Optimization loop failed: " << status;
        }
      });
    }
    return absl::OkStatus();
  }

  std::shared_ptr<model::Model> model_ = nullptr;
  // `ram_budget_manager_` coordinates the memory budget and allocation
  // between prefetch legacy autotune and `tensorflow::data::model::Model`
  std::shared_ptr<model::RamBudgetManager> ram_budget_manager_ = nullptr;
  // Controls cancellation of `model_thread_`. Must be ordered before
  // `model_thread_` so that `model_thread_` is destroyed first.
  std::unique_ptr<CancellationManager> cancellation_manager_;
  mutex mu_;
  std::unique_ptr<Thread> model_thread_ TF_GUARDED_BY(mu_);
  int64_t max_intra_op_parallelism_;
  int64_t threadpool_size_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  // The end time of the previous `GetNextInternal` call.
  uint64_t end_time_usec_ TF_GUARDED_BY(mu_) = 0;

  // Must be ordered last as its execution may depend on other members.
  std::unique_ptr<IteratorBase> input_impl_;
};

RootDataset::RootDataset(const DatasetBase* input, const Params& params)
    : DatasetBase(DatasetContext({name_utils::OpName(kDatasetType),
                                  name_utils::OpName(kDatasetType)})),
      input_(input),
      params_(std::move(params)) {
  AddTraceMetadata(params_, input_->options(), &traceme_metadata_);
}

RootDataset::RootDataset(core::RefCountPtr<DatasetBase> input,
                         const Params& params)
    : DatasetBase(DatasetContext({name_utils::OpName(kDatasetType),
                                  name_utils::OpName(kDatasetType)})),
      params_(std::move(params)) {
  owned_input_ = std::move(input);
  input_ = owned_input_.get();
  random_indexing_compatible_ = absl::OkStatus();
  if (input_ != nullptr) {
    random_indexing_compatible_ = input_->RandomIndexingCompatible();
  }
  AddTraceMetadata(params_, input_->options(), &traceme_metadata_);
}

RootDataset::~RootDataset() = default;

std::unique_ptr<IteratorBase> RootDataset::MakeIteratorInternal(
    const string& prefix) const {
  return std::make_unique<Iterator>(
      Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)});
}

const DataTypeVector& RootDataset::output_dtypes() const {
  return input_->output_dtypes();
}

const std::vector<PartialTensorShape>& RootDataset::output_shapes() const {
  return input_->output_shapes();
}

string RootDataset::DebugString() const {
  return name_utils::DatasetDebugString(kDatasetType);
}

int64_t RootDataset::CardinalityInternal(CardinalityOptions options) const {
  return input_->Cardinality(options);
}

Status RootDataset::Get(OpKernelContext* ctx, int64 index,
                        std::vector<Tensor>* out_tensors) const {
  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(this->InputDatasets(&inputs));
  return inputs[0]->Get(ctx, index, out_tensors);
}

Status RootDataset::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
  inputs->push_back(input_);
  return absl::OkStatus();
}

Status RootDataset::CheckExternalState() const {
  return input_->CheckExternalState();
}

Status RootDataset::AsGraphDefInternal(SerializationContext* ctx,
                                       DatasetGraphDefBuilder* b,
                                       Node** output) const {
  return errors::Unimplemented("RootDataset does not support serialization.");
}

#if !defined(IS_MOBILE_PLATFORM)
Status FinalizeDataset(OpKernelContext* ctx, const DatasetBase* input,
                       DatasetBase** output) {
  const Options& options = input->options();
  absl::flat_hash_set<tstring> optimizations_enabled;
  absl::flat_hash_set<tstring> optimizations_disabled;
  absl::flat_hash_set<tstring> optimizations_default;
  GetOptimizations(options, &optimizations_enabled, &optimizations_disabled,
                   &optimizations_default);
  // Disable `enable_gradient_descent` as it assumes presence of ModelDatasetOp.
  optimizations_disabled.insert("enable_gradient_descent");

  auto experiments = GetExperiments();
  LogAndRecordExperiments(experiments);
  auto optimizations =
      SelectOptimizations(experiments, optimizations_enabled,
                          optimizations_disabled, optimizations_default);
  if (optimizations.empty()) {
    return RootDataset::FromOptions(input, output);
  }

  auto optimization_configs = CreateGraphRewriteConfigs(options);
  auto config_factory = [&optimizations, &optimization_configs]() {
    return CreateRewriterConfig(optimizations, optimization_configs);
  };
  core::RefCountPtr<DatasetBase> rewritten_output;
  Status s = RewriteDataset(ctx, input, std::move(config_factory),
                            /*record_fingerprint=*/false, &rewritten_output);

  *output = rewritten_output.get();
  bool rewritten = (*output != input);
  if (errors::IsDeadlineExceeded(s)) {
    // Ignore DeadlineExceeded as it implies that the attempted rewrite took too
    // long which should not prevent further computation.
    LOG(WARNING) << s;
  } else if (!s.ok()) {
    return s;
  }
  if (!rewritten) {
    return RootDataset::FromOptions(input, output);
  } else {
    return RootDataset::FromOptions(std::move(rewritten_output), output);
  }
  return absl::OkStatus();
}

#else   // !IS_MOBILE_PLATFORM
Status FinalizeDataset(OpKernelContext* ctx, const DatasetBase* input,
                       DatasetBase** output) {
  return RootDataset::FromOptions(input, output);
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace data
}  // namespace tensorflow
