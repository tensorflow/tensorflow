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
#include "tensorflow/core/kernels/data/finalize_dataset_op.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/experimental/threadpool_dataset_op.h"
#include "tensorflow/core/kernels/data/model_dataset_op.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/optimize_dataset_op.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const FinalizeDatasetOp::kDatasetType;
/* static */ constexpr const char* const FinalizeDatasetOp::kInputDataset;
/* static */ constexpr const char* const FinalizeDatasetOp::kOutputTypes;
/* static */ constexpr const char* const FinalizeDatasetOp::kOutputShapes;

namespace {

// Use "Opt" suffix so that they are not confused with the enums in Options
// proto.
constexpr char kMapVectorizationOpt[] = "map_vectorization";
constexpr char kMapAndBatchFusionOpt[] = "map_and_batch_fusion";
constexpr char kNoopEliminationOpt[] = "noop_elimination";
constexpr char kMapParallelizationOpt[] = "map_parallelization";
constexpr char kShuffleAndRepeatFusionOpt[] = "shuffle_and_repeat_fusion";
constexpr char kFilterFusionOpt[] = "filter_fusion";
constexpr char kFilterWithRandomUniformFusionOpt[] =
    "filter_with_random_uniform_fusion";
constexpr char kHoistRandomUniformOpt[] = "hoist_random_uniform";
constexpr char kMapAndFilterFusionOpt[] = "map_and_filter_fusion";
constexpr char kMapFusionOpt[] = "map_fusion";
constexpr char kParallelBatchOpt[] = "parallel_batch";
constexpr char kReorderDataDiscardingOpsOpt[] = "reorder_data_discarding_ops";
constexpr char kAutotuneBufferSizesOpt[] = "autotune_buffer_sizes";
constexpr char kDisablePrefetchLegacyAutotuneOpt[] =
    "disable_prefetch_legacy_autotune";
constexpr char kMakeSloppyOpt[] = "make_sloppy";
constexpr char kUseChooseFastestOpt[] = "use_choose_fastest";
constexpr char kBatchParallelizationOpt[] = "batch_parallelization";
constexpr char kEnableGradientDescentOpt[] = "enable_gradient_descent";
constexpr char kAutotuneOpt[] = "autotune";
constexpr char kSlackOpt[] = "slack";
constexpr char kSlackPeriodOpt[] = "slack_period";

void MapVectorizationGraphRewrites(const Options& options,
                                   std::set<tstring>* optimization_enabled,
                                   std::set<tstring>* optimization_disabled) {
  if (options.optimization_options()
          .map_vectorization()
          .optional_enabled_case() != MapVectorization::kEnabled) {
    return;
  }
  if (options.optimization_options().map_vectorization().enabled()) {
    optimization_enabled->insert(kMapVectorizationOpt);
  } else {
    optimization_disabled->insert(kMapVectorizationOpt);
  }
}

void DefaultOptimizationGraphRewrites(const Options& options,
                                      std::set<tstring>* optimization_enabled,
                                      std::set<tstring>* optimization_disabled,
                                      std::set<tstring>* optimization_default) {
  MapVectorizationGraphRewrites(options, optimization_enabled,
                                optimization_disabled);
  const auto& optimization_options = options.optimization_options();
  if (optimization_options.optional_apply_default_optimizations_case() !=
          OptimizationOptions::kApplyDefaultOptimizations ||
      optimization_options.apply_default_optimizations()) {
    if (optimization_options.optional_map_and_batch_fusion_case() !=
        OptimizationOptions::kMapAndBatchFusion) {
      optimization_default->insert(kMapAndBatchFusionOpt);
    }
    if (optimization_options.optional_noop_elimination_case() !=
        OptimizationOptions::kNoopElimination) {
      optimization_default->insert(kNoopEliminationOpt);
    }
    if (optimization_options.optional_map_parallelization_case() !=
        OptimizationOptions::kMapParallelization) {
      optimization_default->insert(kMapParallelizationOpt);
    }
    if (optimization_options.optional_shuffle_and_repeat_fusion_case() !=
        OptimizationOptions::kShuffleAndRepeatFusion) {
      optimization_default->insert(kShuffleAndRepeatFusionOpt);
    }
  }
  if (optimization_options.optional_filter_fusion_case() ==
      OptimizationOptions::kFilterFusion) {
    if (optimization_options.filter_fusion()) {
      optimization_enabled->insert(kFilterFusionOpt);
    } else {
      optimization_disabled->insert(kFilterFusionOpt);
    }
  }
  if (optimization_options.optional_filter_with_random_uniform_fusion_case() ==
      OptimizationOptions::kFilterWithRandomUniformFusion) {
    if (optimization_options.filter_with_random_uniform_fusion()) {
      optimization_enabled->insert(kFilterWithRandomUniformFusionOpt);
    } else {
      optimization_disabled->insert(kFilterWithRandomUniformFusionOpt);
    }
  }
  if (optimization_options.optional_hoist_random_uniform_case() ==
      OptimizationOptions::kHoistRandomUniform) {
    if (optimization_options.hoist_random_uniform()) {
      optimization_enabled->insert(kHoistRandomUniformOpt);
    } else {
      optimization_disabled->insert(kHoistRandomUniformOpt);
    }
  }
  if (optimization_options.optional_map_and_batch_fusion_case() ==
      OptimizationOptions::kMapAndBatchFusion) {
    if (optimization_options.map_and_batch_fusion()) {
      optimization_enabled->insert(kMapAndBatchFusionOpt);
    } else {
      optimization_disabled->insert(kMapAndBatchFusionOpt);
    }
  }
  if (optimization_options.optional_map_and_filter_fusion_case() ==
      OptimizationOptions::kMapAndFilterFusion) {
    if (optimization_options.map_and_filter_fusion()) {
      optimization_enabled->insert(kMapAndFilterFusionOpt);
    } else {
      optimization_disabled->insert(kMapAndFilterFusionOpt);
    }
  }
  if (optimization_options.optional_map_parallelization_case() ==
      OptimizationOptions::kMapParallelization) {
    if (optimization_options.map_parallelization()) {
      optimization_enabled->insert(kMapParallelizationOpt);
    } else {
      optimization_disabled->insert(kMapParallelizationOpt);
    }
  }
  if (optimization_options.optional_map_fusion_case() ==
      OptimizationOptions::kMapFusion) {
    if (optimization_options.map_fusion()) {
      optimization_enabled->insert(kMapFusionOpt);
    } else {
      optimization_disabled->insert(kMapFusionOpt);
    }
  }
  if (optimization_options.optional_noop_elimination_case() ==
      OptimizationOptions::kNoopElimination) {
    if (optimization_options.noop_elimination()) {
      optimization_enabled->insert(kNoopEliminationOpt);
    } else {
      optimization_disabled->insert(kNoopEliminationOpt);
    }
  }
  if (optimization_options.optional_parallel_batch_case() ==
      OptimizationOptions::kParallelBatch) {
    if (optimization_options.parallel_batch()) {
      optimization_enabled->insert(kParallelBatchOpt);
    } else {
      optimization_disabled->insert(kParallelBatchOpt);
    }
  }
  if (optimization_options.optional_reorder_data_discarding_ops_case() ==
      OptimizationOptions::kReorderDataDiscardingOps) {
    if (optimization_options.reorder_data_discarding_ops()) {
      optimization_enabled->insert(kReorderDataDiscardingOpsOpt);
    } else {
      optimization_disabled->insert(kReorderDataDiscardingOpsOpt);
    }
  }
  if (optimization_options.optional_shuffle_and_repeat_fusion_case() ==
      OptimizationOptions::kShuffleAndRepeatFusion) {
    if (optimization_options.shuffle_and_repeat_fusion()) {
      optimization_enabled->insert(kShuffleAndRepeatFusionOpt);
    } else {
      optimization_disabled->insert(kShuffleAndRepeatFusionOpt);
    }
  }
  const bool has_autotune = optimization_options.optional_autotune_case() ==
                            OptimizationOptions::kAutotune;
  const bool has_autotune_buffers =
      optimization_options.optional_autotune_buffers_case() ==
      OptimizationOptions::kAutotuneBuffers;
  if (!(has_autotune && !optimization_options.autotune()) &&
      (has_autotune_buffers && optimization_options.autotune_buffers())) {
    optimization_enabled->insert(kAutotuneBufferSizesOpt);
    optimization_enabled->insert(kDisablePrefetchLegacyAutotuneOpt);
  }
  if (has_autotune && !optimization_options.autotune()) {
    optimization_disabled->insert(kAutotuneBufferSizesOpt);
    optimization_disabled->insert(kDisablePrefetchLegacyAutotuneOpt);
  }
}

void GraphRewritesOptions(const Options& options,
                          std::set<tstring>* optimization_enabled,
                          std::set<tstring>* optimization_disabled,
                          std::set<tstring>* optimization_default) {
  DefaultOptimizationGraphRewrites(options, optimization_enabled,
                                   optimization_disabled, optimization_default);
  if (options.optional_deterministic_case() == Options::kDeterministic) {
    if (options.deterministic()) {
      optimization_disabled->insert(kMakeSloppyOpt);
    } else {
      optimization_enabled->insert(kMakeSloppyOpt);
    }
  }
  if (options.optional_slack_case() == Options::kSlack) {
    if (options.slack()) {
      optimization_enabled->insert(kSlackOpt);
    } else {
      optimization_disabled->insert(kSlackOpt);
    }
  }
}

void GraphRewriteConfigs(const Options& options,
                         std::vector<std::string>* configs) {
  const auto& optimization_options = options.optimization_options();
  const auto& map_vectorization = optimization_options.map_vectorization();
  if (map_vectorization.optional_enabled_case() == MapVectorization::kEnabled &&
      map_vectorization.enabled() &&
      map_vectorization.optional_use_choose_fastest_case() ==
          MapVectorization::kUseChooseFastest) {
    if (map_vectorization.use_choose_fastest()) {
      configs->push_back(absl::StrCat(kMapVectorizationOpt, ":",
                                      kUseChooseFastestOpt, ":true"));
    } else {
      configs->push_back(absl::StrCat(kMapVectorizationOpt, ":",
                                      kUseChooseFastestOpt, ":false"));
    }
  }
  std::vector<tstring> autotune_only_optimizations = {
      kAutotuneBufferSizesOpt, kBatchParallelizationOpt,
      kDisablePrefetchLegacyAutotuneOpt, kEnableGradientDescentOpt,
      kMapParallelizationOpt};

  if (optimization_options.optional_autotune_case() ==
          OptimizationOptions::kAutotune &&
      !optimization_options.autotune()) {
    for (const auto& optimization : autotune_only_optimizations) {
      configs->push_back(
          absl::StrCat(optimization.data(), ":", kAutotuneOpt, ":false"));
    }
  } else {
    for (const auto& optimization : autotune_only_optimizations) {
      configs->push_back(
          absl::StrCat(optimization.data(), ":", kAutotuneOpt, ":true"));
    }
  }
  if (options.slack()) {
    int num_devices = 1;
    if (options.distribute_options().optional_num_devices_case() ==
        DistributeOptions::kNumDevices) {
      num_devices = options.distribute_options().num_devices();
    }
    configs->push_back(
        absl::StrCat(kSlackOpt, ":", kSlackPeriodOpt, ":", num_devices));
  }
}

void GetOptimizationFromOptions(const Options& options,
                                std::vector<tstring>* optimizations_enabled,
                                std::vector<tstring>* optimizations_disabled,
                                std::vector<tstring>* optimizations_default) {
  std::set<tstring> enabled_set;
  std::set<tstring> disabled_set;
  std::set<tstring> default_set;
  GraphRewritesOptions(options, &enabled_set, &disabled_set, &default_set);
  *optimizations_enabled = {enabled_set.begin(), enabled_set.end()};
  *optimizations_disabled = {disabled_set.begin(), disabled_set.end()};
  *optimizations_default = {default_set.begin(), default_set.end()};
}

void GetModelDatasetParams(const Options& options,
                           model::AutotuneAlgorithm* algorithm,
                           bool* cpu_budget, bool* ram_budget) {
  *algorithm = model::AutotuneAlgorithm::HILL_CLIMB;
  if (options.optimization_options().autotune_buffers()) {
    *algorithm = model::AutotuneAlgorithm::GRADIENT_DESCENT;
  }
  *cpu_budget = options.optimization_options().autotune_cpu_budget();
  *ram_budget = options.optimization_options().autotune_ram_budget();
}

bool ShouldUseMaxIntraOpParallelismDataset(const Options& options) {
  return options.threading_options().optional_max_intra_op_parallelism_case() ==
         ThreadingOptions::kMaxIntraOpParallelism;
}

bool ShouldUsePrivateThreadPoolDataset(const Options& options) {
  return options.threading_options().optional_private_threadpool_size_case() ==
         ThreadingOptions::kPrivateThreadpoolSize;
}

bool ShouldUseModelDataset(const Options& options) {
  return options.optimization_options().optional_autotune_case() !=
             OptimizationOptions::kAutotune ||
         options.optimization_options().autotune();
}

bool ShouldUseOptimizeDataset(const Options& options,
                              const std::vector<tstring>& optimizations_enabled,
                              const std::vector<tstring>& optimizations_default,
                              bool has_captured_ref) {
  if (has_captured_ref) {
    if (!optimizations_enabled.empty() || !optimizations_default.empty()) {
      LOG(WARNING)
          << "tf.data graph rewrites are not compatible with reference "
             "variables. The following rewrites will be disabled: "
          << absl::StrJoin(optimizations_enabled, ", ") << ", "
          << absl::StrJoin(optimizations_default, ", ")
          << ". To enable rewrites, use resource variables instead by calling "
             "`tf.enable_resource_variables()` at the start of the program.";
    }
    return false;
  }
  return (options.optimization_options()
                  .optional_apply_default_optimizations_case() !=
              OptimizationOptions::kApplyDefaultOptimizations ||
          options.optimization_options().apply_default_optimizations() ||
          !optimizations_enabled.empty() || !optimizations_default.empty());
}

}  // namespace

FinalizeDatasetOp::FinalizeDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  if (ctx->HasAttr(kHasCapturedRef)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kHasCapturedRef, &has_captured_ref_));
  } else {
    has_captured_ref_ = false;
  }
}

class FinalizeDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, bool has_captured_ref)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        has_captured_ref_(has_captured_ref) {
    input_->Ref();
  }
  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    DCHECK(false) << "FinalizeDatasetOp::Dataset::MakeIteratorInternal is "
                     "not expected to be called because it is supposed to "
                     "forward the iterator to its input dataset(s).";
    LOG(ERROR) << "Datasets of type " << type_string()
               << " forwards its iterator to its input dataset. "
                  "`MakeIteratorInternal` is not implemented.";
    return nullptr;
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    AttrValue has_captured_ref_attr;
    b->BuildAttrValue(has_captured_ref_, &has_captured_ref_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node},
                                     {{kHasCapturedRef, has_captured_ref_attr}},
                                     output));
    return Status::OK();
  }

 private:
  const DatasetBase* input_;
  bool has_captured_ref_;
};

void FinalizeDatasetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
  DatasetBase* input_dataset;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(0), &input_dataset));
  std::vector<DatasetBase*> new_datasets;
  const Options& options = input_dataset->options();
  if (ShouldUseMaxIntraOpParallelismDataset(options)) {
    experimental::MaxIntraOpParallelismDatasetOp::MakeDatasetFromOptions(
        ctx, input_dataset,
        options.threading_options().max_intra_op_parallelism(), output);
    input_dataset = *output;
    new_datasets.push_back(*output);
  }
  if (ShouldUsePrivateThreadPoolDataset(options)) {
    experimental::PrivateThreadPoolDatasetOp::MakeDatasetFromOptions(
        ctx, input_dataset,
        options.threading_options().private_threadpool_size(), output);
    input_dataset = *output;
    new_datasets.push_back(*output);
  }
  if (ShouldUseModelDataset(options)) {
    model::AutotuneAlgorithm algorithm;
    bool cpu_budget;
    bool ram_budget;
    GetModelDatasetParams(options, &algorithm, &cpu_budget, &ram_budget);
    ModelDatasetOp::MakeDatasetFromOptions(ctx, input_dataset, algorithm,
                                           cpu_budget, ram_budget, output);
    input_dataset = *output;
    new_datasets.push_back(*output);
  }
  std::vector<tstring> optimizations_enabled;
  std::vector<tstring> optimizations_disabled;
  std::vector<tstring> optimizations_default;
  GetOptimizationFromOptions(options, &optimizations_enabled,
                             &optimizations_disabled, &optimizations_default);
  if (ShouldUseOptimizeDataset(options, optimizations_enabled,
                               optimizations_default, has_captured_ref_)) {
    std::vector<std::string> optimization_configs;
    GraphRewriteConfigs(options, &optimization_configs);
    OptimizeDatasetOp::MakeDatasetFromOptions(
        ctx, input_dataset, optimizations_enabled, optimizations_disabled,
        optimizations_default, optimization_configs, output);
    input_dataset = *output;
    new_datasets.push_back(*output);
  }

  *output = new Dataset(ctx, input_dataset, has_captured_ref_);
  for (auto new_dataset : new_datasets) {
    new_dataset->Unref();
  }
}

namespace {
REGISTER_KERNEL_BUILDER(Name("FinalizeDataset").Device(DEVICE_CPU).Priority(2),
                        FinalizeDatasetOp);
REGISTER_KERNEL_BUILDER(Name("FinalizeDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_dataset")
                            .HostMemory("handle")
                            .Priority(1),
                        FinalizeDatasetNoopOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
