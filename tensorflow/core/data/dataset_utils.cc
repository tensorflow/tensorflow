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

#include "tensorflow/core/data/dataset_utils.h"

#include <memory>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDelimiter[] = "@@";
constexpr char kComponent[] = "component";
constexpr char kNumElements[] = "num_elements";
constexpr char kNumComponents[] = "num_components";
constexpr char kOutputSize[] = "output_size";
constexpr char kCode[] = "code";
constexpr char kMessage[] = "msg";
constexpr char kOutput[] = "output";

static mutex* get_dataset_experiment_registry_lock() {
  static mutex dataset_experiment_registry_lock(LINKER_INITIALIZED);
  return &dataset_experiment_registry_lock;
}

static absl::flat_hash_map<string, int64>* get_dataset_experiments() {
  static absl::flat_hash_map<string, int64>* experiments =
      new absl::flat_hash_map<string, int64>;
  return experiments;
}

// We assume that all keys are of the form <iterator_prefix>:<name>. We extract
// the iterator name by getting rid of everything post the final colon.
Status GetIteratorName(StringPiece key, string* name) {
  if (!str_util::StartsWith(key, data::kFullNameRandomHex)) {
    return errors::InvalidArgument("Save key: ", key,
                                   " not generated using full_name.");
  }
  std::vector<string> split_keys = str_util::Split(key, data::kPipe);
  if (split_keys.size() != 2) {
    return errors::InvalidArgument("Save key: ", key,
                                   " not generated using full_name.");
  }
  string real_key = split_keys[1];
  const int pos = real_key.rfind(kColon);
  *name = real_key.substr(0, pos);
  return Status::OK();
}

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

void MapVectorizationGraphRewrites(
    const Options& options, absl::flat_hash_set<tstring>* optimization_enabled,
    absl::flat_hash_set<tstring>* optimization_disabled) {
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

void DefaultOptimizationGraphRewrites(
    const Options& options, absl::flat_hash_set<tstring>* optimization_enabled,
    absl::flat_hash_set<tstring>* optimization_disabled,
    absl::flat_hash_set<tstring>* optimization_default) {
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
                          absl::flat_hash_set<tstring>* optimization_enabled,
                          absl::flat_hash_set<tstring>* optimization_disabled,
                          absl::flat_hash_set<tstring>* optimization_default) {
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

}  // namespace

Status WriteElementsToCheckpoint(
    IteratorStateWriter* writer, StringPiece key_prefix,
    const std::vector<std::vector<Tensor>>& elements) {
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(key_prefix, kNumElements, elements.size()));
  for (int i = 0; i < elements.size(); ++i) {
    const std::vector<Tensor>& element = elements[i];
    std::string element_prefix = absl::StrCat(key_prefix, "::", i);
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(element_prefix, kNumComponents, element.size()));
    for (int j = 0; j < elements[i].size(); ++j) {
      TF_RETURN_IF_ERROR(writer->WriteTensor(
          element_prefix, absl::StrCat(kComponent, "[", j, "]"), element[j]));
    }
  }
  return Status::OK();
}

Status ReadElementsFromCheckpoint(IteratorStateReader* reader,
                                  StringPiece key_prefix,
                                  std::vector<std::vector<Tensor>>* elements) {
  int64 num_elements;
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(key_prefix, kNumElements, &num_elements));
  elements->reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    std::string element_prefix = absl::StrCat(key_prefix, "::", i);
    int64 num_components;
    TF_RETURN_IF_ERROR(
        reader->ReadScalar(element_prefix, kNumComponents, &num_components));
    elements->emplace_back();
    std::vector<Tensor>& element = elements->at(i);
    element.reserve(num_components);
    for (int j = 0; j < num_components; ++j) {
      element.emplace_back();
      TF_RETURN_IF_ERROR(reader->ReadTensor(
          element_prefix, absl::StrCat(kComponent, "[", j, "]"),
          &element.back()));
    }
  }
  return Status::OK();
}

std::pair<int64, int64> MaybeOverrideSeeds(std::pair<int64, int64> seeds) {
  if (seeds.first == 0 && seeds.second == 0) {
    return {random::New64(), random::New64()};
  }
  return seeds;
}

Status VerifyTypeMatch(const DataType& expected, const DataType& received,
                       int index) {
  if (expected != received) {
    return errors::InvalidArgument("Data type mismatch at component ", index,
                                   ": expected ", DataTypeString(expected),
                                   " but got ", DataTypeString(received), ".");
  }
  return Status::OK();
}

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(VerifyTypeMatch(expected[i], received[i], i));
  }
  return Status::OK();
}

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const std::vector<Tensor>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(VerifyTypeMatch(expected[i], received[i].dtype(), i));
  }
  return Status::OK();
}

Status VerifyShapeCompatible(const PartialTensorShape& expected,
                             const PartialTensorShape& received, int index) {
  if (!expected.IsCompatibleWith(received)) {
    return errors::InvalidArgument("Incompatible shapes at component ", index,
                                   ": expected ", expected.DebugString(),
                                   " but got ", received.DebugString(), ".");
  }
  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(VerifyShapeCompatible(expected[i], received[i], i));
  }

  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<Tensor>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(
        VerifyShapeCompatible(expected[i], received[i].shape(), i));
  }

  return Status::OK();
}

VariantTensorDataReader::VariantTensorDataReader(
    const std::vector<const tensorflow::VariantTensorData*>& data) {
  for (const auto& d : data) {
    string metadata;
    d->get_metadata(&metadata);
    auto keys = str_util::Split(metadata, kDelimiter, str_util::SkipEmpty());
    const string name = keys[0];
    data_[name] = d;
    map_[name] = std::map<string, size_t>();
    for (size_t i = 1; i < keys.size(); ++i) {
      map_[name][keys[i]] = i - 1;
    }
  }
}

Status VariantTensorDataReader::ReadScalar(StringPiece key, int64* val) const {
  return ReadScalarInternal(key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece key,
                                           tstring* val) const {
  return ReadScalarInternal(key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece key, Tensor* val) const {
  return ReadTensorInternal(key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           int64* val) const {
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           tstring* val) const {
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece name, StringPiece key,
                                           Tensor* val) const {
  return ReadTensorInternal(name, key, val);
}

bool VariantTensorDataReader::Contains(StringPiece key) const {
  string name;
  if (!GetIteratorName(key, &name).ok()) {
    return false;
  }
  return Contains(name, key);
}

bool VariantTensorDataReader::Contains(StringPiece n, StringPiece key) const {
  string name(n);
  auto it = map_.find(name);
  if (it == map_.end()) {
    return false;
  }
  const auto& bucket = it->second;
  return bucket.find(string(key)) != bucket.end();
}

template <typename T>
Status VariantTensorDataReader::ReadScalarInternal(StringPiece key,
                                                   T* val) const {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadTensorInternal(StringPiece key,
                                                   Tensor* val) const {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadTensorInternal(name, key, val);
}

template <typename T>
Status VariantTensorDataReader::ReadScalarInternal(StringPiece n,
                                                   StringPiece key,
                                                   T* val) const {
  string name(n);
  auto it = map_.find(name);
  if (it == map_.end()) {
    return errors::NotFound(name);
  }
  const auto& bucket = it->second;
  auto key_it = bucket.find(string(key));
  if (key_it == bucket.end()) {
    return errors::NotFound(key);
  }
  *val = data_.at(name)->tensors(key_it->second).scalar<T>()();
  return Status::OK();
}

Status VariantTensorDataReader::ReadTensorInternal(StringPiece n,
                                                   StringPiece key,
                                                   Tensor* val) const {
  string name(n);
  auto it = map_.find(name);
  if (it == map_.end()) {
    return errors::NotFound(name);
  }
  const auto& bucket = it->second;
  auto key_it = bucket.find(string(key));
  if (key_it == bucket.end()) {
    return errors::NotFound(key);
  }
  *val = data_.at(name)->tensors(key_it->second);
  return Status::OK();
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key, const int64 val) {
  return WriteScalarInternal(key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key,
                                            const tstring& val) {
  return WriteScalarInternal(key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece key,
                                            const Tensor& val) {
  return WriteTensorInternal(key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const int64 val) {
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const tstring& val) {
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece name, StringPiece key,
                                            const Tensor& val) {
  return WriteTensorInternal(name, key, val);
}

void VariantTensorDataWriter::MaybeFlush() {
  if (is_flushed_) return;
  for (auto& keys : keys_) {
    const string name = keys.first;
    string metadata = name;
    for (size_t i = 0; i < keys_[name].size(); ++i) {
      strings::StrAppend(&metadata, kDelimiter, keys_[name][i]);
    }
    data_[name]->set_metadata(metadata);
  }
  is_flushed_ = true;
}

void VariantTensorDataWriter::Reset() {
  is_flushed_ = false;
  data_.clear();
  keys_.clear();
}

void VariantTensorDataWriter::ReleaseData(
    std::vector<std::unique_ptr<VariantTensorData>>* variants) {
  MaybeFlush();
  for (auto& it : data_) {
    variants->push_back(std::move(it.second));
  }
  Reset();
}

void VariantTensorDataWriter::GetData(
    std::vector<const VariantTensorData*>* variants) {
  MaybeFlush();
  for (auto& it : data_) {
    variants->push_back(it.second.get());
  }
}

template <typename T>
Status VariantTensorDataWriter::WriteScalarInternal(StringPiece key,
                                                    const T& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteScalar after GetData or ReleaseData is called");
  }
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteTensorInternal(StringPiece key,
                                                    const Tensor& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteTensor after GetData or ReleaseData is called");
  }
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteTensorInternal(name, key, val);
}

template <typename T>
Status VariantTensorDataWriter::WriteScalarInternal(StringPiece name,
                                                    StringPiece key,
                                                    const T& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteScalar after GetData or ReleaseData is called");
  }
  Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
  val_t.scalar<T>()() = val;
  return WriteTensorInternal(name, key, val_t);
}

Status VariantTensorDataWriter::WriteTensorInternal(StringPiece n,
                                                    StringPiece key,
                                                    const Tensor& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteTensor after GetData or ReleaseData is called");
  }
  DCHECK_EQ(key.find(kDelimiter), string::npos);
  string name(n);
  if (keys_.count(name) == 0) {
    keys_[name] = std::vector<string>();
  }
  keys_[name].push_back(string(key));
  if (data_.count(name) == 0) {
    data_[name] = absl::make_unique<VariantTensorData>();
    data_[name]->set_type_name("tensorflow::Iterator");
  }
  *(data_[name]->add_tensors()) = val;
  return Status::OK();
}

Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionLibraryDefinition& to_add) {
  for (const auto& fn : to_add.ListFunctionNames()) {
    if (auto found = base->Find(fn)) {
      if (!OpDefEqual(found->signature(), to_add.Find(fn)->signature())) {
        return errors::InvalidArgument("Cannot add function '", fn,
                                       "' because a different function with "
                                       "the same signature already exists.");
      }
      TF_RETURN_IF_ERROR(base->RemoveFunction(fn));
    }
  }
  return base->AddLibrary(to_add);
}

Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionDefLibrary& to_add) {
  for (const auto& fd : to_add.function()) {
    if (auto found = base->Find(fd.signature().name())) {
      if (!OpDefEqual(found->signature(), fd.signature())) {
        return errors::InvalidArgument("Cannot add function '",
                                       fd.signature().name(),
                                       "' because a different function with "
                                       "the same signature already exists.");
      }
      TF_RETURN_IF_ERROR(base->RemoveFunction(fd.signature().name()));
    }
  }
  return base->AddLibrary(to_add);
}

std::function<void(std::function<void()>)> RunnerWithMaxParallelism(
    std::function<void(std::function<void()>)> runner, int max_parallelism) {
  return std::bind(
      [max_parallelism](
          // Note: `runner` is a const reference to avoid copying it.
          const std::function<void(std::function<void()>)>& runner,
          std::function<void()> fn) {
        std::function<void()> scoped_fn = std::bind(
            [max_parallelism](const std::function<void()>& fn) {
              ScopedPerThreadMaxParallelism scope(max_parallelism);
              fn();
            },
            std::move(fn));
        runner(std::move(scoped_fn));
      },
      std::move(runner), std::placeholders::_1);
}

Status DeterminismPolicy::FromString(const std::string& s,
                                     DeterminismPolicy* out) {
  DeterminismPolicy::Type type;
  if (s == DeterminismPolicy::kDeterministic) {
    type = DeterminismPolicy::Type::kDeterministic;
  } else if (s == DeterminismPolicy::kNondeterministic) {
    type = DeterminismPolicy::Type::kNondeterministic;
  } else if (s == DeterminismPolicy::kDefault) {
    type = DeterminismPolicy::Type::kDefault;
  } else {
    return errors::InvalidArgument("Unrecognized determinism policy: ", s);
  }
  *out = DeterminismPolicy(type);
  return Status::OK();
}

DeterminismPolicy::DeterminismPolicy(bool is_deterministic) {
  if (is_deterministic) {
    determinism_ = DeterminismPolicy::Type::kDeterministic;
  } else {
    determinism_ = DeterminismPolicy::Type::kNondeterministic;
  }
}

std::string DeterminismPolicy::String() const {
  switch (determinism_) {
    case DeterminismPolicy::Type::kDeterministic:
      return DeterminismPolicy::kDeterministic;
    case DeterminismPolicy::Type::kNondeterministic:
      return DeterminismPolicy::kNondeterministic;
    case DeterminismPolicy::Type::kDefault:
      return DeterminismPolicy::kDefault;
    default:
      LOG(ERROR) << "Unrecognized determinism value";
      return "Unrecognized";
  }
}

bool MatchesAnyVersion(StringPiece op_prefix, StringPiece op_to_match) {
  if (!absl::StartsWith(op_to_match, op_prefix)) {
    return false;
  }
  if (op_to_match.length() == op_prefix.length()) {
    return true;
  }
  size_t index = op_to_match.length() - 1;
  while (isdigit(op_to_match[index])) {
    index--;
  }
  return (op_to_match[index] == 'V') && (op_prefix.length() == index);
}

std::vector<string> GetExperiments() {
  return GetExperiments(port::JobName(),
                        [](const tstring& str) { return Hash64(str); });
}

std::vector<string> GetExperiments(
    const string& job_name, std::function<uint64(const string&)> hash_func) {
  std::vector<string> experiments;

  if (job_name.empty()) {
    return experiments;
  }

  // Parse the opt-in and opt-out settings.
  const char* opt_ins_raw_cs = std::getenv("TF_DATA_EXPERIMENT_OPT_IN");
  const char* opt_outs_raw_cs = std::getenv("TF_DATA_EXPERIMENT_OPT_OUT");
  string opt_ins_raw;
  if (opt_ins_raw_cs != nullptr) {
    opt_ins_raw = string(opt_ins_raw_cs);
  }
  string opt_outs_raw;
  if (opt_outs_raw_cs != nullptr) {
    opt_outs_raw = string(opt_outs_raw_cs);
  }

  // Identify opted out experiments.
  absl::flat_hash_map<string, int64> live_experiments =
      DatasetExperimentRegistry::Experiments();
  absl::flat_hash_set<string> opt_outs;
  if (opt_outs_raw == "all") {
    for (const auto& pair : live_experiments) {
      opt_outs.insert(pair.first);
    }
  } else {
    for (const auto& experiment :
         str_util::Split(opt_outs_raw, ',', str_util::SkipEmpty())) {
      opt_outs.insert(experiment);
    }
  }

  // Include opted in experiments unless they are opted out.
  absl::flat_hash_set<string> experiments_set;
  if (opt_ins_raw == "all") {
    for (const auto& pair : live_experiments) {
      auto experiment = pair.first;
      if (std::find(opt_outs.begin(), opt_outs.end(), experiment) ==
          opt_outs.end()) {
        experiments_set.insert(experiment);
      }
    }
  } else {
    for (const auto& experiment :
         str_util::Split(opt_ins_raw, ',', str_util::SkipEmpty())) {
      if (std::find(opt_outs.begin(), opt_outs.end(), experiment) ==
          opt_outs.end()) {
        experiments_set.insert(experiment);
      }
    }
  }

  // Stochastically include live experiments unless they are opted out.
  for (const auto& pair : live_experiments) {
    auto& experiment = pair.first;
    if ((hash_func(strings::StrCat(job_name, experiment)) % 100 <
         pair.second) &&
        (std::find(opt_outs.begin(), opt_outs.end(), experiment) ==
         opt_outs.end())) {
      experiments_set.insert(experiment);
    }
  }

  experiments.insert(experiments.end(), experiments_set.begin(),
                     experiments_set.end());
  return experiments;
}

void LogAndRecordExperiments(const std::vector<string>& experiments) {
  if (!experiments.empty()) {
    VLOG(1) << "The input pipeline is subject to tf.data experiments. "
               "Please see `go/tf-data-experiments` for more details.";
  }
  for (auto& experiment : experiments) {
    VLOG(1) << "The experiment \"" << experiment << "\" is applied.";
    metrics::RecordTFDataExperiment(experiment);
  }
}

void GetOptimizations(const Options& options,
                      std::vector<tstring>* optimizations_enabled,
                      std::vector<tstring>* optimizations_disabled,
                      std::vector<tstring>* optimizations_default) {
  absl::flat_hash_set<tstring> enabled_set;
  absl::flat_hash_set<tstring> disabled_set;
  absl::flat_hash_set<tstring> default_set;
  GraphRewritesOptions(options, &enabled_set, &disabled_set, &default_set);
  *optimizations_enabled = {enabled_set.begin(), enabled_set.end()};
  *optimizations_disabled = {disabled_set.begin(), disabled_set.end()};
  *optimizations_default = {default_set.begin(), default_set.end()};
}

std::vector<tstring> SelectOptimizations(
    const std::vector<string>& experiments,
    const std::vector<tstring>& optimizations_enabled,
    const std::vector<tstring>& optimizations_disabled,
    const std::vector<tstring>& optimizations_default) {
  absl::flat_hash_set<tstring> optimizations_set;

  // Add the enabled and default optimizations.
  optimizations_set.insert(optimizations_enabled.begin(),
                           optimizations_enabled.end());
  optimizations_set.insert(optimizations_default.begin(),
                           optimizations_default.end());

  // Add experiments unless they correspond to a disabled optimization.
  for (auto& experiment : experiments) {
    if (std::find(optimizations_disabled.begin(), optimizations_disabled.end(),
                  experiment) == optimizations_disabled.end()) {
      optimizations_set.insert(experiment);
    }
  }

  std::vector<tstring> optimizations;
  optimizations.insert(optimizations.end(), optimizations_set.begin(),
                       optimizations_set.end());
  return optimizations;
}

void StripDevicePlacement(FunctionDefLibrary* library) {
  for (auto& function : (*library->mutable_function())) {
    for (auto& node : (*function.mutable_node_def())) {
      if (!node.device().empty()) {
        *node.mutable_device() = "";
      }
    }
  }
}

Status CopyPartialBatch(int64 num_elements, const Tensor& value,
                        Tensor* output) {
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

Status ReadBatch(int64 batch_size, const string& iterator_prefix,
                 const string& batch_prefix, IteratorContext* ctx,
                 IteratorStateReader* reader, std::vector<Tensor>* batch) {
  int64 output_size;
  TF_RETURN_IF_ERROR(reader->ReadScalar(
      FullName(iterator_prefix,
               strings::StrCat(batch_prefix, "_", kOutputSize)),
      &output_size));
  batch->reserve(output_size);
  for (int i = 0; i < output_size; i++) {
    Tensor t;
    TF_RETURN_IF_ERROR(reader->ReadTensor(
        FullName(iterator_prefix,
                 strings::StrCat(batch_prefix, "_", kOutput, "_", i)),
        &t));
    // If the batch was not full, we may have stored only the relevant slice.
    // Since tensors in `BatchResult.output` are expected to have the leading
    // dimension of size batch_size, we build a larger tensor and copy the slice
    // read from the checkpoint into it.
    if (t.dim_size(0) < batch_size) {
      TensorShape component_shape(t.shape());
      component_shape.set_dim(0, batch_size);
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      Tensor new_t(ctx->allocator(attr), t.dtype(), component_shape);
      TF_RETURN_IF_ERROR(CopyPartialBatch(t.dim_size(0), t, &new_t));
      batch->emplace_back(std::move(new_t));
    } else {
      batch->emplace_back(std::move(t));
    }
  }
  return Status::OK();
}

Status WriteBatch(int64 batch_size, int64 num_elements,
                  const string& iterator_prefix, const string& batch_prefix,
                  IteratorStateWriter* writer, std::vector<Tensor>* batch) {
  TF_RETURN_IF_ERROR(writer->WriteScalar(
      FullName(iterator_prefix,
               strings::StrCat(batch_prefix, "_", kOutputSize)),
      batch->size()));
  for (int i = 0; i < batch->size(); i++) {
    // If the batch is not full, we only store the first `num_elements` values.
    // The rest of the batch tensor is *uninitialized* and accessing that will
    // raise msan errors.
    if (num_elements < batch_size) {
      TF_RETURN_IF_ERROR(writer->WriteTensor(
          FullName(iterator_prefix,
                   strings::StrCat(batch_prefix, "_", kOutput, "_", i)),
          (*batch)[i].Slice(0, num_elements)));
    } else {
      TF_RETURN_IF_ERROR(writer->WriteTensor(
          FullName(iterator_prefix,
                   strings::StrCat(batch_prefix, "_", kOutput, "_", i)),
          (*batch)[i]));
    }
  }
  return Status::OK();
}

Status ReadStatus(const string& iterator_prefix, const string& prefix,
                  IteratorStateReader* reader, Status* status) {
  int64 code_int;
  TF_RETURN_IF_ERROR(reader->ReadScalar(
      FullName(iterator_prefix, strings::StrCat(prefix, "_", kCode)),
      &code_int));
  error::Code code = static_cast<error::Code>(code_int);

  if (code != error::Code::OK) {
    tstring error_message;
    TF_RETURN_IF_ERROR(reader->ReadScalar(
        FullName(iterator_prefix, strings::StrCat(prefix, "_", kMessage)),
        &error_message));
    *status = Status(code, error_message);
  } else {
    *status = Status::OK();
  }
  return Status::OK();
}

Status WriteStatus(const string& iterator_prefix, const string& prefix,
                   const Status& status, IteratorStateWriter* writer) {
  TF_RETURN_IF_ERROR(writer->WriteScalar(
      FullName(iterator_prefix, strings::StrCat(prefix, "_", kCode)),
      static_cast<int64>(status.code())));
  if (!status.ok()) {
    TF_RETURN_IF_ERROR(writer->WriteScalar(
        FullName(iterator_prefix, strings::StrCat(prefix, "_", kMessage)),
        status.error_message()));
  }
  return Status::OK();
}

Status ProcessBatch(int64 batch_size, int64 num_elements, bool drop_remainder,
                    const Status& status, IteratorContext* ctx,
                    std::vector<Tensor>* output, bool* end_of_sequence,
                    std::vector<Tensor>* batch) {
  if (num_elements == 0) {
    if (status.ok() || errors::IsOutOfRange(status)) {
      *end_of_sequence = true;
      return Status::OK();
    } else {
      *end_of_sequence = false;
      return status;
    }
  }
  if (!status.ok() && !errors::IsOutOfRange(status)) {
    *end_of_sequence = false;
    return status;
  }
  if (num_elements < batch_size) {
    if (drop_remainder) {
      *end_of_sequence = true;
      return Status::OK();
    }
    for (size_t i = 0; i < batch->size(); ++i) {
      TensorShape component_shape((*batch)[i].shape());
      component_shape.set_dim(0, num_elements);
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      output->emplace_back(ctx->allocator(attr), (*batch)[i].dtype(),
                           component_shape);
      if (!output->back().IsInitialized()) {
        return errors::ResourceExhausted(
            "Failed to allocate memory for the batch of component ", i);
      }
      TF_RETURN_IF_ERROR(
          CopyPartialBatch(num_elements, (*batch)[i], &output->back()));
    }
  } else {
    *output = std::move(*batch);
  }
  *end_of_sequence = false;
  return Status::OK();
}

Status CopyBatch(bool parallel_copy, IteratorContext* ctx,
                 std::vector<Tensor>* out_tensors,
                 std::vector<std::vector<Tensor>>* batch_elements) {
  const size_t num_tuple_components = (*batch_elements)[0].size();
  out_tensors->reserve(num_tuple_components);
  const int64 num_batch_elements = batch_elements->size();
  for (size_t component_index = 0; component_index < num_tuple_components;
       ++component_index) {
    const Tensor& first_element = (*batch_elements)[0][component_index];
    TensorShape batch_component_shape({num_batch_elements});
    // NOTE(mrry): Copy the shape of the first element here, because
    // `first_element.shape()` will become undefined after the 0th batch element
    // is moved into the output batch.
    TensorShape first_element_shape(first_element.shape());
    batch_component_shape.AppendShape(first_element_shape);
    out_tensors->emplace_back(ctx->allocator({}), first_element.dtype(),
                              batch_component_shape);
    if (!out_tensors->back().IsInitialized()) {
      return errors::ResourceExhausted(
          "Failed to allocate memory for the batch of component ",
          component_index);
    }
    Tensor& batch_component = out_tensors->back();
    // Build the output tuple component by copying one slice from each input
    // element in the batch.
    auto copy_element_fn = [component_index, &batch_elements,
                            &batch_component](int index) {
      TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(
          std::move((*batch_elements)[index][component_index]),
          &batch_component, index));
      return Status::OK();
    };
    Status status;
    std::unique_ptr<BlockingCounter> counter;
    std::unique_ptr<mutex> status_mu;
    if (TF_PREDICT_FALSE(parallel_copy)) {
      counter = std::make_unique<BlockingCounter>(num_batch_elements);
      status_mu = std::make_unique<mutex>();
    }
    for (size_t i = 0; i < num_batch_elements; ++i) {
      if ((*batch_elements)[i][component_index].shape() !=
          first_element_shape) {
        return errors::InvalidArgument(
            "Cannot batch tensors with different shapes in component ",
            component_index, ". First element had shape ",
            first_element_shape.DebugString(), " and element ", i,
            " had shape ",
            (*batch_elements)[i][component_index].shape().DebugString(), ".");
      }
      if (TF_PREDICT_FALSE(parallel_copy)) {
        (*ctx->runner())(
            [i, &status, &status_mu, &counter, &copy_element_fn]() {
              Status s = copy_element_fn(i);
              {
                mutex_lock l(*status_mu);
                status.Update(s);
              }
              counter->DecrementCount();
            });
      } else {
        status.Update(copy_element_fn(i));
      }
    }
    if (TF_PREDICT_FALSE(parallel_copy)) {
      counter->Wait();
    }
    TF_RETURN_IF_ERROR(status);
  }
  return Status::OK();
}

void CreateGraphRewriteConfigs(const Options& options,
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

bool ShouldConfigureMaxIntraOpParallelism(const Options& options) {
  return options.threading_options().optional_max_intra_op_parallelism_case() ==
         ThreadingOptions::kMaxIntraOpParallelism;
}

bool ShouldUsePrivateThreadPool(const Options& options) {
  return options.threading_options().optional_private_threadpool_size_case() ==
         ThreadingOptions::kPrivateThreadpoolSize;
}

bool ShouldUseAutotuning(const Options& options) {
  return options.optimization_options().optional_autotune_case() !=
             OptimizationOptions::kAutotune ||
         options.optimization_options().autotune();
}

bool ShouldApplyOptimizations(
    const Options& options, const std::vector<tstring>& optimizations_enabled,
    const std::vector<tstring>& optimizations_default) {
  return (options.optimization_options()
                  .optional_apply_default_optimizations_case() !=
              OptimizationOptions::kApplyDefaultOptimizations ||
          options.optimization_options().apply_default_optimizations() ||
          !optimizations_enabled.empty() || !optimizations_default.empty());
}

// static
void DatasetExperimentRegistry::Register(const string& experiment,
                                         int64 rollout_pct) {
  mutex_lock l(*get_dataset_experiment_registry_lock());
  get_dataset_experiments()->insert(std::make_pair(experiment, rollout_pct));
}

// static
absl::flat_hash_map<string, int64> DatasetExperimentRegistry::Experiments() {
  mutex_lock l(*get_dataset_experiment_registry_lock());
  return *get_dataset_experiments();
}

namespace {

REGISTER_DATASET_EXPERIMENT("enable_gradient_descent", 0);

}
}  // namespace data
}  // namespace tensorflow
