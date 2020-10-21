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

#include "tensorflow/core/kernels/data/dataset_utils.h"

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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {
constexpr char kDelimiter[] = "@@";
constexpr char kComponent[] = "component";
constexpr char kNumElements[] = "num_elements";
constexpr char kNumComponents[] = "num_components";
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

Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    std::function<void()> register_fn,
                                    std::function<void()>* deregister_fn) {
  if (cancellation_manager) {
    CancellationToken token = cancellation_manager->get_cancellation_token();
    if (!cancellation_manager->RegisterCallback(token,
                                                std::move(register_fn))) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [cancellation_manager, token]() {
      cancellation_manager->DeregisterCallback(token);
    };
  } else {
    VLOG(1) << "Cancellation manager is not set. Cancellation callback will "
               "not be registered.";
    *deregister_fn = []() {};
  }
  return Status::OK();
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

namespace {

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

}  // namespace

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

std::vector<tstring> SelectOptimizations(
    const string& job_name,
    const absl::flat_hash_map<string, uint64>& live_experiments,
    const std::vector<tstring>& optimizations_enabled,
    const std::vector<tstring>& optimizations_disabled,
    const std::vector<tstring>& optimizations_default,
    std::function<uint64(const string&)> hash_func) {
  std::vector<tstring> optimizations;
  if (job_name.empty()) {
    // If `job_name` is empty, apply the enabled and default optimizations
    // directly.
    optimizations.insert(optimizations.end(), optimizations_enabled.begin(),
                         optimizations_enabled.end());
    optimizations.insert(optimizations.end(), optimizations_default.begin(),
                         optimizations_default.end());
    return optimizations;
  }

  // If `job_name` is non-empty, we determine which optimizations to apply to
  // this job based on the enable/disable settings from tf.data.Options, the
  // opt in/out settings from environment variables, and rollout condition from
  // `live_experiments`.
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

  // Creates a set of optimizations.
  absl::flat_hash_set<tstring> optimizations_set;

  // Creates the opt in and opt out settings.
  std::vector<string> opt_ins, opt_outs;
  if (opt_ins_raw == "all") {
    for (auto& pair : live_experiments) {
      opt_ins.push_back(pair.first);
    }
  } else {
    opt_ins = str_util::Split(opt_ins_raw, ',', str_util::SkipEmpty());
  }
  if (opt_outs_raw == "all") {
    for (auto& pair : live_experiments) {
      opt_outs.push_back(pair.first);
    }
  } else {
    opt_outs = str_util::Split(opt_outs_raw, ',', str_util::SkipEmpty());
  }

  // Checks if the opt in and opt out experiments are live experiments.
  for (auto& optimization : opt_ins) {
    if (live_experiments.find(optimization) == live_experiments.end()) {
      LOG(WARNING) << "The experiment \"" << optimization
                   << "\" is opted in but it is not a live experiment.";
    }
  }
  for (auto& optimization : opt_outs) {
    if (live_experiments.find(optimization) == live_experiments.end()) {
      LOG(WARNING) << "The experiment \"" << optimization
                   << "\" is opted out but it is not a live experiment.";
    }
  }

  // Checks if the opt in settings conflict with opt out settings.
  for (auto& optimization : opt_ins) {
    if (std::find(opt_outs.begin(), opt_outs.end(), optimization) !=
        opt_outs.end()) {
      LOG(WARNING) << "The experiment \"" << optimization
                   << "\" is set in both \"TF_DATA_EXPERIMENT_OPT_IN\" and "
                      "\"TF_DATA_EXPERIMENT_OPT_OUT\". Unless the experiment "
                      "corresponds to an explicitly enabled optimization, it "
                      "is not applied.";
    }
  }

  // Checks if the enable/disable settings from tf.data.Options conflict with
  // user opt in/out settings. In which case we assume tf.data.Options settings
  // have higher priority to overwrite.
  for (auto& optimization : optimizations_enabled) {
    if (std::find(opt_outs.begin(), opt_outs.end(), optimization) !=
        opt_outs.end()) {
      LOG(WARNING) << "The optimization \"" << optimization
                   << "\" is opt out, but is still applied since"
                      " it is enabled through tf.data.Options.";
    }
  }
  for (auto& optimization : optimizations_disabled) {
    if (std::find(opt_ins.begin(), opt_ins.end(), optimization) !=
        opt_ins.end()) {
      LOG(WARNING) << "The optimization \"" << optimization
                   << "\" is opt in, but is not applied since"
                      " it is disabled through tf.data.Options.";
    }
  }

  // Add the enabled optimizations.
  optimizations_set.insert(optimizations_enabled.begin(),
                           optimizations_enabled.end());

  // Add the default optimizations that are not explicitly opted out.
  for (auto& optimization : optimizations_default) {
    if (std::find(opt_outs.begin(), opt_outs.end(), optimization) ==
        opt_outs.end()) {
      optimizations_set.insert(optimization);
    }
  }

  // Add the live experiments stochastically if they are neither opted in nor
  // opted out.
  for (auto& pair : live_experiments) {
    string experiment = pair.first;
    // Skip experiments that are explicitly opted out.
    if (std::find(opt_outs.begin(), opt_outs.end(), experiment) !=
        opt_outs.end()) {
      continue;
    }
    // Skip experiments whose transformations are explicitly disabled.
    if (std::find(optimizations_disabled.begin(), optimizations_disabled.end(),
                  experiment) != optimizations_disabled.end()) {
      continue;
    }
    // Apply experiments that are explicitly opted in.
    if (std::find(opt_ins.begin(), opt_ins.end(), experiment) !=
        opt_ins.end()) {
      optimizations_set.insert(experiment);
      continue;
    }
    // Otherwise, apply experiment stochastically based on job name and
    // experiment roll out percentage.
    if (hash_func(strings::StrCat(job_name, experiment)) % 100 < pair.second) {
      optimizations_set.insert(experiment);
    }
  }

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

}  // namespace data
}  // namespace tensorflow
