/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/serialization_utils.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/data/compression_utils.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDelimiter[] = "@@";
constexpr char kComponent[] = "component";
constexpr char kNumComponents[] = "num_components";
constexpr char kNumElements[] = "num_elements";
constexpr char kIsDataset[] = ".is_dataset";
constexpr char kIteratorVariantTypeName[] = "tensorflow::Iterator";
constexpr char kOutputNode[] = ".output_node";

absl::Status FromGraphDef(
    FunctionLibraryRuntime* flr, const GraphDef& graph_def,
    const std::vector<std::pair<string, Tensor>>& input_list,
    const string& output_node, Tensor* result) {
  FunctionLibraryRuntime* cloned_flr = nullptr;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = nullptr;
  std::unique_ptr<FunctionLibraryDefinition> lib_def = nullptr;
  TF_RETURN_IF_ERROR(flr->Clone(&lib_def, &pflr, &cloned_flr, true));
  TF_RETURN_IF_ERROR(AddToFunctionLibrary(lib_def.get(), graph_def.library()));
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
  std::vector<Tensor> outputs;
  GraphRunner graph_runner(cloned_flr->device());
  TF_RETURN_IF_ERROR(graph_runner.Run(&graph, cloned_flr, input_list,
                                      {output_node}, &outputs));
  *result = outputs[0];
  return absl::OkStatus();
}

// FindStatefulOps searches `graph_def` for all of its stateful ops storing
// their names in `stateful_op_names`.
absl::Status FindStatefulOps(const GraphDef& graph_def,
                             std::vector<string>* stateful_op_names) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), graph_def.library());

  // Iterate over all nodes in the graph.
  for (const auto& node : graph_def.node()) {
    // Each Dataset graph has a _Retval op in the end which is marked stateful
    if (node.op() == FunctionLibraryDefinition::kRetOp) continue;
    if (!IsNodeStateful(lib_def, node).ok()) {
      stateful_op_names->push_back(node.op());
    }
  }

  // Iterate over all functions.
  for (const auto& fdef : graph_def.library().function()) {
    if (!fdef.signature().is_stateful()) continue;
    for (const auto& node : fdef.node_def()) {
      if (!IsNodeStateful(lib_def, node).ok()) {
        stateful_op_names->push_back(
            absl::StrCat(node.op(), " in function: ", fdef.signature().name()));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ReadElementsFromCheckpoint(
    IteratorContext* ctx, IteratorStateReader* reader,
    absl::string_view key_prefix, std::vector<std::vector<Tensor>>* elements) {
  int64_t num_elements;
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(key_prefix, kNumElements, &num_elements));
  DCHECK(elements->empty());
  elements->reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    std::string element_prefix = absl::StrCat(key_prefix, "::", i);
    int64_t num_components;
    TF_RETURN_IF_ERROR(
        reader->ReadScalar(element_prefix, kNumComponents, &num_components));
    elements->emplace_back();
    std::vector<Tensor>& element = elements->at(i);
    element.reserve(num_components);
    for (int j = 0; j < num_components; ++j) {
      element.emplace_back();
      TF_RETURN_IF_ERROR(reader->ReadTensor(
          ctx->flr(), element_prefix, absl::StrCat(kComponent, "[", j, "]"),
          &element.back()));
    }
  }
  return absl::OkStatus();
}

absl::Status WriteElement(IteratorStateWriter* writer,
                          absl::string_view key_prefix,
                          const std::vector<std::vector<Tensor>>& elements,
                          int64_t index) {
  const std::vector<Tensor>& element = elements[index];
  std::string element_prefix = absl::StrCat(key_prefix, "::", index);
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(element_prefix, kNumComponents, element.size()));
  for (int j = 0; j < element.size(); ++j) {
    TF_RETURN_IF_ERROR(writer->WriteTensor(
        element_prefix, absl::StrCat(kComponent, "[", j, "]"), element[j]));
  }
  return absl::OkStatus();
}

absl::Status WriteElementsToCheckpoint(
    IteratorStateWriter* writer, absl::string_view key_prefix,
    const std::vector<std::vector<Tensor>>& elements) {
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(key_prefix, kNumElements, elements.size()));
  for (int i = 0; i < elements.size(); ++i) {
    TF_RETURN_IF_ERROR(WriteElement(writer, key_prefix, elements, i));
  }
  return absl::OkStatus();
}

absl::Status UpdateCheckpointElements(
    IteratorStateWriter* writer, absl::string_view key_prefix,
    const std::vector<std::vector<Tensor>>& elements,
    const absl::flat_hash_set<int64_t>& checkpoint_indices) {
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(key_prefix, kNumElements, elements.size()));
  for (int64_t i : checkpoint_indices) {
    TF_RETURN_IF_ERROR(WriteElement(writer, key_prefix, elements, i));
  }
  return absl::OkStatus();
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

absl::Status VariantTensorDataReader::ReadScalar(absl::string_view key,
                                                 int64_t* val) const {
  string prefix;
  TF_RETURN_IF_ERROR(ExtractIteratorPrefix(key, &prefix));
  return ReadScalar(prefix, key, val);
}

absl::Status VariantTensorDataReader::ReadScalar(absl::string_view name,
                                                 absl::string_view key,
                                                 int64_t* val) const {
  return ReadScalarInternal(name, key, val);
}

absl::Status VariantTensorDataReader::ReadScalar(absl::string_view key,
                                                 tstring* val) const {
  string prefix;
  TF_RETURN_IF_ERROR(ExtractIteratorPrefix(key, &prefix));
  return ReadScalar(prefix, key, val);
}

absl::Status VariantTensorDataReader::ReadScalar(absl::string_view name,
                                                 absl::string_view key,
                                                 tstring* val) const {
  return ReadScalarInternal(name, key, val);
}

absl::Status VariantTensorDataReader::ReadTensor(absl::string_view key,
                                                 Tensor* val) const {
  string prefix;
  TF_RETURN_IF_ERROR(ExtractIteratorPrefix(key, &prefix));
  return ReadTensor(prefix, key, val);
}

absl::Status VariantTensorDataReader::ReadTensor(FunctionLibraryRuntime* flr,
                                                 absl::string_view key,
                                                 Tensor* val) const {
  string prefix;
  TF_RETURN_IF_ERROR(ExtractIteratorPrefix(key, &prefix));
  return ReadTensorInternal(flr, prefix, key, val);
}

absl::Status VariantTensorDataReader::ReadTensor(absl::string_view name,
                                                 absl::string_view key,
                                                 Tensor* val) const {
  return ReadTensor(/*flr=*/nullptr, name, key, val);
}

absl::Status VariantTensorDataReader::ReadTensor(FunctionLibraryRuntime* flr,
                                                 absl::string_view name,
                                                 absl::string_view key,
                                                 Tensor* val) const {
  return ReadTensorInternal(flr, name, key, val);
}

bool VariantTensorDataReader::Contains(absl::string_view key) const {
  string prefix;
  if (!ExtractIteratorPrefix(key, &prefix).ok()) {
    return false;
  }
  return Contains(prefix, key);
}

bool VariantTensorDataReader::Contains(absl::string_view n,
                                       absl::string_view key) const {
  string name(n);
  auto it = map_.find(name);
  if (it == map_.end()) {
    return false;
  }
  const auto& bucket = it->second;
  return bucket.find(string(key)) != bucket.end();
}

template <typename T>
absl::Status VariantTensorDataReader::ReadScalarInternal(absl::string_view n,
                                                         absl::string_view key,
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
  return absl::OkStatus();
}

absl::Status VariantTensorDataReader::ReadTensorInternal(
    FunctionLibraryRuntime* flr, absl::string_view n, absl::string_view key,
    Tensor* val) const {
  if (Contains(n, absl::StrCat(key, kIsDataset))) {
    return ReadDatasetInternal(flr, n, key, val);
  }
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
  return absl::OkStatus();
}

absl::Status VariantTensorDataReader::ReadDatasetInternal(
    FunctionLibraryRuntime* flr, absl::string_view n, absl::string_view key,
    Tensor* val) const {
  if (flr == nullptr) {
    return errors::Internal(
        "Function library runtime is needed to restore a dataset.");
  }
  tstring output_node, serialized_graph_def;
  TF_RETURN_IF_ERROR(
      ReadScalar(n, absl::StrCat(key, kOutputNode), &output_node));
  TF_RETURN_IF_ERROR(ReadScalar(n, key, &serialized_graph_def));
  GraphDef graph_def;
  graph_def.ParseFromString(serialized_graph_def);
  TF_RETURN_IF_ERROR(FromGraphDef(flr, graph_def, {}, output_node, val));
  return absl::OkStatus();
}

std::map<string, Tensor> VariantTensorDataReader::ReadAllTensors() {
  std::map<string, Tensor> result;
  for (const auto& entry : map_) {
    string key1 = entry.first;
    for (const auto& inner : entry.second) {
      string key2 = inner.first;
      size_t index = inner.second;
      result[absl::StrCat(key1, kDelimiter, key2)] =
          data_[key1]->tensors(index);
    }
  }
  return result;
}

absl::Status VariantTensorDataWriter::WriteScalar(absl::string_view key,
                                                  const int64_t val) {
  string prefix;
  TF_RETURN_IF_ERROR(ExtractIteratorPrefix(key, &prefix));
  return WriteScalar(prefix, key, val);
}

absl::Status VariantTensorDataWriter::WriteScalar(absl::string_view name,
                                                  absl::string_view key,
                                                  const int64_t val) {
  return WriteScalarInternal(name, key, val);
}

absl::Status VariantTensorDataWriter::WriteScalar(absl::string_view key,
                                                  const tstring& val) {
  string prefix;
  TF_RETURN_IF_ERROR(ExtractIteratorPrefix(key, &prefix));
  return WriteScalar(prefix, key, val);
}

absl::Status VariantTensorDataWriter::WriteScalar(absl::string_view name,
                                                  absl::string_view key,
                                                  const tstring& val) {
  return WriteScalarInternal(name, key, val);
}

absl::Status VariantTensorDataWriter::WriteTensor(absl::string_view key,
                                                  const Tensor& val) {
  string prefix;
  TF_RETURN_IF_ERROR(ExtractIteratorPrefix(key, &prefix));
  return WriteTensor(prefix, key, val);
}

absl::Status VariantTensorDataWriter::WriteTensor(absl::string_view name,
                                                  absl::string_view key,
                                                  const Tensor& val) {
  return WriteTensorInternal(name, key, val);
}

void VariantTensorDataWriter::MaybeFlush() {
  if (is_flushed_) return;
  for (auto& keys : keys_) {
    const string name = keys.first;
    string metadata = name;
    for (size_t i = 0; i < keys_[name].size(); ++i) {
      absl::StrAppend(&metadata, kDelimiter, keys_[name][i]);
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
absl::Status VariantTensorDataWriter::WriteScalarInternal(
    absl::string_view name, absl::string_view key, const T& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteScalar after GetData or ReleaseData is called");
  }
  Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
  val_t.scalar<T>()() = val;
  return WriteTensorInternal(name, key, val_t);
}

absl::Status VariantTensorDataWriter::WriteTensorInternal(absl::string_view n,
                                                          absl::string_view key,
                                                          const Tensor& val) {
  DatasetBase* dataset;
  if (GetDatasetFromVariantTensor(val, &dataset).ok()) {
    return WriteDatasetInternal(n, key, dataset);
  }
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
    data_[name] = std::make_unique<VariantTensorData>();
    data_[name]->set_type_name("tensorflow::Iterator");
  }
  *(data_[name]->add_tensors()) = val;
  return absl::OkStatus();
}

absl::Status VariantTensorDataWriter::WriteDatasetInternal(
    absl::string_view n, absl::string_view key, const DatasetBase* dataset) {
  GraphDef graph_def;
  SerializationContext ctx((SerializationContext::Params()));
  TF_RETURN_IF_ERROR(AsGraphDef(dataset, std::move(ctx), &graph_def));
  string output_node;
  for (const auto& node : graph_def.node()) {
    if (node.op() == kRetvalOp) {
      output_node = node.input(0);
      break;
    }
  }
  string result;
  graph_def.SerializeToString(&result);
  TF_RETURN_IF_ERROR(WriteScalar(n, absl::StrCat(key, kIsDataset), ""));
  TF_RETURN_IF_ERROR(
      WriteScalar(n, absl::StrCat(key, kOutputNode), output_node));
  TF_RETURN_IF_ERROR(WriteScalar(n, key, result));
  return absl::OkStatus();
}

std::string IteratorStateVariant::TypeName() {
  return kIteratorVariantTypeName;
}

IteratorStateVariant::IteratorStateVariant(const IteratorStateVariant& other) {
  if (other.data_) {
    data_ = std::make_unique<VariantTensorData>(*other.data_);
  }
}

absl::Status IteratorStateVariant::InitializeFromVariantData(
    std::unique_ptr<VariantTensorData> data) {
  data_ = std::move(data);
  return absl::OkStatus();
}

void IteratorStateVariant::Encode(VariantTensorData* data) const {
  CompressedElement compressed_tensors;
  absl::Status s = CompressElement(data_->tensors(), &compressed_tensors);
  if (!s.ok()) {
    LOG(WARNING) << "Failed to compress iterator state variant: " << s;
    *data = *data_;
    return;
  }

  data->set_type_name(TypeName());
  data->set_metadata(data_->metadata_string());
  Tensor tensor(DT_VARIANT, TensorShape({}));
  tensor.scalar<Variant>()() = std::move(compressed_tensors);
  *data->add_tensors() = std::move(tensor);
}

bool IteratorStateVariant::Decode(VariantTensorData data) {
  if (data.type_name() != TypeName()) {
    return false;
  }

  const CompressedElement* compressed = GetCompressedElement(data);
  if (!compressed) {
    data_ = std::make_unique<VariantTensorData>(std::move(data));
    return true;
  }

  std::vector<Tensor> tensors;
  absl::Status s = UncompressElement(*compressed, &tensors);
  if (!s.ok()) {
    LOG(WARNING) << "Failed to uncompress iterator state variant: " << s;
    data_ = std::make_unique<VariantTensorData>(std::move(data));
    return true;
  }

  data_ = std::make_unique<VariantTensorData>();
  data_->set_type_name(TypeName());
  data_->set_metadata(std::move(data.metadata_string()));
  for (auto& tensor : tensors) {
    *data_->add_tensors() = std::move(tensor);
  }
  return true;
}

const CompressedElement* IteratorStateVariant::GetCompressedElement(
    const VariantTensorData& data) {
  bool should_uncompress =
      data.tensors_size() == 1 &&
      TensorShapeUtils::IsScalar(data.tensors(0).shape()) &&
      data.tensors(0).dtype() == DT_VARIANT;
  if (!should_uncompress) {
    return nullptr;
  }

  const Variant& variant = data.tensors(0).scalar<Variant>()();
  return variant.get<CompressedElement>();
}

std::string IteratorStateVariant::DebugString() const {
  if (data_) {
    return absl::StrCat("IteratorStateVariant<", data_->DebugString(), ">");
  } else {
    return absl::StrCat("IteratorStateVariant<empty>");
  }
}

// Register the reader class in the global variant decode_fn registry
// so that a Variant containing a serialized representation of iterator state
// can be decoded using DecodeUnaryVariant. If we don't do this we will need
// to manually decode the returned Variant using MaybeDecodeAndCopy in
// DeserializeIteratorOp which is not recommended.
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(IteratorStateVariant,
                                       kIteratorVariantTypeName);

absl::Status AsGraphDefForRewrite(
    OpKernelContext* ctx, const DatasetBase* input,
    std::vector<std::pair<string, Tensor>>* input_list, GraphDef* result,
    string* dataset_node) {
  SerializationContext::Params params(ctx);
  params.input_list = input_list;
  params.external_state_policy = ExternalStatePolicy::POLICY_IGNORE;
  params.is_graph_rewrite = true;
  SerializationContext serialization_ctx(params);
  TF_RETURN_IF_ERROR(AsGraphDef(input, std::move(serialization_ctx), result));

  // Symbolic `_Retval` node indicates which node corresponds to the dataset.
  for (const auto& node : result->node()) {
    if (node.op() == kRetvalOp) {
      *dataset_node = node.input(0);
    }
  }
  return absl::OkStatus();
}

absl::Status AsGraphDef(const DatasetBase* dataset,
                        SerializationContext&& serialization_ctx,
                        GraphDef* graph_def) {
  if (serialization_ctx.external_state_policy() ==
      ExternalStatePolicy::POLICY_FAIL) {
    TF_RETURN_IF_ERROR(dataset->CheckExternalState());
  }
  if (serialization_ctx.external_state_policy() ==
      ExternalStatePolicy::POLICY_WARN) {
    std::vector<string> stateful_op_names;
    TF_RETURN_IF_ERROR(FindStatefulOps(*graph_def, &stateful_op_names));
    if (!stateful_op_names.empty()) {
      LOG(WARNING) << "We found the following stateful ops in the dataset "
                      "construction graph whose state would not be "
                      "serialized and might "
                      "cause subtle bugs: "
                   << absl::StrJoin(stateful_op_names, ", ");
    }
  }
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node* output_node = nullptr;
  TF_RETURN_IF_ERROR(
      db.AddInputDataset(&serialization_ctx, dataset, &output_node));
  // Insert a purely symbolic _Retval node to indicate to consumers which node
  // represents `dataset`.
  ops::UnaryOp(std::string(kRetvalOp), output_node,
               b.opts()
                   .WithName("dataset")
                   .WithAttr("T", DT_VARIANT)
                   .WithAttr("index", 0));
  TF_RETURN_IF_ERROR(b.ToGraphDef(graph_def));
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<std::string, int64_t>> CheckpointStats(
    const std::string& checkpoint_bytes) {
  TensorProto proto;
  if (!ParseProtoUnlimited(&proto, checkpoint_bytes)) {
    return absl::InvalidArgumentError(
        "Failed to parse checkpoint bytes into proto.");
  }
  Tensor t;
  if (!t.FromProto(proto)) {
    return absl::InvalidArgumentError(
        "Failed to parse checkpoint tensor from proto.");
  }

  auto variant = t.scalar<Variant>()();
  auto* w = variant.get<IteratorStateVariant>();
  if (!w) {
    return absl::InvalidArgumentError(
        "Failed to access IteratorStateVariant inside checkpoint tensor");
  }
  const VariantTensorData* data = w->GetData();
  auto reader = std::make_unique<VariantTensorDataReader>(
      std::vector<const VariantTensorData*>{data});
  absl::flat_hash_map<std::string, int64_t> stats;
  for (const auto& [key, tensor] : reader->ReadAllTensors()) {
    stats[key] = tensor.TotalBytes();
  }
  return stats;
}

}  // namespace data
}  // namespace tensorflow
