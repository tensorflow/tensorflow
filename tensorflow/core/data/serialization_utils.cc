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

#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph_def_builder.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDelimiter[] = "@@";
constexpr char kComponent[] = "component";
constexpr char kNumComponents[] = "num_components";
constexpr char kNumElements[] = "num_elements";
constexpr char kIsDataset[] = ".is_dataset";
constexpr char kOutputNode[] = ".output_node";

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
  return OkStatus();
}

Status FromGraphDef(FunctionLibraryRuntime* flr, const GraphDef& graph_def,
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
  return OkStatus();
}

// FindStatefulOps searches `graph_def` for all of its stateful ops storing
// their names in `stateful_op_names`.
Status FindStatefulOps(const GraphDef& graph_def,
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
  return OkStatus();
}

}  // namespace

Status ReadElementsFromCheckpoint(IteratorContext* ctx,
                                  IteratorStateReader* reader,
                                  StringPiece key_prefix,
                                  std::vector<std::vector<Tensor>>* elements) {
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
  return OkStatus();
}

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
  return OkStatus();
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

Status VariantTensorDataReader::ReadScalar(StringPiece key,
                                           int64_t* val) const {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadScalar(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           int64_t* val) const {
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece key,
                                           tstring* val) const {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadScalar(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           tstring* val) const {
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece key, Tensor* val) const {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadTensor(name, key, val);
}

Status VariantTensorDataReader::ReadTensor(FunctionLibraryRuntime* flr,
                                           StringPiece key, Tensor* val) const {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadTensorInternal(flr, name, key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece name, StringPiece key,
                                           Tensor* val) const {
  return ReadTensor(/*flr=*/nullptr, name, key, val);
}

Status VariantTensorDataReader::ReadTensor(FunctionLibraryRuntime* flr,
                                           StringPiece name, StringPiece key,
                                           Tensor* val) const {
  return ReadTensorInternal(flr, name, key, val);
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
  return OkStatus();
}

Status VariantTensorDataReader::ReadTensorInternal(FunctionLibraryRuntime* flr,
                                                   StringPiece n,
                                                   StringPiece key,
                                                   Tensor* val) const {
  if (Contains(n, strings::StrCat(key, kIsDataset))) {
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
  return OkStatus();
}

Status VariantTensorDataReader::ReadDatasetInternal(FunctionLibraryRuntime* flr,
                                                    StringPiece n,
                                                    StringPiece key,
                                                    Tensor* val) const {
  if (flr == nullptr) {
    return errors::Internal(
        "Function library runtime is needed to restore a dataset.");
  }
  tstring output_node, serialized_graph_def;
  TF_RETURN_IF_ERROR(
      ReadScalar(n, strings::StrCat(key, kOutputNode), &output_node));
  TF_RETURN_IF_ERROR(
      ReadScalar(n, strings::StrCat(key), &serialized_graph_def));
  GraphDef graph_def;
  graph_def.ParseFromString(serialized_graph_def);
  TF_RETURN_IF_ERROR(FromGraphDef(flr, graph_def, {}, output_node, val));
  return OkStatus();
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key,
                                            const int64_t val) {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteScalar(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const int64_t val) {
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key,
                                            const tstring& val) {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteScalar(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const tstring& val) {
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece key,
                                            const Tensor& val) {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteTensor(name, key, val);
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
    data_[name] = absl::make_unique<VariantTensorData>();
    data_[name]->set_type_name("tensorflow::Iterator");
  }
  *(data_[name]->add_tensors()) = val;
  return OkStatus();
}

Status VariantTensorDataWriter::WriteDatasetInternal(
    StringPiece n, StringPiece key, const DatasetBase* dataset) {
  GraphDef graph_def;
  SerializationContext ctx((SerializationContext::Params()));
  TF_RETURN_IF_ERROR(AsGraphDef(dataset, std::move(ctx), &graph_def));
  string output_node;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
      break;
    }
  }
  string result;
  graph_def.SerializeToString(&result);
  TF_RETURN_IF_ERROR(WriteScalar(n, strings::StrCat(key, kIsDataset), ""));
  TF_RETURN_IF_ERROR(
      WriteScalar(n, strings::StrCat(key, kOutputNode), output_node));
  TF_RETURN_IF_ERROR(WriteScalar(n, key, result));
  return OkStatus();
}

Status AsGraphDefForRewrite(OpKernelContext* ctx, const DatasetBase* input,
                            std::vector<std::pair<string, Tensor>>* input_list,
                            GraphDef* result, string* dataset_node) {
  SerializationContext::Params params(ctx);
  params.input_list = input_list;
  params.external_state_policy =
      SerializationContext::ExternalStatePolicy::kIgnore;
  params.is_graph_rewrite = true;
  SerializationContext serialization_ctx(params);
  TF_RETURN_IF_ERROR(AsGraphDef(input, std::move(serialization_ctx), result));

  // Symbolic `_Retval` node indicates which node corresponds to the dataset.
  for (const auto& node : result->node()) {
    if (node.op() == "_Retval") {
      *dataset_node = node.input(0);
    }
  }
  return OkStatus();
}

Status AsGraphDef(const DatasetBase* dataset,
                  SerializationContext&& serialization_ctx,
                  GraphDef* graph_def) {
  if (serialization_ctx.external_state_policy() ==
      SerializationContext::ExternalStatePolicy::kFail) {
    TF_RETURN_IF_ERROR(dataset->CheckExternalState());
  }
  if (serialization_ctx.external_state_policy() ==
      SerializationContext::ExternalStatePolicy::kWarn) {
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
  ops::UnaryOp("_Retval", output_node,
               b.opts()
                   .WithName("dataset")
                   .WithAttr("T", DT_VARIANT)
                   .WithAttr("index", 0));
  TF_RETURN_IF_ERROR(b.ToGraphDef(graph_def));
  return OkStatus();
}

}  // namespace data
}  // namespace tensorflow
