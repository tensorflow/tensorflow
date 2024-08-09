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
#include "tensorflow/core/framework/dataset.h"

#include <unordered_map>
#include <vector>

#include "tensorflow/core/activity_watcher/activity.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"

// On Windows, disable some macros that would break compile
#if defined(PLATFORM_WINDOWS)
#undef GetMessage
#endif

namespace tensorflow {
namespace data {
namespace {

static mutex* get_dataset_op_registry_lock() {
  static mutex dataset_op_registry_lock(LINKER_INITIALIZED);
  return &dataset_op_registry_lock;
}

static std::unordered_set<string>* get_dataset_op_registry() {
  static std::unordered_set<string>* names = new std::unordered_set<string>;
  return names;
}

std::string UniqueNodeName(const std::string& base) {
  static std::atomic<int64_t> counter(0);
  return strings::StrCat(base, "/", counter.fetch_add(1));
}

// A wrapper class for storing a `DatasetBase` instance in a DT_VARIANT tensor.
// Objects of the wrapper class own a reference on an instance of `DatasetBase`,
// and the wrapper's copy constructor and destructor take care of managing the
// reference count.
//
// NOTE(mrry): This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `DatasetBase` object, so the `Encode()` and `Decode()` methods are not
// implemented.
class DatasetVariantWrapper {
 public:
  DatasetVariantWrapper() : dataset_(nullptr) {}

  // Transfers ownership of `dataset` to `*this`.
  explicit DatasetVariantWrapper(DatasetBase* dataset) : dataset_(dataset) {}

  DatasetVariantWrapper(const DatasetVariantWrapper& other)
      : dataset_(other.dataset_) {
    if (dataset_) dataset_->Ref();
  }

  DatasetVariantWrapper& operator=(DatasetVariantWrapper&& other) {
    if (&other == this) return *this;
    std::swap(dataset_, other.dataset_);
    return *this;
  }

  DatasetVariantWrapper& operator=(const DatasetVariantWrapper& other) = delete;

  ~DatasetVariantWrapper() {
    if (dataset_) dataset_->Unref();
  }

  DatasetBase* get() const { return dataset_; }

  string TypeName() const { return "tensorflow::DatasetVariantWrapper"; }
  string DebugString() const {
    if (dataset_) {
      return dataset_->DebugString();
    } else {
      return "<Uninitialized DatasetVariantWrapper>";
    }
  }
  void Encode(VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
  }
  bool Decode(const VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
    return false;
  }

 private:
  DatasetBase* dataset_;  // Owns one reference.
};

const char kWrappedDatasetVariantTypeName[] =
    "tensorflow::data::WrappedDatasetVariant";

class WrappedDatasetVariantWrapper {
 public:
  WrappedDatasetVariantWrapper() {}

  explicit WrappedDatasetVariantWrapper(const Tensor& ds_tensor)
      : ds_tensor_(ds_tensor) {}

  Tensor get() const { return ds_tensor_; }

  string TypeName() const { return "tensorflow::WrappedDatasetVariantWrapper"; }

  string DebugString() const {
    return "tensorflow::WrappedDatasetVariantWrapper::DebugString";
  }

  void Encode(VariantTensorData* data) const {
    *(data->add_tensors()) = ds_tensor_;
  }

  bool Decode(const VariantTensorData& data) {
    ds_tensor_ = data.tensors(0);
    return true;
  }

 private:
  Tensor ds_tensor_;
};

class WrapDatasetVariantOp : public OpKernel {
 public:
  explicit WrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    DatasetBase* unused;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(tensor, &unused));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->scalar<Variant>()() = WrappedDatasetVariantWrapper(tensor);
  }
};

REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant").Device(DEVICE_CPU),
                        WrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        WrapDatasetVariantOp);

class UnwrapDatasetVariantOp : public OpKernel {
 public:
  explicit UnwrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    Variant variant = tensor.scalar<Variant>()();
    const WrappedDatasetVariantWrapper* wrapper =
        variant.get<WrappedDatasetVariantWrapper>();
    OP_REQUIRES(ctx, wrapper != nullptr,
                errors::InvalidArgument(
                    "Tensor must be a WrappedDataset variant object."));
    Tensor ds_tensor = wrapper->get();
    OP_REQUIRES_OK(ctx, ctx->set_output("output_handle", ds_tensor));
  }
};

REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant").Device(DEVICE_CPU),
                        UnwrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        UnwrapDatasetVariantOp);

static Status WrappedDatasetVariantDeviceCopy(
    const WrappedDatasetVariantWrapper& from, WrappedDatasetVariantWrapper* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  *to = WrappedDatasetVariantWrapper(from);
  return OkStatus();
}

#define REGISTER_OPTIONAL_COPY(DIRECTION)               \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      WrappedDatasetVariantWrapper, DIRECTION,          \
      WrappedDatasetVariantDeviceCopy)

REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(WrappedDatasetVariantWrapper,
                                       kWrappedDatasetVariantTypeName);

}  // namespace

Status GraphDefBuilderWrapper::AddDataset(const DatasetBase* dataset,
                                          const std::vector<Node*>& inputs,
                                          Node** output) {
  return AddDataset(dataset, inputs, {}, output);
}

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset, const std::vector<Node*>& inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    Node** output) {
  std::vector<std::pair<size_t, Node*>> enumerated_inputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    enumerated_inputs[i] = std::make_pair(i, inputs[i]);
  }
  return AddDataset(dataset, enumerated_inputs, {}, attrs, output);
}

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset,
    const std::vector<std::pair<size_t, Node*>>& inputs,
    const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    Node** output) {
  return AddDataset(dataset, inputs, list_inputs, attrs,
                    /*use_dataset_name=*/false, output);
}

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset,
    const std::vector<std::pair<size_t, Node*>>& inputs,
    const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    bool use_dataset_name, Node** output) {
  auto& type_string = dataset->type_string();
  auto opts = absl::make_unique<GraphDefBuilder::Options>(b_->opts());
  // TODO(srbs|mrry): Not all datasets have output_types and output_shapes
  // attributes defined. It will be nice to have a consistent pattern.
  bool has_output_types_attr = HasAttr(type_string, "output_types");
  bool has_output_shapes_attr = HasAttr(type_string, "output_shapes");
  if (has_output_shapes_attr) {
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr("output_shapes", dataset->output_shapes()));
  }
  if (has_output_types_attr) {
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr("output_types", dataset->output_dtypes()));
  }
  bool has_metadata_attr = HasAttr(type_string, "metadata");
  if (has_metadata_attr) {
    std::string serialized_metadata;
    dataset->metadata().SerializeToString(&serialized_metadata);
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr("metadata", serialized_metadata));
  }
  for (const auto& attr : attrs) {
    opts = absl::make_unique<GraphDefBuilder::Options>(
        opts->WithAttr(attr.first, attr.second));
  }
  if (opts->HaveError()) {
    return errors::Internal("AddDataset: Failed to build Options with error ",
                            opts->StatusToString());
  }
  NodeBuilder node_builder(
      use_dataset_name ? dataset->node_name() : opts->GetNameForOp(type_string),
      type_string, opts->op_registry());
  {
    size_t total_size = inputs.size() + list_inputs.size();
    auto inputs_iter = inputs.begin();
    auto list_inputs_iter = list_inputs.begin();
    for (int i = 0; i < total_size; i++) {
      if (inputs_iter != inputs.end() && inputs_iter->first == i) {
        node_builder.Input(NodeBuilder::NodeOut(inputs_iter->second));
        inputs_iter++;
      } else if (list_inputs_iter != list_inputs.end() &&
                 list_inputs_iter->first == i) {
        std::vector<NodeBuilder::NodeOut> nodeout_inputs;
        nodeout_inputs.reserve(list_inputs_iter->second.size());
        for (Node* n : list_inputs_iter->second) {
          nodeout_inputs.emplace_back(n);
        }
        node_builder.Input(nodeout_inputs);
        list_inputs_iter++;
      } else {
        return errors::InvalidArgument("No input found for index ", i);
      }
    }
  }
  *output = opts->FinalizeBuilder(&node_builder);
  if (*output == nullptr) {
    return errors::Internal("AddDataset: Failed to build ", type_string,
                            " op with error ", opts->StatusToString());
  }
  return OkStatus();
}

Status GraphDefBuilderWrapper::AddFunction(
    SerializationContext* ctx, const string& function_name,
    const FunctionLibraryDefinition& lib_def) {
  if (b_->HasFunction(function_name)) {
    VLOG(1) << "Function with name " << function_name << "already exists in"
            << " the graph. It will not be added again.";
    return OkStatus();
  }
  const FunctionDef* f_def = lib_def.Find(function_name);
  if (f_def == nullptr) {
    return errors::InvalidArgument("Unable to find FunctionDef for ",
                                   function_name, " in the registry.");
  }
  FunctionDefLibrary def;
  *def.add_function() = *f_def;
  const string gradient_func = lib_def.FindGradient(function_name);
  if (!gradient_func.empty()) {
    GradientDef* g_def = def.add_gradient();
    g_def->set_function_name(function_name);
    g_def->set_gradient_func(gradient_func);
  }
  TF_RETURN_IF_ERROR(b_->AddFunctionLibrary(def));

  // Recursively add functions in inputs of function_name.
  for (const NodeDef& node_def : f_def->node_def()) {
    const OpRegistrationData* op_reg_data = nullptr;
    TF_RETURN_IF_ERROR(lib_def.LookUp(node_def.op(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      TF_RETURN_IF_ERROR(AddFunction(ctx, op_reg_data->op_def.name(), lib_def));
    }
    // Recursively add functions in attrs of this NodeDef.
    for (const auto& pair : node_def.attr()) {
      TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, pair.second, lib_def));
    }
  }

  // Recursively add functions in attrs of function_name.
  for (auto iter = f_def->attr().begin(); iter != f_def->attr().end(); iter++) {
    TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, iter->second, lib_def));
  }
  return OkStatus();
}

void GraphDefBuilderWrapper::AddPlaceholderInternal(const Tensor& val,
                                                    Node** output) {
  *output = ops::SourceOp(
      "Placeholder",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("shape", val.shape()));
}

void GraphDefBuilderWrapper::AddTensorInternal(const Tensor& val,
                                               Node** output) {
  *output = ops::SourceOp(
      "Const",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("value", val));
}

bool GraphDefBuilderWrapper::HasAttr(const string& name,
                                     const string& attr_name) const {
  const OpDef* op_def = nullptr;
  Status s = b_->opts().op_registry()->LookUpOpDef(name, &op_def);
  if (!s.ok() || op_def == nullptr) {
    return false;
  }
  return HasAttr(op_def, attr_name);
}

int32_t GetRunnerThreadpoolSizeFromOpKernelContext(OpKernelContext* ctx) {
  thread::ThreadPool* thread_pool =
      ctx->device()->tensorflow_device_thread_pool();
  if (thread_pool) {
    return thread_pool->NumThreads();
  } else {
    static const int32_t kDefaultRunnerThreadpoolSize = port::MaxParallelism();
    return kDefaultRunnerThreadpoolSize;
  }
}

int64_t MemoryCheckpoint::IdRegistry::Add(const std::string& prefix,
                                          const std::string& key) {
  mutex_lock l(mu_);
  auto pair = std::make_pair(prefix, key);
  if (string_to_int_.contains(pair)) {
    return string_to_int_[pair];
  }
  int64_t id = next_id_++;
  int_to_string_[id] = pair;
  string_to_int_[pair] = id;
  return id;
}

std::vector<int64_t> MemoryCheckpoint::IdRegistry::GetMatchingIds(
    const std::string& prefix_to_match) {
  mutex_lock l(mu_);
  std::vector<int64_t> ids;
  for (const auto& [pair, id] : string_to_int_) {
    auto [prefix, key] = pair;
    if (prefix.compare(0, prefix_to_match.length(), prefix_to_match) == 0) {
      ids.push_back(id);
    }
  }
  return ids;
}

std::pair<std::string, std::string> MemoryCheckpoint::IdRegistry::Get(
    int64_t id) {
  mutex_lock l(mu_);
  auto result = int_to_string_.find(id);
  DCHECK(result != int_to_string_.end())
      << "Failed find id " << id << " in IdRegistry. "
      << "Max id is: " << next_id_ - 1;
  return result->second;
}

void MemoryCheckpoint::IdRegistry::RemoveIds(const std::vector<int64_t>& ids) {
  mutex_lock l(mu_);
  for (const auto& id : ids) {
    string_to_int_.erase(int_to_string_[id]);
    int_to_string_.erase(id);
  }
}

std::string MemoryCheckpoint::DebugString() const {
  std::string result = absl::StrCat("status=", status_.ToString(),
                                    ", "
                                    "root=",
                                    (is_root_ ? "true" : "false"), "\n");
  absl::StrAppend(&result, "number of integers: ", int_values_.size(), "\n");
  for (const auto& [k, v] : int_values_) {
    absl::StrAppend(&result, "  ", id_registry_->Get(k).first, ":",
                    id_registry_->Get(k).second, ": ", v, "\n");
  }
  absl::StrAppend(&result, "number of strings: ", str_values_.size(), "\n");
  for (const auto& [k, v] : str_values_) {
    absl::StrAppend(&result, "  ", id_registry_->Get(k).first, ":",
                    id_registry_->Get(k).second, ": ", v, "\n");
  }
  absl::StrAppend(&result, "number of tensors: ", tensor_values_.size(), "\n");

  absl::StrAppend(
      &result, "number of expired prefixes: ", expired_prefixes_.size(), "\n");
  return result;
}

void MemoryCheckpoint::Merge(MemoryCheckpoint* other) {
  if (!status_.ok()) {
    return;
  }

  if (!other->status_.ok()) {
    status_ = other->status_;
    int_values_.clear();
    str_values_.clear();
    tensor_values_.clear();
  }

  for (const auto& [k, v] : other->int_values_) {
    int_values_[k] = v;
  }
  for (const auto& [k, v] : other->str_values_) {
    str_values_[k] = v;
  }
  for (const auto& [k, v] : other->tensor_values_) {
    tensor_values_[k] = v;
  }

  // Get the expired prefixes from `other`. Since the info only needs to be
  // propagated once downstream, we also clean the `expired_prefixes_` of
  // `other` here.
  for (const auto& prefix : other->expired_prefixes_) {
    Purge(prefix);
  }

  other->expired_prefixes_.clear();
  VLOG(5) << "MemoryCheckpoint::Merge " << DebugString();
}

void MemoryCheckpoint::Purge(const std::string& prefix) {
  std::vector<int64_t> ids = id_registry_->GetMatchingIds(prefix);
  for (const auto& id : ids) {
    int_values_.erase(id);
    str_values_.erase(id);
    tensor_values_.erase(id);
  }
  if (!is_root_) {
    expired_prefixes_.insert(prefix);
  } else {
    // We no longer need the mapping after change has been propagated all the
    // way to root.
    id_registry_->RemoveIds(ids);
  }
}

Status MemoryCheckpoint::Save(IteratorStateWriter* writer) const {
  for (const auto& [id, value] : int_values_) {
    auto [prefix, key] = id_registry_->Get(id);
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix, key, value));
  }
  for (const auto& [id, value] : str_values_) {
    auto [prefix, key] = id_registry_->Get(id);
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix, key, value));
  }
  for (const auto& [id, value] : tensor_values_) {
    auto [prefix, key] = id_registry_->Get(id);
    TF_RETURN_IF_ERROR(writer->WriteTensor(prefix, key, value));
  }
  return OkStatus();
}

Status IteratorBase::InitializeBase(IteratorContext* ctx,
                                    const IteratorBase* parent) {
  parent_ = parent;
  id_ =
      Hash64CombineUnordered(Hash64(prefix()), reinterpret_cast<uint64>(this));
  if (parent_) {
    parent_id_ = Hash64CombineUnordered(Hash64(parent_->prefix()),
                                        reinterpret_cast<uint64>(parent_));
    // This block of code is executed only when `parent_` is not a `nullptr`
    // because we do not create a `Node` in the `Model` for `RootDataset`.
    if (const auto& model = ctx->model()) {
      auto factory = [ctx, this](model::Node::Args args) {
        return CreateNode(ctx, std::move(args));
      };
      model->AddNode(std::move(factory), prefix(), parent->model_node(),
                     &node_);
      cleanup_fns_.push_back([this, model]() { model->RemoveNode(node_); });
    }
  }
  return OkStatus();
}

Status GetCompressedElementFromVariantTensor(
    const Tensor& tensor, const CompressedElement** out_compressed_element) {
  if (!(tensor.dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "`CompressedElement` tensor must be a scalar of dtype `DT_VARIANT`.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const CompressedElement* compressed_element =
      variant.get<CompressedElement>();
  if (compressed_element == nullptr) {
    return errors::InvalidArgument(
        "Tensor must be a `CompressedElement` object.");
  }
  *out_compressed_element = compressed_element;
  return OkStatus();
}

int64_t GetAllocatedBytes(const std::vector<Tensor>& element) {
  int64_t allocated_bytes = 0;
  for (auto& tensor : element) {
    if (tensor.dtype() == DT_VARIANT) {
      // Special case certain variants where AllocatedBytes() doesn't give an
      // accurate byte count.
      DatasetBase* dataset;
      if (GetDatasetFromVariantTensor(tensor, &dataset).ok()) {
        allocated_bytes += dataset->AllocatedBytes();
        continue;
      }
      const CompressedElement* compressed_element;
      if (GetCompressedElementFromVariantTensor(tensor, &compressed_element)
              .ok()) {
        allocated_bytes += compressed_element->ByteSizeLong();
        continue;
      }
    }
    allocated_bytes += tensor.AllocatedBytes();
  }
  return allocated_bytes;
}

int64_t GetTotalBytes(const std::vector<Tensor>& element) {
  int64_t total_bytes = 0;
  for (auto& tensor : element) {
    if (tensor.dtype() == DT_VARIANT) {
      // Special case certain variants where TotalBytes() doesn't give an
      // accurate byte count.
      DatasetBase* dataset;
      if (GetDatasetFromVariantTensor(tensor, &dataset).ok()) {
        total_bytes += dataset->TotalBytes();
        continue;
      }
      const CompressedElement* compressed_element;
      if (GetCompressedElementFromVariantTensor(tensor, &compressed_element)
              .ok()) {
        total_bytes += compressed_element->ByteSizeLong();
        continue;
      }
    }
    total_bytes += tensor.TotalBytes();
  }
  return total_bytes;
}

std::string FullName(const std::string& prefix, const std::string& name) {
  if (str_util::StrContains(name, kColon)) {
    LOG(ERROR) << name << " should not contain " << kColon;
  }

  return strings::StrCat(kFullNameRandomHex, kPipe, prefix, kColon, name);
}

Status ExtractIteratorPrefix(StringPiece key, string* prefix) {
  if (!str_util::StartsWith(key, data::kFullNameRandomHex)) {
    return errors::InvalidArgument("Key: ", key,
                                   " was not generated using full_name.");
  }
  std::vector<string> split_keys = str_util::Split(key, data::kPipe);
  if (split_keys.size() != 2) {
    return errors::InvalidArgument("Key: ", key,
                                   " was not generated using full_name.");
  }
  string real_key = split_keys[1];
  const int pos = real_key.rfind(kColon);
  *prefix = real_key.substr(0, pos);
  return OkStatus();
}

Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset) {
  if (!(tensor.dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const DatasetVariantWrapper* wrapper = variant.get<DatasetVariantWrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be a Dataset object.");
  }
  *out_dataset = wrapper->get();
  if (*out_dataset == nullptr) {
    return errors::Internal("Read uninitialized Dataset variant.");
  }
  return OkStatus();
}

Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor) {
  if (!(tensor->dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = DatasetVariantWrapper(dataset);
  return OkStatus();
}

namespace internal {

#define WARN_PROTO_FIELD_CONFLICT(reflection, field, field_type, src, dst)     \
  {                                                                            \
    auto source_value = reflection->Get##field_type(src, field);               \
    auto destination_value = reflection->Get##field_type(*dst, field);         \
    if (source_value != destination_value) {                                   \
      LOG(WARNING) << "Changing the value of option field " << field->name()   \
                   << " from " << destination_value << " to " << source_value; \
    }                                                                          \
  }

#define WARN_PROTO_ENUM_FIELD_CONFLICT(reflection, field, src, dst) \
  {                                                                 \
    auto source_value = reflection->GetEnum(src, field);            \
    auto destination_value = reflection->GetEnum(*dst, field);      \
    if (source_value != destination_value) {                        \
      LOG(WARNING) << "Changing the value of option enum field "    \
                   << field->name() << " from "                     \
                   << destination_value->full_name() << " to "      \
                   << source_value->full_name();                    \
    }                                                               \
  }

void WarnProtoConflicts(const protobuf::Message& src, protobuf::Message* dst) {
  std::vector<const protobuf::FieldDescriptor*> set_src;
  std::vector<const protobuf::FieldDescriptor*> set_dst;
  const protobuf::Reflection* reflection = src.GetReflection();
  reflection->ListFields(src, &set_src);
  reflection->ListFields(*dst, &set_dst);
  std::sort(set_src.begin(), set_src.end());
  std::sort(set_dst.begin(), set_dst.end());

  std::vector<const protobuf::FieldDescriptor*> in_both;
  std::set_intersection(set_src.begin(), set_src.end(), set_dst.begin(),
                        set_dst.end(), std::back_inserter(in_both));

  for (auto field : in_both) {
    // Used for Job Instrumentation, users should not be warned.
    if (field->name() == "framework_type") {
      continue;
    }
    if (field->type() == protobuf::FieldDescriptor::TYPE_MESSAGE) {
      WarnProtoConflicts(reflection->GetMessage(src, field),
                         reflection->MutableMessage(dst, field));
    } else {
      switch (field->cpp_type()) {
        case protobuf::FieldDescriptor::CPPTYPE_INT32:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Int32, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_INT64:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Int64, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_UINT32:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, UInt32, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_UINT64:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, UInt64, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Double, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_FLOAT:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Float, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_BOOL:
          WARN_PROTO_FIELD_CONFLICT(reflection, field, Bool, src, dst);
          break;
        case protobuf::FieldDescriptor::CPPTYPE_ENUM:
          WARN_PROTO_ENUM_FIELD_CONFLICT(reflection, field, src, dst);
          break;
        default: {
          LOG(ERROR) << "Unrecognized proto type for field "
                     << field->full_name();
        }
      }
    }
  }
}

#undef WARN_PROTO_ENUM_FIELD_CONFLICT
#undef WARN_PROTO_FIELD_CONFLICT

void MergeOptions(const protobuf::Message& source,
                  protobuf::Message* destination) {
  WarnProtoConflicts(source, destination);
  destination->MergeFrom(source);
}

void MergeOptions(const protobuf::MessageLite& source,
                  protobuf::MessageLite* destination) {
  destination->CheckTypeAndMergeFrom(source);
}

}  // namespace internal

void DatasetBase::Initialize(const Metadata& metadata) {
  Status s = ComputeNumSources();
  if (!s.ok()) {
    LOG_EVERY_N_SEC(ERROR, 10) << s;
  }
  s = MergeOptionsFromInputs();
  if (!s.ok()) {
    LOG_EVERY_N_SEC(ERROR, 10) << s;
  }
  metadata_ = metadata;
  if (metadata_.name() == "") {
    static std::atomic<int64_t> id_counter(0);
    *metadata_.mutable_name() =
        strings::StrCat(type_string(), ":", id_counter.fetch_add(1));
  }
}

Status DatasetBase::ComputeNumSources() {
  std::vector<const DatasetBase*> inputs;
  Status s = InputDatasets(&inputs);
  if (errors::IsUnimplemented(s)) {
    return s;
  }
  if (num_sources_ >= 0) {
    // Already computed.
    return OkStatus();
  }
  num_sources_ = 0;
  if (inputs.empty()) {
    num_sources_ = 1;
    return OkStatus();
  }
  for (const auto& input : inputs) {
    if (input->num_sources() < 0) {
      return errors::FailedPrecondition(
          "Cannot compute input sources for dataset of type ", type_string(),
          ", because sources could not be computed for input dataset of type ",
          input->type_string());
    }
    num_sources_ += input->num_sources();
  }
  return OkStatus();
}

Status DatasetBase::CheckRandomAccessCompatible(const int64 index) const {
  CardinalityOptions options;
  options.set_compute_level(CardinalityOptions::CARDINALITY_COMPUTE_MODERATE);
  int64 cardinality = Cardinality(options);
  if (cardinality == kInfiniteCardinality ||
      cardinality == kUnknownCardinality) {
    return tensorflow::errors::FailedPrecondition(
        "Dataset of type ", this->DebugString(), " has ",
        cardinality == kInfiniteCardinality ? "infinite" : "unknown",
        " cardinality, which does not support random access.");
  }
  if (index < 0 || index >= cardinality) {
    return errors::OutOfRange("Index out of range [0, ", cardinality,
                              "):", index);
  }
  return OkStatus();
}

Status DatasetBase::Get(OpKernelContext* ctx, int64 index,
                        std::vector<Tensor>* out_tensors) const {
  return errors::Unimplemented("Random access is not implemented for dataset ",
                               DebugString());
}

Status DatasetBase::Get(AnyContext ctx, int64 index,
                        std::vector<Tensor>* out_tensors) const {
  return errors::Unimplemented("Random access is not implemented for dataset ",
                               DebugString());
}

absl::StatusOr<DatasetBase*> DatasetBase::Finalize(
    OpKernelContext* ctx,
    std::function<absl::StatusOr<core::RefCountPtr<DatasetBase>>()>
        make_finalized_dataset) const {
  mutex_lock l(mu_);
  if (!finalized_dataset_) {
    TF_ASSIGN_OR_RETURN(finalized_dataset_, make_finalized_dataset());
  }
  return finalized_dataset_.get();
}

Status DatasetBase::MergeOptionsFromInputs() {
  std::vector<const DatasetBase*> inputs;
  Status s = InputDatasets(&inputs);
  if (errors::IsUnimplemented(s)) {
    return s;
  }
  if (inputs.empty()) {
    return OkStatus();
  }
  // Merge options from inputs sequentially before merging options from dataset.
  // Since the last options merged takes precedence, the options that may be set
  // for the current dataset through OptionsDataset takes precedence over those
  // set on the input datasets.
  Options merged_options = inputs[0]->options_;
  for (int i = 1; i < inputs.size(); ++i) {
    internal::MergeOptions(inputs[i]->options_, &merged_options);
  }
  internal::MergeOptions(options_, &merged_options);
  options_ = merged_options;
  return OkStatus();
}

Status DatasetBase::MakeIterator(
    IteratorContext* ctx, const IteratorBase* parent,
    const string& output_prefix,
    std::unique_ptr<IteratorBase>* iterator) const {
  if (type_string() == "OptionsDataset" || type_string() == "FinalizeDataset") {
    std::vector<const DatasetBase*> inputs;
    Status s = InputDatasets(&inputs);
    return inputs[0]->MakeIterator(ctx, parent, output_prefix, iterator);
  }
  profiler::TraceMe traceme(
      [&] {
        return profiler::TraceMeEncode(
            strings::StrCat("MakeIterator::", type_string()), {});
      },
      profiler::TraceMeLevel::kInfo);
  *iterator = MakeIteratorInternal(output_prefix);
  Status s = (*iterator)->InitializeBase(ctx, parent);
  if (s.ok()) {
    s.Update((*iterator)->Initialize(ctx));
    ctx->SaveCheckpoint(iterator->get());
  }
  if (!s.ok()) {
    // Reset the iterator to avoid returning an uninitialized iterator.
    iterator->reset();
  }
  return s;
}

Status DatasetBase::MakeSplitProviders(
    std::vector<std::unique_ptr<SplitProvider>>* split_providers) const {
  std::vector<const DatasetBase*> inputs;
  Status s = InputDatasets(&inputs);
  if (errors::IsUnimplemented(s)) {
    return errors::Unimplemented(
        "Cannot create split providers for dataset of type ", type_string(),
        ", because the dataset implements neither `InputDatasets` nor "
        "`MakeSplitProvider`.");
  }
  if (inputs.size() != 1) {
    return errors::Unimplemented(
        "Cannot create split providers for dataset of type ", type_string(),
        ", because the dataset is not unary (instead having arity ",
        inputs.size(),
        "), and no custom implementation of `MakeSplitProvider` is defined.");
  }
  return inputs[0]->MakeSplitProviders(split_providers);
}

std::optional<int64_t> DatasetBase::GetEstimatedElementSize() const {
  const auto& shapes = output_shapes();
  const auto& dtypes = output_dtypes();
  if (shapes.size() != dtypes.size()) {
    LOG(ERROR) << "This should not happen because the sizes of output_shapes() "
                  "and output_dtypes() should always be "
                  "the same.";
    return std::nullopt;
  }

  size_t num_outputs = shapes.size();
  int64_t element_size = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto& partial_shape = shapes[i];
    const auto& dtype = dtypes[i];
    auto num_elements = partial_shape.num_elements();
    if (num_elements == -1) {
      return std::nullopt;
    }
    element_size += num_elements * DataTypeSize(dtype);
  }
  return element_size;
}

int64_t DatasetBase::Cardinality() const {
  mutex_lock l(cardinality_mu_);
  if (cardinality_ == kUnknownCardinality) {
    CardinalityOptions options;
    cardinality_ = CardinalityInternal(options);
  }
  return cardinality_;
}

int64_t DatasetBase::Cardinality(CardinalityOptions options) const {
  mutex_lock l(cardinality_mu_);
  if (cardinality_ == kUnknownCardinality) {
    cardinality_ = CardinalityInternal(options);
  }
  return cardinality_;
}

Status DatasetBase::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
  return errors::Unimplemented(
      "Cannot compute input sources for dataset of type ", type_string(),
      ", because the dataset does not implement `InputDatasets`. To fix this, "
      "your dataset should override the `InputDatasets` method. If it is a "
      "source dataset, it should return empty inputs.");
}

Status DatasetBase::DatasetGraphDefBuilder::AddInputDataset(
    SerializationContext* ctx, const DatasetBase* dataset, Node** output) {
  Status status = dataset->AsGraphDefInternal(ctx, this, output);
  if (ctx->is_graph_rewrite()) {
    if (status.ok()) {
      // Record cardinality in an unregistered attributes so that rewrites have
      // this information.
      (*output)->AddAttr(kCardinalityAttrForRewrite, dataset->Cardinality());
    } else if (errors::IsUnimplemented(status)) {
      Tensor t(DT_VARIANT, TensorShape({}));
      // `StoreDatasetInVariantTensor` will transfer ownership of `dataset`. We
      // increment the refcount of `dataset` here to retain ownership.
      dataset->Ref();
      TF_RETURN_IF_ERROR(
          StoreDatasetInVariantTensor(const_cast<DatasetBase*>(dataset), &t));
      TF_RETURN_IF_ERROR(AddPlaceholder(t, output));
      DCHECK_NE(ctx->input_list(), nullptr);
      ctx->input_list()->emplace_back((*output)->name(), std::move(t));
      LOG_EVERY_N_SEC(WARNING, 30)
          << "Input of " << dataset->DebugString()
          << " will not be optimized because the dataset does not implement "
             "the "
             "AsGraphDefInternal() method needed to apply optimizations.";
      return OkStatus();
    }
  }
  return status;
}

Status DatasetBase::DatasetGraphDefBuilder::AddDatasetOrTensor(
    SerializationContext* ctx, const Tensor& t, Node** output) {
  if (t.dtype() == DT_VARIANT) {
    // If the input tensor is a variant, it may represent a multi-dimensional
    // array of datasets. We attempt to decode each dataset so that we can use
    // their custom serialization logic and combine the result of their
    // individual serializations using the `Pack` operation.
    //
    // If this fails, we fallback to using its Variant::Encode() based
    // serialization.
    Status s = AddDatasetOrTensorHelper(ctx, t, output);
    if (s.ok()) {
      return s;
    }
  }
  if (t.dtype() == DT_RESOURCE && !ctx->is_graph_rewrite()) {
    Status s = AddResourceHelper(ctx, t, output);
    if (!errors::IsUnimplemented(s)) {
      // Fall through to AddTensor if AsGraphDef is not implemented for this
      // resource.
      return s;
    }
  }
  return AddTensor(t, output);
}

Status DatasetBase::DatasetGraphDefBuilder::AddIdentity(
    SerializationContext* ctx, const std::string& name_prefix, Node** input,
    Node** output) {
  *output =
      ops::UnaryOp("Identity", *input,
                   builder()->opts().WithName(UniqueNodeName(name_prefix)));
  return OkStatus();
}

Status DatasetBase::DatasetGraphDefBuilder::AddDatasetOrTensorHelper(
    SerializationContext* ctx, const Tensor& t, Node** output) {
  if (t.dims() == 0) {
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(t, &dataset));
    return AddInputDataset(ctx, dataset, output);
  }
  std::vector<NodeBuilder::NodeOut> nodes;
  for (int i = 0; i < t.dim_size(0); ++i) {
    Node* node;
    TF_RETURN_IF_ERROR(AddDatasetOrTensorHelper(ctx, t.SubSlice(i), &node));
    nodes.emplace_back(node);
  }
  auto op_name = "Pack";
  auto opts = builder()->opts();
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  node_builder.Input(std::move(nodes));
  *output = opts.FinalizeBuilder(&node_builder);
  return OkStatus();
}

Status DatasetBase::DatasetGraphDefBuilder::AddResourceHelper(
    SerializationContext* ctx, const Tensor& t, Node** output) {
  if (t.NumElements() == 0) {
    return errors::InvalidArgument("Empty resouce handle");
  }
  const ResourceHandle& handle = t.flat<ResourceHandle>()(0);
  if (!DeviceNameUtils::HaveSameDeviceName(ctx->device_name(),
                                           handle.device())) {
    return errors::InvalidArgument("Trying to access resource ", handle.name(),
                                   " located in device ", handle.device(),
                                   " from device ", ctx->device_name());
  }
  ResourceBase* resource;
  TF_RETURN_IF_ERROR(ctx->resource_mgr()->Lookup(handle, &resource));
  core::ScopedUnref unref(resource);
  return resource->AsGraphDef(builder(), output);
}

DatasetBaseIterator::DatasetBaseIterator(const BaseParams& params)
    : params_(params) {
  params_.dataset->Ref();
  VLOG(2) << prefix() << " constructor";
  strings::StrAppend(&traceme_metadata_, "name=", dataset()->metadata().name());
  strings::StrAppend(&traceme_metadata_, ",shapes=");
  auto& shapes = output_shapes();
  for (int i = 0; i < shapes.size(); ++i) {
    if (i > 0) {
      strings::StrAppend(&traceme_metadata_, " ");
    }
    strings::StrAppend(&traceme_metadata_, shapes.at(i).DebugString());
  }
  strings::StrAppend(&traceme_metadata_, ",types=");
  auto& types = output_dtypes();
  for (int i = 0; i < types.size(); ++i) {
    if (i > 0) {
      strings::StrAppend(&traceme_metadata_, " ");
    }
    strings::StrAppend(&traceme_metadata_, DataTypeString(types.at(i)));
  }
}

DatasetBaseIterator::~DatasetBaseIterator() {
  VLOG(2) << prefix() << " destructor";
  params_.dataset->Unref();
}

string DatasetBaseIterator::BuildTraceMeName() {
  string result =
      strings::StrCat(params_.prefix, "#", traceme_metadata_, ",id=", id_);
  if (parent_) {
    strings::StrAppend(&result, ",parent_id=", parent_id_);
  }
  TraceMeMetadata metadata = GetTraceMeMetadata();
  for (const auto& pair : metadata) {
    strings::StrAppend(&result, ",", pair.first, "=", pair.second);
  }
  if (model_node() != nullptr) {
    if (model_node()->buffered_elements() > 0) {
      strings::StrAppend(
          &result, ",buffered_elements=",
          static_cast<long long>(model_node()->buffered_elements()));
      strings::StrAppend(
          &result, ",buffered_bytes_MB=",
          static_cast<long long>(
              static_cast<double>(model_node()->buffered_bytes()) * 1e-6));
    }
  }
  strings::StrAppend(&result, "#");
  return result;
}

Status DatasetBaseIterator::GetNext(IteratorContext* ctx,
                                    std::vector<Tensor>* out_tensors,
                                    bool* end_of_sequence) {
  activity_watcher::ActivityScope activity_scope([&]() {
    activity_watcher::Activity::Attributes attributes;
    attributes["iterator_prefix"] = prefix();
    return std::make_unique<activity_watcher::Activity>(
        "Iterator::GetNext", activity_watcher::ActivityCategory::kDatasetOp,
        std::move(attributes));
  });
  profiler::TraceMe activity([&] { return BuildTraceMeName(); },
                             profiler::TraceMeLevel::kInfo);
  DVLOG(3) << prefix() << " GetNext enter";
  auto model = ctx->model();
  bool output_was_recording =
      node_ && node_->output() && node_->output()->is_recording();
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    if (output_was_recording) {
      node_->output()->record_stop(now_nanos);
    }
    node_->record_start(now_nanos);
  }
  out_tensors->clear();
  Status s = GetNextInternal(ctx, out_tensors, end_of_sequence);
  ctx->SaveCheckpoint(this);
  if (!SymbolicCheckpointCompatible()) {
    ctx->UpdateCheckpointStatus([this]() {
      return errors::Unimplemented(dataset()->type_string(),
                                   " does not support symbolic checkpointing.");
    });
  }
  if (TF_PREDICT_TRUE(s.ok())) {
    if (TF_PREDICT_TRUE(!*end_of_sequence)) {
      DCHECK_EQ(out_tensors->size(), dataset()->output_dtypes().size());
      RecordElement(ctx, out_tensors);
    } else {
      out_tensors->clear();
    }
  }
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    node_->record_stop(now_nanos);
    if (output_was_recording) {
      node_->output()->record_start(now_nanos);
    }
  }
  if (TF_PREDICT_FALSE(errors::IsOutOfRange(s))) {
    s = errors::Internal("Iterator \"", params_.prefix,
                         "\" returned `OutOfRange`. This indicates an "
                         "implementation error as `OutOfRange` errors are not "
                         "expected to be returned here. Original message: ",
                         s.message());
    LOG(ERROR) << s;
  }
  DVLOG(3) << prefix() << " GetNext exit";
  return s;
}

Status DatasetBaseIterator::Skip(IteratorContext* ctx, int num_to_skip,
                                 bool* end_of_sequence, int* num_skipped) {
  profiler::TraceMe activity([&] { return BuildTraceMeName(); },
                             profiler::TraceMeLevel::kInfo);
  DVLOG(3) << prefix() << " Skip enter";
  auto model = ctx->model();
  bool output_was_recording =
      node_ && node_->output() && node_->output()->is_recording();
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    auto output = node_->output();
    if (output_was_recording) {
      output->record_stop(now_nanos);
    }
    node_->record_start(now_nanos);
  }
  Status s = SkipInternal(ctx, num_to_skip, end_of_sequence, num_skipped);
  if (collect_resource_usage(ctx)) {
    int64_t now_nanos = EnvTime::NowNanos();
    node_->record_stop(now_nanos);
    auto output = node_->output();
    if (output_was_recording) {
      output->record_start(now_nanos);
    }
  }
  if (TF_PREDICT_FALSE(errors::IsOutOfRange(s))) {
    s = errors::Internal("Iterator \"", params_.prefix,
                         "\" returned `OutOfRange`. This indicates an "
                         "implementation error as `OutOfRange` errors are not "
                         "expected to be returned here. Original message: ",
                         s.message());
    LOG(ERROR) << s;
  }
  DVLOG(3) << prefix() << " Skip exit";
  return s;
}

Status DatasetBaseIterator::SkipInternal(IteratorContext* ctx, int num_to_skip,
                                         bool* end_of_sequence,
                                         int* num_skipped) {
  *num_skipped = 0;
  for (int i = 0; i < num_to_skip; ++i) {
    std::vector<Tensor> out_tensors;
    TF_RETURN_IF_ERROR(GetNextInternal(ctx, &out_tensors, end_of_sequence));
    if (*end_of_sequence) {
      return OkStatus();
    }
    // RecordElement is used to count the number of element computed and
    // help calculate the CPU time spent on a given iterator to do the
    // autotuning.
    // Here we only call RecordElement in the default implementation of
    // SkipInternal (which trivially calls GetNextInternal) and assume
    // that the overridden SkipInternal in the derived class will have
    // negligible cost compare to its GetNextInternal.
    RecordElement(ctx, &out_tensors);
    (*num_skipped)++;
  }
  return OkStatus();
}

void DatasetOpKernel::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset = nullptr;
  MakeDataset(ctx, &dataset);
  if (ctx->status().ok()) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, StoreDatasetInVariantTensor(dataset, output));
    if (ctx->stack_trace().has_value() && VLOG_IS_ON(4)) {
      VLOG(4) << "Dataset " << dataset->type_string()
              << " created using the following stack trace:";
      for (const auto& stack_frame : ctx->stack_trace()->ToStackFrames(
               {}, {}, /*reverse_traversal=*/false, /*limit=*/-1)) {
        VLOG(4) << stack_frame.file_name << ":" << stack_frame.line_number
                << " in " << stack_frame.function_name << "()";
      }
    }
    dataset->Initialize(metadata_);
  }
}

string DatasetOpKernel::TraceString(const OpKernelContext& ctx,
                                    bool verbose) const {
  return profiler::TraceMeOp(name_view(), type_string_view());
}

// static
bool DatasetOpKernel::IsDatasetOp(const OpDef& op_def) {
  if (op_def.output_arg_size() != 1) return false;
  if (op_def.output_arg(0).type() != DT_VARIANT) return false;
  absl::string_view op_name = op_def.name();

  // When running eager ops as a function, we check if the current op is a
  // Dataset op by unwrapping it. Below are some example op names when running
  // eager ops as a function:
  // 1. __wrapped__MapDataset_Targuments_0_device<...>
  // 2. __wrapped__FlatMapDataset_Targuments_0_device<...>
  // 3. __wrapped__ParallelMapDatasetV2_Targuments_0_device<...>
  //
  // Below are the corresponding unwrapped op names:
  // 1. MapDataset
  // 2. FlatMapDataset
  // 3. ParallelMapDatasetV2

  std::vector<std::string> v1, v2;  // Declared here so that v2 outlives op_name
  if (absl::StartsWith(op_name, "__wrapped__")) {
    v1 = absl::StrSplit(op_name, "__wrapped__", absl::SkipEmpty());
    if (v1.empty()) return false;
    v2 = absl::StrSplit(v1[0], "_", absl::SkipEmpty());
    op_name = v2.empty() ? v1[0] : v2[0];
  }

  if (op_name == "DatasetFromGraph") return true;
  if (absl::EndsWith(op_name, "Dataset")) return true;

  // Check if the suffix matches "DatasetV[0-9]+".
  size_t index = op_name.length() - 1;
  while (index >= 0 && isdigit(op_name[index])) {
    index--;
  }
  constexpr absl::string_view kDatasetPrefix = "DatasetV";
  constexpr absl::string_view::size_type kPrefixLength = kDatasetPrefix.size();
  if (index < kPrefixLength - 1 || index == op_name.length() - 1) return false;
  return op_name.substr(index - kPrefixLength + 1, kPrefixLength) ==
         kDatasetPrefix;
}

void UnaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  MakeDataset(ctx, input, output);
}

void BinaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  DatasetBase* another_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(1), &another_input));
  MakeDataset(ctx, input, another_input, output);
}

const char DatasetBase::kDatasetGraphKey[] = "_DATASET_GRAPH";
const char DatasetBase::kDatasetGraphOutputNodeKey[] =
    "_DATASET_GRAPH_OUTPUT_NODE";

BackgroundWorker::BackgroundWorker(Env* env, const char* name)
    : env_(env), name_(name) {}

BackgroundWorker::~BackgroundWorker() {
  {
    mutex_lock l(mu_);
    cancelled_ = true;
  }
  cond_var_.notify_one();
  // Block until the background thread has terminated.
  //
  // NOTE(mrry): We explicitly free and join the thread here because
  // `WorkerLoop()` uses other members of this object, and so we must join
  // the thread before destroying them.
  thread_.reset();
}

void BackgroundWorker::Schedule(std::function<void()> work_item) {
  {
    mutex_lock l(mu_);
    if (!thread_) {
      thread_ = absl::WrapUnique(env_->StartThread(
          {} /* thread_options */, name_, [this]() { WorkerLoop(); }));
    }
    work_queue_.push_back(std::move(work_item));
  }
  cond_var_.notify_one();
}

void BackgroundWorker::WorkerLoop() {
  tensorflow::ResourceTagger tag(kTFDataResourceTag, "Background");
  while (true) {
    std::function<void()> work_item = nullptr;
    {
      mutex_lock l(mu_);
      while (!cancelled_ && work_queue_.empty()) {
        cond_var_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      DCHECK(!work_queue_.empty());
      work_item = std::move(work_queue_.front());
      work_queue_.pop_front();
    }
    DCHECK(work_item != nullptr);
    work_item();
  }
}

namespace {
class RunnerImpl : public Runner {
 public:
  void Run(const std::function<void()>& f) override {
    tensorflow::ResourceTagger tag(kTFDataResourceTag, "Runner");
    f();

    // NOTE: We invoke a virtual function to prevent `f` being tail-called, and
    // thus ensure that this function remains on the stack until after `f`
    // returns.
    PreventTailCall();
  }

 private:
  virtual void PreventTailCall() {}
};
}  // namespace

/* static */
Runner* Runner::get() {
  static Runner* singleton = new RunnerImpl;
  return singleton;
}

}  // namespace data
}  // namespace tensorflow
