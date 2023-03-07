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
#ifndef TENSORFLOW_CORE_FRAMEWORK_DATASET_H_
#define TENSORFLOW_CORE_FRAMEWORK_DATASET_H_

#include <deque>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/dataset_metadata.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/dataset_stateful_op_allowlist.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/thread_factory.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

// Polymorphic datasets should support all primitive TensorFlow
// types. Use this macro to expand `m(T)` once for each primitive type
// `T`, e.g. to build a `switch` statement.
#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)

namespace tensorflow {

// Forward declarations to avoid introducing a dependency on headers in
// "tensorflow/core/graph/...".
class GraphDefBuilder;
class Node;

namespace data {

namespace internal {
// Merges Options from source to destination. If there is a conflict on a field,
// the field value from the source takes precedence.
void MergeOptions(const protobuf::Message& source,
                  protobuf::Message* destination);
void MergeOptions(const protobuf::MessageLite& source,
                  protobuf::MessageLite* destination);
}  // namespace internal

using TraceMeMetadata = std::vector<std::pair<StringPiece, string>>;

constexpr char kTFDataFunction[] = "_tf_data_function";

constexpr int kInfiniteCardinality = -1;
constexpr int kUnknownCardinality = -2;

// This constant is a magic number that is used (as a prefix) to identify keys
// used for serialization of iterator state.
constexpr char kFullNameRandomHex[] = "60d899aa0d8ce4351e7c3b419e92d25b";
constexpr int kFullNameRandomHexLen = std::size(kFullNameRandomHex) - 1;
constexpr char kPipe[] = "|";
constexpr char kColon[] = ":";

constexpr char kTFDataResourceTag[] = "tfdata";
constexpr char kTraceInfoUnavailable[] = "unavailable";
constexpr char kMetadata[] = "metadata";

constexpr char kCardinalityAttrForRewrite[] = "_cardinality";

class DatasetBase;
class IteratorContext;
class SerializationContext;

inline bool IsTFDataFunction(const FunctionDef& func) {
  auto iter = func.attr().find(data::kTFDataFunction);
  return (iter != func.attr().end() && iter->second.b());
}

// Interface for reading values from a key-value store.
// Used for restoring iterator state. This class is thread safe.
// Please see comment on IteratorStateWriter for guidance around using the
// Read*(key, val) vs Read*(name, key, val).
class IteratorStateReader {
 public:
  // Determines whether the iterator state contains the given key.
  virtual bool Contains(StringPiece key) const = 0;
  virtual bool Contains(StringPiece name, StringPiece key) const = 0;

  // Reads an integer for the given key.
  virtual Status ReadScalar(StringPiece key, int64_t* val) const = 0;
  virtual Status ReadScalar(StringPiece name, StringPiece key,
                            int64_t* val) const = 0;

  // Reads a string for the given key.
  virtual Status ReadScalar(StringPiece key, tstring* val) const = 0;
  virtual Status ReadScalar(StringPiece name, StringPiece key,
                            tstring* val) const = 0;

  // Reads a tensor for the given key.
  // TODO(jsimsa): Remove non-FLR overrides once all callers are updated.
  virtual Status ReadTensor(StringPiece key, Tensor* val) const = 0;
  virtual Status ReadTensor(FunctionLibraryRuntime* flr, StringPiece key,
                            Tensor* val) const = 0;
  virtual Status ReadTensor(StringPiece name, StringPiece key,
                            Tensor* val) const = 0;
  virtual Status ReadTensor(FunctionLibraryRuntime* flr, StringPiece name,
                            StringPiece key, Tensor* val) const = 0;

  virtual ~IteratorStateReader() {}
};

// Interface for writing values to a key-value store.
// Used for saving iterator state. Not thread safe.
// The IteratorStateWriter creates a tensor for each unique iterator name it
// sees. For the Write*(key, val) API's the key is expected to encode this
// name as keys are required to be produced using the full_name() method.
// Each tensor has an upper limit of 2 GB and so if the state for an iterator
// might exceed the 2 GB limit, you can pass an explicit name in via the
// Write*(name, key, val) APIs allowing you to further split up the state
// into more manageable chunks.
class IteratorStateWriter {
 public:
  // Writes an integer for the given key.
  virtual Status WriteScalar(StringPiece key, const int64_t val) = 0;
  virtual Status WriteScalar(StringPiece name, StringPiece key,
                             const int64_t val) = 0;

  // Writes a string for the given key.
  virtual Status WriteScalar(StringPiece key, const tstring& val) = 0;
  virtual Status WriteScalar(StringPiece name, StringPiece key,
                             const tstring& val) = 0;

  // Writes a tensor for the given key.
  virtual Status WriteTensor(StringPiece key, const Tensor& val) = 0;
  virtual Status WriteTensor(StringPiece name, StringPiece key,
                             const Tensor& val) = 0;

  virtual ~IteratorStateWriter() {}
};

// Generates a full name key for iterator checkpointing. All keys generated for
// iterator checkpoints should go through this function.
std::string FullName(const std::string& prefix, const std::string& name);

// Interface for objects that can be checkpointed.
class Checkpointable {
 public:
  Checkpointable() = default;
  virtual ~Checkpointable() = default;

  virtual Status Save(SerializationContext* ctx,
                      IteratorStateWriter* writer) = 0;
  virtual Status Restore(IteratorContext* ctx, IteratorStateReader* reader) = 0;
};

// Wrapper around GraphDefBuilder. Used to serialize Dataset graph.
class GraphDefBuilderWrapper {
 public:
  explicit GraphDefBuilderWrapper(GraphDefBuilder* b) : b_(b) {}

  // Adds a Const node with scalar value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  template <typename T>
  Status AddScalar(const T& val, Node** output) {
    Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
    val_t.scalar<T>()() = val;
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddScalar: Failed to build Const op.");
    }
    return OkStatus();
  }

  // Adds a Const node with vector value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  // TODO(shivaniagrawal): Consider changing to gtl::ArraySlice?
  template <typename T>
  Status AddVector(const std::vector<T>& val, Node** output) {
    Tensor val_t = Tensor(DataTypeToEnum<T>::v(),
                          TensorShape({static_cast<int64_t>(val.size())}));
    for (size_t i = 0; i < val.size(); i++) {
      val_t.flat<T>()(i) = val[i];
    }
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddVector: Failed to build Const op.");
    }
    return OkStatus();
  }

  Status AddVector(const std::vector<string>& val, Node** output) {
    Tensor val_t = Tensor(DataTypeToEnum<tstring>::v(),
                          TensorShape({static_cast<int64_t>(val.size())}));
    for (size_t i = 0; i < val.size(); i++) {
      val_t.flat<tstring>()(i) = val[i];
    }
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddVector: Failed to build Const op.");
    }
    return OkStatus();
  }

  // Adds a `Const` node for the given tensor value to the graph.
  //
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status. The returned `Node`
  // pointer is owned by the backing graph of `GraphDefBuilder`.
  Status AddTensor(const Tensor& val, Node** output) {
    AddTensorInternal(val, output);
    if (*output == nullptr) {
      return errors::Internal("AddTensor: Failed to build Const op.");
    }
    return OkStatus();
  }

  // Adds a `Placeholder` node for the given tensor value to the graph.
  //
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status. The returned `Node`
  // pointer is owned by the backing graph of `GraphDefBuilder`.
  Status AddPlaceholder(const Tensor& val, Node** output) {
    AddPlaceholderInternal(val, output);
    if (*output == nullptr) {
      return errors::Internal(
          "AddPlaceholder: Failed to build Placeholder op.");
    }
    return OkStatus();
  }

  // Adds a node for the given dataset to the `Graph`. The value of
  // `DatasetBase::type_string()` is used as the op type for the node. Values
  // for the `output_types` and `output_shapes` node attributes are also written
  // if those attributes are defined in the `OpDef`.
  //
  // If `use_dataset_name` is set, the value of `DatasetBase::node_name()` is
  // used as the op name for the node. This argument should only be set when
  // serializing `DatasetBase` instances which might not have been created
  // through op kernel execution to make sure the dataset op name is preserved
  // across serialization boundaries, which is in turn needed to make sure
  // iterator checkpoints are valid across serialization boundaries. When
  // `use_dataset_name` is set, the caller is responsible for making sure that
  // the op name is unique across the graph.
  //
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status. The returned `Node`
  // pointer is owned by the backing `Graph` of `GraphDefBuilder`.
  Status AddDataset(const DatasetBase* dataset,
                    const std::vector<Node*>& inputs, Node** output);
  Status AddDataset(const DatasetBase* dataset,
                    const std::vector<Node*>& inputs,
                    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
                    Node** output);
  Status AddDataset(
      const DatasetBase* dataset,
      const std::vector<std::pair<size_t, Node*>>& inputs,
      const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
      const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
      Node** output);
  Status AddDataset(
      const DatasetBase* dataset,
      const std::vector<std::pair<size_t, Node*>>& inputs,
      const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
      const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
      bool use_dataset_name, Node** output);

  // Adds a user-defined function with name `function_name` to the graph and
  // recursively adds all functions it references. If a function with a matching
  // name has already been added, returns with OK status. If a user-defined with
  // name `function_name` is not found in the context's function library,
  // returns an InvalidArgumentError. If the function with name `function_name`
  // or any of its dependent functions are stateful, and the context does not
  // explicitly permit stateful functions, returns an InvalidArgument error.
  Status AddFunction(SerializationContext* ctx, const string& function_name,
                     const FunctionLibraryDefinition& lib_def);

  template <typename T>
  void BuildAttrValue(const T& value, AttrValue* attr) {
    SetAttrValue(value, attr);
  }

  template <typename T>
  AttrValue BuildAttrValue(const T& value) {
    AttrValue attr;
    SetAttrValue(value, &attr);
    return attr;
  }

 protected:
  GraphDefBuilder* builder() { return b_; }

 private:
  void AddPlaceholderInternal(const Tensor& val, Node** output);
  void AddTensorInternal(const Tensor& val, Node** output);
  bool HasAttr(const string& op_type_name, const string& attr_name) const;

  bool HasAttr(const OpDef* op_def, const string& attr_name) const {
    for (const auto& attr : op_def->attr()) {
      if (attr.name() == attr_name) {
        return true;
      }
    }
    return false;
  }

  Status AddAttrFunctions(SerializationContext* ctx,
                          const AttrValue& attr_value,
                          const FunctionLibraryDefinition& lib_def) {
    if (attr_value.has_func()) {
      TF_RETURN_IF_ERROR(AddFunction(ctx, attr_value.func().name(), lib_def));
    } else if (attr_value.has_list()) {
      for (const NameAttrList& name_attr_list : attr_value.list().func()) {
        TF_RETURN_IF_ERROR(AddFunction(ctx, name_attr_list.name(), lib_def));
      }
    }
    return OkStatus();
  }

  GraphDefBuilder* b_;
};

class StatsAggregator;

// A utility class for running a function and ensuring that there is always a
// `tensorflow::data` symbol on the stack.
class Runner {
 public:
  virtual ~Runner() {}

  // Runs the given function.
  virtual void Run(const std::function<void()>& f) = 0;

  // Returns a global singleton Runner.
  static Runner* get();
};

// A class which provides a sequence of splits. Splits represent subdivisions of
// a dataset, e.g. filenames or ranges within files. We use splitting to
// partition input data into smaller pieces for distributed processing (see
// go/tf-data-splitting-design).
//
// Datasets provide a `MakeSplitProvider` method to expose a listing of their
// splits.
//
// Iterators created with a split provider will only iterate over the splits
// provided by the split provider.
class SplitProvider {
 public:
  virtual ~SplitProvider() {}
  // Stores the next split in `*split`, setting `*end_of_splits` to indicate
  // whether there were any splits left.
  virtual Status GetNext(Tensor* split, bool* end_of_splits) = 0;
  // Resets the split provider to its beginning.
  virtual Status Reset() = 0;
  // Saves the state of this split provider.
  virtual Status Save(std::function<std::string(std::string)> full_name,
                      IteratorStateWriter* writer) = 0;
  // Restores the state of this split provider.
  virtual Status Restore(std::function<std::string(std::string)> full_name,
                         IteratorStateReader* reader) = 0;
};

// Returns the runner threadpool size from an OpKernelContext.
int32_t GetRunnerThreadpoolSizeFromOpKernelContext(OpKernelContext* ctx);

// In-memory representation of a checkpoint. The checkpoint is represented as a
// collection of key-value pairs and are expected to be written using the
// `IteratorStateWriter` interface.
//
// The implementation is not thread-safe.
class MemoryCheckpoint : public IteratorStateWriter {
 public:
  // IdRegistry maintains the mapping between a string key and an integer.
  // The main purpose of this registry is to allow us using integers as map keys
  // in MemoryCheckpoint to reduce the cost in checkpoint merging.
  class IdRegistry {
   public:
    IdRegistry() = default;

    // Inserts the key into the registry and get the integer id for the key.
    // If the key already exists in the registry, the corresponding id is
    // directly returned.
    int64_t InsertKey(const std::string& key) {
      mutex_lock l(mu_);
      if (key_to_id_.contains(key)) {
        return key_to_id_[key];
      }
      int64_t id = next_id_++;
      id_to_key_[id] = key;
      key_to_id_[key] = id;
      return id;
    }

    // Gets all ids for keys starting with the given prefix.
    std::vector<int64_t> GetIdsWithPrefix(const std::string& prefix) {
      mutex_lock l(mu_);
      std::vector<int64_t> ids;
      for (const auto& [key, id] : key_to_id_) {
        if (key.length() >= kFullNameRandomHexLen + 1 + prefix.length() &&
            key.compare(kFullNameRandomHexLen + 1, prefix.length(), prefix) ==
                0) {
          ids.push_back(id);
        }
      }
      return ids;
    }

    // Gets the key corresponding to the given id.
    std::string GetKey(int64_t id) {
      mutex_lock l(mu_);
      if (!id_to_key_.contains(id)) {
        LOG(ERROR) << "Failed find key in IdRegistry: " << id
                   << ", max id is: " << next_id_ - 1;
      }
      return id_to_key_[id];
    }

    // Removes the given ids from the registry along with their corresponding
    // keys.
    void RemoveIds(const std::vector<int64_t>& ids) {
      mutex_lock l(mu_);
      for (const auto& id : ids) {
        key_to_id_.erase(id_to_key_[id]);
        id_to_key_.erase(id);
      }
    }

   private:
    mutex mu_;
    int64_t next_id_ TF_GUARDED_BY(mu_) = 0;
    absl::flat_hash_map<int64_t, std::string> id_to_key_ TF_GUARDED_BY(mu_);
    absl::flat_hash_map<std::string, int64_t> key_to_id_ TF_GUARDED_BY(mu_);
  };

  MemoryCheckpoint() = delete;
  explicit MemoryCheckpoint(std::shared_ptr<IdRegistry> registry)
      : id_registry_(registry) {}

  MemoryCheckpoint(MemoryCheckpoint&& other) = default;

  static MemoryCheckpoint CreateRootCheckpoint(
      std::shared_ptr<IdRegistry> registry) {
    return MemoryCheckpoint(/*id_registry*/ registry, /*is_root=*/true);
  }

  // BEGIN implementation of `IteratorStateWriter` interface
  Status WriteScalar(StringPiece key, int64_t val) override {
    auto id = id_registry_->InsertKey(string(key));
    int_values_[id] = val;
    return OkStatus();
  }
  Status WriteScalar(StringPiece name, StringPiece key, int64_t val) override {
    return WriteScalar(FullName(string(name), string(key)), val);
  }
  Status WriteScalar(StringPiece key, const tstring& val) override {
    auto id = id_registry_->InsertKey(string(key));
    str_values_[id] = val;
    return OkStatus();
  }
  Status WriteScalar(StringPiece name, StringPiece key,
                     const tstring& val) override {
    return WriteScalar(FullName(string(name), string(key)), val);
  }
  Status WriteTensor(StringPiece key, const Tensor& val) override {
    auto id = id_registry_->InsertKey(string(key));
    tensor_values_[id] = val;
    return OkStatus();
  }
  Status WriteTensor(StringPiece name, StringPiece key,
                     const Tensor& val) override {
    return WriteTensor(FullName(string(name), string(key)), val);
  }
  // END implementation of `IteratorStateWriter` interface

  // String representation for the in-memory checkpoint suitable for debugging.
  std::string DebugString() const {
    std::string result = absl::StrCat("status=", status_.ToString(),
                                      ", "
                                      "root=",
                                      (is_root_ ? "true" : "false"), "\n");
    absl::StrAppend(&result, "number of integers: ", int_values_.size(), "\n");

    absl::StrAppend(&result, "number of strings: ", str_values_.size(), "\n");
    absl::StrAppend(&result, "number of tensors: ", tensor_values_.size(),
                    "\n");

    absl::StrAppend(&result,
                    "number of expired prefixes: ", expired_prefixes_.size(),
                    "\n");
    return result;
  }

  // Returns the status of the in-memory checkpoint.
  Status GetStatus() const { return status_; }

  // Merges key-values pair of another checkpoint with this checkpoint. If a key
  // exists with another checkpoint, then the key-value pair from the `other`
  // argument is used.
  //
  // Merge also garbage collects expired prefixes.
  void Merge(MemoryCheckpoint* other) {
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

  // Purge removes all keys with given prefix from checkpoint. It also adds the
  // prefix for tracking unless it is the root checkpoint.
  void Purge(const std::string& prefix) {
    std::vector<int64_t> ids = id_registry_->GetIdsWithPrefix(prefix);
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

  // Stores the in-memory checkpoint to the given writer.
  Status Save(IteratorStateWriter* writer) const {
    for (const auto& [id, value] : int_values_) {
      auto key = id_registry_->GetKey(id);
      TF_RETURN_IF_ERROR(writer->WriteScalar(key, value));
    }
    for (const auto& [id, value] : str_values_) {
      auto key = id_registry_->GetKey(id);
      TF_RETURN_IF_ERROR(writer->WriteScalar(key, value));
    }
    for (const auto& [id, value] : tensor_values_) {
      auto key = id_registry_->GetKey(id);
      TF_RETURN_IF_ERROR(writer->WriteTensor(key, value));
    }
    return OkStatus();
  }

  // Updates the status of the in-memory checkpoint with the given status.
  void UpdateStatus(Status status) { status_.Update(status); }

 private:
  explicit MemoryCheckpoint(std::shared_ptr<IdRegistry> registry, bool is_root)
      : is_root_(is_root), id_registry_(registry) {}
  TF_DISALLOW_COPY_AND_ASSIGN(MemoryCheckpoint);

  Status status_ = OkStatus();
  // Only set to true for the checkpoint in IteratorResource.
  // Root checkpoint does not track expired prefixes.
  const bool is_root_ = false;
  absl::flat_hash_map<int64_t, int64_t> int_values_;
  absl::flat_hash_map<int64_t, std::string> str_values_;
  absl::flat_hash_map<int64_t, Tensor> tensor_values_;

  // Keeps track of expired prefixes for propagation. Cleaned after it's merged.
  absl::flat_hash_set<std::string> expired_prefixes_;

  std::shared_ptr<IdRegistry> id_registry_;
};

// Aggregates runtime support needed for dataset and iterator serialization.
class SerializationContext {
 public:
  // Handles the external state according to the external state policy.
  Status HandleCheckExternalStateStatus(Status s) {
    if (s.ok()) {
      return s;
    }
    switch (params_.external_state_policy) {
      case ExternalStatePolicy::POLICY_WARN:
        LOG(WARNING) << s.ToString();
        return OkStatus();
      case ExternalStatePolicy::POLICY_IGNORE:
        VLOG(2) << "Ignoring error status: " << s.ToString();
        return OkStatus();
      case ExternalStatePolicy::POLICY_FAIL:
        return s;
      default:
        return errors::InvalidArgument("Unexpected value of external policy: ",
                                       params_.external_state_policy);
    }
  }

  struct Params {
    explicit Params() = default;

    explicit Params(OpKernelContext* ctx)
        : resource_mgr(ctx->resource_manager()),
          device_name(ctx->device()->attributes().name()) {}

    std::vector<std::pair<string, Tensor>>* input_list = nullptr;  // Not owned.

    // Indicates what to do if the dataset depends on external state.
    ExternalStatePolicy external_state_policy =
        ExternalStatePolicy::POLICY_WARN;

    // Indicates whether the serialization is for rewrites.
    //
    // If true:
    //   * A dataset that doesn't implement serialization is replaced with a
    //     placeholder returned in `input_list`.
    //   * Data tensors are replaced with a placeholder returned in
    //     `input_list`.
    //   * Datasets that use random seeds should not serialize the random seeds.
    //     This doesn't affect datasets that use fixed seeds; fixed seeds will
    //     always be preserved.
    //   * Cardinality is serialized as an unregistered attribute
    //     `_cardinality`.
    // If false:
    //   * A dataset that doesn't implement serialization should result in an
    //     error.
    //   * Data tensors (potentially large) should be serialized.
    //   * Datasets that use random seeds should serialize the random seeds.
    bool is_graph_rewrite = false;

    // A resource manager for looking up resources during serialization.
    ResourceMgr* resource_mgr;

    // The name of the device doing the serialization.
    std::string device_name;

    // Determines whether checkpointing should represent input pipeline state
    // symbolically, using cursors into source iterators, or explicitly, by
    // storing internal state of each iterator.
    bool symbolic_checkpoint = false;
  };

  explicit SerializationContext(Params params) : params_(params) {}

  std::vector<std::pair<string, Tensor>>* input_list() {
    return params_.input_list;
  }

  ExternalStatePolicy external_state_policy() const {
    return params_.external_state_policy;
  }

  bool is_graph_rewrite() const { return params_.is_graph_rewrite; }

  const ResourceMgr* resource_mgr() const { return params_.resource_mgr; }

  const std::string& device_name() const { return params_.device_name; }

  bool symbolic_checkpoint() const { return params_.symbolic_checkpoint; }

 private:
  Params params_;

  TF_DISALLOW_COPY_AND_ASSIGN(SerializationContext);
};

// A cut-down version of `OpKernelContext` for running computations in
// iterators. Note that we cannot simply use `OpKernelContext` here because we
// might run computation in an iterator whose lifetime is not nested within the
// lifetime of a single `OpKernelContext` (e.g. asynchronous prefetching).
//
// TODO(mrry): We're making some daring assumptions about the lifetime of the
// runner passed in here. A runner will be deleted when the original step ends,
// but all existing runners only close over session-lifetime (or longer-lived)
// state, so we can make a copy of the function. There's nothing in the
// definition of the API from which we took the runner to guarantee that what we
// are doing is safe. We should formalize the properties here.
class IteratorContext {
 public:
  struct Params {
    explicit Params(IteratorContext* ctx)
        : allocator_getter(ctx->allocator_getter()),
          cancellation_manager(ctx->cancellation_manager()),
          collective_executor(ctx->collective_executor()),
          env(ctx->env()),
          flr(ctx->flr()),
          function_handle_cache(ctx->function_handle_cache()),
          interleave_depth(ctx->interleave_depth()),
          is_restoring(ctx->is_restoring()),
          model(ctx->model()),
          resource_mgr(ctx->resource_mgr()),
          runner(*(ctx->runner())),
          runner_threadpool_size(ctx->runner_threadpool_size()),
          split_providers(ctx->split_providers()),
          stats_aggregator(ctx->stats_aggregator()),
          symbolic_checkpoint(ctx->symbolic_checkpoint()),
          thread_factory(ctx->thread_factory()),
          thread_pool(ctx->thread_pool()),
          id_registry(ctx->id_registry()),
          warm_start(ctx->warm_start()) {}

    explicit Params(OpKernelContext* ctx)
        : collective_executor(ctx->collective_executor()),
          env(ctx->env()),
          flr(ctx->function_library()) {
      // NOTE: need reinterpret_cast because function.h forward-declares Device.
      DeviceBase* device =
          reinterpret_cast<DeviceBase*>(ctx->function_library()->device());
      allocator_getter = [device](AllocatorAttributes attrs) {
        return device->GetAllocator(attrs);
      };

      runner_threadpool_size = GetRunnerThreadpoolSizeFromOpKernelContext(ctx);

      // NOTE: Wrap every runner invocation in a call to Runner()->Run(), so
      // that a symbol in the tensorflow::data namespace is always on the stack
      // when executing a function inside a Dataset.
      runner = std::bind(
          [](
              // Note: `runner` is a const reference to avoid copying it.
              const std::function<void(std::function<void()>)>& ctx_runner,
              std::function<void()> fn) {
            std::function<void()> wrapped_fn = std::bind(
                [](const std::function<void()>& fn) { Runner::get()->Run(fn); },
                std::move(fn));
            ctx_runner(std::move(wrapped_fn));
          },
          *ctx->runner(), std::placeholders::_1);
    }

    // The Allocator to be used to allocate the output of an iterator.
    std::function<Allocator*(AllocatorAttributes)> allocator_getter = nullptr;

    // The CancellationManager to be used to cancel execution of ops.
    CancellationManager* cancellation_manager = nullptr;

    // Collective support.
    CollectiveExecutor* collective_executor = nullptr;

    // Interface to operating system functionality.
    Env* env = nullptr;

    // The FunctionLibraryRuntime object to be used to make function calls.
    FunctionLibraryRuntime* flr = nullptr;

    // A FunctionHandleCache that owns all the function handles. Not owned.
    FunctionHandleCache* function_handle_cache = nullptr;

    // Records the number of ParallelInterleave operations in the path from the
    // root node to this node (not including this node) in the input pipeline
    // tree.
    int64 interleave_depth = 0;

    // Marks whether the iterator is restored from a checkpoint.
    bool is_restoring = false;

    // If non-null, identifies the object used for performance modeling.
    std::shared_ptr<model::Model> model = nullptr;

    // The input pipeline options.
    const Options* options = nullptr;

    // A resource manager for storing dataset-related state, e.g. random
    // seeds or cached tensors. Not owned.
    ResourceMgr* resource_mgr = nullptr;

    // Function call support.
    std::function<void(std::function<void()>)> runner = nullptr;

    // Number of threads used for executing user-defined functions.
    int32 runner_threadpool_size = 0;

    // Split providers indicating which splits to process. May be empty,
    // indicating that the iterator should process all splits.
    std::vector<std::shared_ptr<SplitProvider>> split_providers;

    // The `StatsAggregator` object to record statistics about the iterator.
    //
    // TODO(b/147325552): Remove this API and any of its uses after we switch to
    // using C++ based implementation for tf.data options (on 4/12/2021).
    std::shared_ptr<StatsAggregator> stats_aggregator = nullptr;

    // Indicates whether to use symbolic checkpointing.
    bool symbolic_checkpoint = false;

    // A factory for creating threads to perform blocking work.
    std::shared_ptr<ThreadFactory> thread_factory = nullptr;

    // A shared thread pool to schedule computation into.
    thread::ThreadPoolInterface* thread_pool = nullptr;

    std::shared_ptr<MemoryCheckpoint::IdRegistry> id_registry =
        std::make_shared<MemoryCheckpoint::IdRegistry>();

    // If `true` background threads of asynchronous operations are started when
    // the iterator is created. Otherwise, they are started upon first `GetNext`
    // request. Default value is set to false to ensure backward compatibility.
    bool warm_start = false;
  };

  explicit IteratorContext(IteratorContext* ctx)
      : IteratorContext(Params{ctx}) {}

  explicit IteratorContext(OpKernelContext* ctx)
      : IteratorContext(Params{ctx}) {}

  explicit IteratorContext(Params params)
      : params_(std::move(params)),
        checkpoint_(MemoryCheckpoint{params_.id_registry}) {}

  IteratorContext(const IteratorContext& other)
      : IteratorContext(Params{other.params_}) {
    // MemoryCheckpoint should not be copied over as the child context should
    // not care what's in the checkpoint of parent context.
  }

  std::shared_ptr<MemoryCheckpoint::IdRegistry> id_registry() {
    return params_.id_registry;
  }

  Allocator* allocator(AllocatorAttributes attrs) {
    return params_.allocator_getter(attrs);
  }

  std::function<Allocator*(AllocatorAttributes)> allocator_getter() {
    return params_.allocator_getter;
  }

  CancellationManager* cancellation_manager() {
    return params_.cancellation_manager;
  }

  CollectiveExecutor* collective_executor() {
    return params_.collective_executor;
  }

  Env* env() const { return params_.env; }

  FunctionLibraryRuntime* flr() { return params_.flr; }

  FunctionHandleCache* function_handle_cache() {
    return params_.function_handle_cache;
  }

  MemoryCheckpoint* checkpoint() { return &checkpoint_; }

  int64 interleave_depth() { return params_.interleave_depth; }

  bool is_restoring() { return params_.is_restoring; }

  const std::shared_ptr<model::Model>& model() { return params_.model; }

  ResourceMgr* resource_mgr() { return params_.resource_mgr; }

  std::function<void(std::function<void()>)>* runner() {
    return &params_.runner;
  }

  int32 runner_threadpool_size() { return params_.runner_threadpool_size; }

  std::vector<std::shared_ptr<SplitProvider>> split_providers() {
    return params_.split_providers;
  }

  std::shared_ptr<StatsAggregator> stats_aggregator() {
    return params_.stats_aggregator;
  }

  bool symbolic_checkpoint() { return params_.symbolic_checkpoint; }

  const std::shared_ptr<ThreadFactory>& thread_factory() {
    return params_.thread_factory;
  }

  thread::ThreadPoolInterface* thread_pool() { return params_.thread_pool; }

  bool warm_start() { return params_.warm_start; }

  std::unique_ptr<thread::ThreadPool> CreateThreadPool(const string& name,
                                                       int num_threads) {
    if (params_.thread_pool) {
      // Create a `ThreadPool` instance by wrapping `params_.thread_pool` (which
      // is an instance of `thread::ThreadPoolInterface`). Notably, the
      // ownership of `params_.thread_pool` is *not* transferred onto the newly
      // created `ThreadPool` instance.
      return absl::make_unique<thread::ThreadPool>(params_.thread_pool);
    } else {
      return absl::make_unique<thread::ThreadPool>(params_.env, ThreadOptions(),
                                                   name, num_threads,
                                                   /*low_latency_hint=*/false);
    }
  }

  // Merges the given checkpoint with the checkpoint of this context.
  //
  // The intended for this API is that methods, such as
  // `IteratorBase::Initialize`, `IteratorBase::GetNextInternal`, or
  // `IteratorBase::RestoreInternal` that store data in the in-memory
  // checkpoint, use a separate instance of `IteratorContext` for a nested call,
  // then the checkpoint collected by the `IteratorContext` instance passed into
  // the callee should be merged into the `IteratorContext` of the caller:
  //
  // ```
  // Status GetNextInternal(IteratorContext* ctx, ...) {
  //   ...
  //   IteratorContext nested_ctx(...);
  //   TF_RETURN_IF_ERROR(input_impl_->GetNext(&nested_ctx, ...));
  //   ctx->MergeCheckpoint(nested_ctx->checkpoint());
  //   ...
  // }
  // ```
  void MergeCheckpoint(MemoryCheckpoint* checkpoint) {
    if (symbolic_checkpoint()) {
      checkpoint_.Merge(checkpoint);
    }
  }

  // Removes any keys with the given prefix from the checkpoint.
  //
  // The intended use for this API is to clean the stale state in checkpoint,
  // e.g. when a pipeline created by `flat_map` is exhausted, the state
  // associated with the iterator of that pipeline is no longer needed and
  // should be removed.
  void PurgeCheckpoint(const std::string& prefix) {
    if (symbolic_checkpoint()) {
      checkpoint_.Purge(prefix);
    }
  }

  // Saves the state of the given iterator into the checkpoint.
  void SaveCheckpoint(Checkpointable* iterator) {
    if (symbolic_checkpoint()) {
      SerializationContext::Params params;
      params.symbolic_checkpoint = true;
      SerializationContext ctx(std::move(params));
      checkpoint_.UpdateStatus(iterator->Save(&ctx, &checkpoint_));
    }
  }

  std::unique_ptr<Thread> StartThread(const string& name,
                                      std::function<void()> fn) {
    if (params_.thread_factory) {
      return params_.thread_factory->StartThread(name, std::move(fn));
    } else {
      return absl::WrapUnique(
          Env::Default()->StartThread({}, name, std::move(fn)));
    }
  }

  // Updates the status of the checkpoint with the given status.
  void UpdateCheckpointStatus(std::function<Status()> status_fn) {
    if (symbolic_checkpoint()) {
      checkpoint_.UpdateStatus(status_fn());
    }
  }

 private:
  Params params_;
  MemoryCheckpoint checkpoint_;
};

// Represents the current position in a range of outputs, where the
// range of outputs is typically represented by an `DatasetBase`,
// defined below.
class IteratorBase : public Checkpointable {
 public:
  virtual ~IteratorBase() {
    for (auto rit = cleanup_fns_.rbegin(); rit != cleanup_fns_.rend(); ++rit) {
      (*rit)();
    }
  }

  // Gets the next output from the range that this iterator is traversing.
  //
  // If at least one output remains in this iterator's range, that
  // output will be stored in `*out_tensors` and `false` will be
  // stored in `*end_of_sequence`.
  //
  // If no more outputs remain in this iterator's range, `true` will be stored
  // in `*end_of_sequence`, and `*out_tensors` will be empty.
  //
  // Implementations should never return `OutOfRange` error. If at end of
  // sequence, set `*end_of_sequence = true` and return `OkStatus()`.
  // Internally raised `OutOfRange` errors that do not imply end of sequence
  // should be converted to a different error type before being propagated to
  // the caller.
  //
  // Implementations must explicitly set `*end_of_sequence = false` if an
  // `OkStatus()` status is returned and the iterator is not at the end of the
  // sequence.
  //
  // `out_tensors` and `end_of_sequence` are output parameters. `*out_tensors`
  // and `*end_of_sequence` should not be read by implementations of `GetNext`
  // before they are assigned.
  //
  // This method is thread-safe.
  //
  // TODO(mrry): Define `GetNextAsync()` or `GetNextManyAsync()`, and
  // potentially remove this method.
  virtual Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) = 0;

  Status GetNext(IteratorContext&& ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
    return GetNext(&ctx, out_tensors, end_of_sequence);
  }

  // Skips the next `num_to_skip` outputs from the range that this iterator
  // is traversing.
  //
  // If there are not enough outputs to skip, it will set
  // `*end_of_sequence = true` and return `OkStatus()`. `*num_skipped` will
  // store the number of outputs that are skipped. When `*end_of_sequence` is
  // `false`, `*num_skipped` should equal to `num_to_skip`.
  virtual Status Skip(IteratorContext* ctx, int num_to_skip,
                      bool* end_of_sequence, int* num_skipped) = 0;

  virtual Status Skip(IteratorContext&& ctx, int num_to_skip,
                      bool* end_of_sequence, int* num_skipped) {
    return Skip(&ctx, num_to_skip, end_of_sequence, num_skipped);
  }

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // iterator.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this iterator.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // Returns a string that identifies the sequence of iterators leading up to
  // this iterator.
  virtual const string& prefix() const = 0;

  // Indicates whether the iterator is compatible with symbolic checkpointing.
  virtual bool SymbolicCheckpointCompatible() const { return false; }

  // Performs initialization that needs to happen outside of a constructor to
  // properly propagate errors.
  virtual Status Initialize(IteratorContext* ctx) { return OkStatus(); }

  // Performs initialization of the base iterator.
  Status InitializeBase(IteratorContext* ctx, const IteratorBase* parent);

  // Saves the state of this iterator.
  Status Save(SerializationContext* ctx, IteratorStateWriter* writer) override {
    int64_t start_us = EnvTime::NowMicros();
    TF_RETURN_IF_ERROR(SaveInternal(ctx, writer));
    VLOG(1) << "Saved " << prefix() << " in "
            << (EnvTime::NowMicros() - start_us) << "us";
    return OkStatus();
  }

  // Restores the state of this iterator.
  Status Restore(IteratorContext* ctx, IteratorStateReader* reader) override {
    int64_t start_us = EnvTime::NowMicros();
    TF_RETURN_IF_ERROR(RestoreInternal(ctx, reader));
    ctx->SaveCheckpoint(this);
    VLOG(1) << "Restored " << prefix() << " in "
            << (EnvTime::NowMicros() - start_us) << "us";
    return OkStatus();
  }

  // Returns the total number of bytes buffered by the iterator across all nodes
  // in the subtree for which autotuning is enabled.
  int64_t TotalBufferedBytes() const {
    if (node_) return node_->TotalBufferedBytes();
    return 0;
  }

 protected:
  // Returns a node that models this iterator.
  virtual std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const = 0;

  // This is needed so that sub-classes of IteratorBase can call
  // `SaveInternal` on their input iterators.
  Status SaveInput(SerializationContext* ctx, IteratorStateWriter* writer,
                   const std::unique_ptr<IteratorBase>& input) {
    if (ctx->symbolic_checkpoint()) {
      return OkStatus();
    }
    return input->Save(ctx, writer);
  }

  // This is needed so that sub-classes of IteratorBase can call
  // `RestoreInternal` on their input iterators.
  Status RestoreInput(IteratorContext* ctx, IteratorStateReader* reader,
                      const std::unique_ptr<IteratorBase>& input) {
    return input->Restore(ctx, reader);
  }

  Status RestoreInput(IteratorContext&& ctx, IteratorStateReader* reader,
                      const std::unique_ptr<IteratorBase>& input) {
    return RestoreInput(&ctx, reader, input);
  }

  // Saves the state of this iterator.
  //
  // This method is used to store the state of the iterator in a checkpoint.
  // implementations have an override.
  virtual Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) = 0;

  // Restores the state of this iterator.
  //
  // This method is used to restore the state of the iterator from a checkpoint.
  //
  // Implementations may assume that the iterator is in a clean state. That is,
  // its `Initialize` method has been called, but its `GetNext` method has
  // never been called.
  // implementations have an override.
  virtual Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) = 0;

  // Returns a pointer to the node representing this iterator in the performance
  // model. It may be null, if performance modeling is not enabled for this
  // iterator.
  std::shared_ptr<model::Node> model_node() const { return node_; }

  // Returns the number of elements produced by this iterator.
  int64_t num_elements() const {
    if (node_) return node_->num_elements();
    return 0;
  }

 private:
  // For access to `AddCleanupFunction` and `Restore`.
  friend class DatasetBase;
  friend class DatasetBaseIterator;  // for access to `node_`

  std::vector<std::function<void()>> cleanup_fns_;
  std::shared_ptr<model::Node> node_ = nullptr;
  const IteratorBase* parent_ = nullptr;  // Not owned.
  int64_t id_ = 0;
  int64_t parent_id_ = 0;
};

// Represents runtime information needed to construct a dataset.
class DatasetContext {
 public:
  struct Params {
    string type_string;  // op type name of this dataset.
    string node_name;    // graph node name of this dataset op, uniquely
                         // identifying the dataset in the graph.
  };

  explicit DatasetContext(Params params) : params_(std::move(params)) {}

  explicit DatasetContext(OpKernelContext* ctx) {
    params_.type_string = ctx->op_kernel().type_string();
    params_.node_name = ctx->op_kernel().name();
  }

  const string& type_string() const { return params_.type_string; }
  const string& node_name() const { return params_.node_name; }

 private:
  Params params_;
};

// Returns the number of bytes allocated for the given tensor.
int64_t GetAllocatedBytes(const std::vector<Tensor>& element);

// Returns the estimated memory usage in bytes of the given tensor.
int64_t GetTotalBytes(const std::vector<Tensor>& element);

// Validates and extracts a `DatasetBase` object from `tensor`.
//
// `tensor` must have been written by a call to SetVariantTensorToDataset().
//
// The retrieved pointer is a borrowed reference to the dataset, which is owned
// by the tensor. The consumer must either acquire its own reference to the
// dataset by calling `(*out_dataset)->Ref()`, or ensure that `tensor` is not
// destroyed or mutated while the retrieved pointer is in use.
Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset);

// Stores a `DatasetBase` object in `tensor`.
//
// The ownership of `dataset` is transferred to `tensor`.
Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor);

// Represents a (potentially infinite) range of outputs, where each
// output is a tuple of tensors.
class DatasetBase : public core::RefCounted {
 public:
  // Key for storing the Dataset graph in the serialized format.
  TF_EXPORT static const char kDatasetGraphKey[];

  // Key for storing the output node of the Dataset graph in the serialized
  // format.
  TF_EXPORT static const char kDatasetGraphOutputNodeKey[];

  explicit DatasetBase(DatasetContext&& ctx)
      : type_string_(ctx.type_string()), node_name_(ctx.node_name()) {}

  // Op type name of this dataset.
  const string& type_string() const { return type_string_; }

  // Graph node name of this dataset op, uniquely identifying the dataset in
  // the graph.
  const string& node_name() const { return node_name_; }

  const Metadata& metadata() const { return metadata_; }

  const Options& options() const { return options_; }

  int64_t num_sources() const { return num_sources_; }

  // Initializes the dataset using the given metadata.
  void Initialize(const Metadata& metadata);

  // Returns a new iterator for iterating over the range of elements in
  // this dataset.
  //
  // This method may be called multiple times on the same instance,
  // and the resulting iterators will have distinct state. Each
  // iterator will traverse all elements in this dataset from the
  // start.
  //
  // The prefix identifies the sequence of iterators leading up to the newly
  // created iterator.
  Status MakeIterator(IteratorContext* ctx, const IteratorBase* parent,
                      const string& output_prefix,
                      std::unique_ptr<IteratorBase>* iterator) const;

  Status MakeIterator(IteratorContext&& ctx, const IteratorBase* parent,
                      const string& output_prefix,
                      std::unique_ptr<IteratorBase>* iterator) const {
    return MakeIterator(&ctx, parent, output_prefix, iterator);
  }

  // Returns a new iterator restored from the checkpoint data in `reader`.
  Status MakeIteratorFromCheckpoint(
      IteratorContext* ctx, const string& output_prefix,
      IteratorStateReader* reader,
      std::unique_ptr<IteratorBase>* iterator) const {
    std::unique_ptr<IteratorBase> it;
    IteratorContext::Params params(ctx);
    params.is_restoring = true;
    IteratorContext restore_ctx(std::move(params));
    TF_RETURN_IF_ERROR(MakeIterator(&restore_ctx,
                                    /*parent=*/nullptr, output_prefix, &it));
    TF_RETURN_IF_ERROR(it->Restore(&restore_ctx, reader));
    ctx->MergeCheckpoint(restore_ctx.checkpoint());
    *iterator = std::move(it);
    return OkStatus();
  }

  Status MakeIteratorFromCheckpoint(
      IteratorContext&& ctx, const string& output_prefix,
      IteratorStateReader* reader,
      std::unique_ptr<IteratorBase>* iterator) const {
    return MakeIteratorFromCheckpoint(&ctx, output_prefix, reader, iterator);
  }

  // Returns a split provider which partitions the dataset's data into splits
  // and provides them in a sequence. The split provider is stored in
  // `*split_provider`.
  virtual Status MakeSplitProviders(
      std::vector<std::unique_ptr<SplitProvider>>* split_providers) const;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // dataset.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this dataset.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // Returns the number of bytes allocated for tensors of this dataset.
  virtual int64_t AllocatedBytes() const { return 0; }

  // Returns the estimated number of bytes used for tensors of this dataset.
  virtual int64_t TotalBytes() const { return 0; }

  // Returns the cardinality of this dataset.
  // TODO(shilpakrish): Remove this overload once all callers are migrated
  // to the API which passes in the options parameter.
  ABSL_DEPRECATED("Use the overload that passes in the options parameter.")
  int64_t Cardinality() const;

  // Returns the cardinality of this dataset based on the options.
  int64_t Cardinality(CardinalityOptions options) const;

  // Internal implementation of cardinality for a dataset.
  // TODO(shilpakrish): Remove this overload once all callers are migrated
  // to the API which passes in the options parameter.
  ABSL_DEPRECATED("Use the overload that passes in the options parameter.")
  virtual int64_t CardinalityInternal() const
      TF_EXCLUSIVE_LOCKS_REQUIRED(cardinality_mu_) {
    return kUnknownCardinality;
  }

  // Internal implementation of cardinality for a dataset based on the options.
  virtual int64_t CardinalityInternal(CardinalityOptions options) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(cardinality_mu_) {
    return kUnknownCardinality;
  }

  // A human-readable debug string for this dataset.
  virtual string DebugString() const = 0;

  // Stores the dataset's input datasets in `*inputs`. The pointers stored in
  // `*inputs` are borrowed. The only valid non-ok return status is
  // UNIMPLEMENTED in case `InputDatasets` is not implemented by a dataset
  // subclass. Implementing `InputDatasets` enables `DatasetBase` to provide a
  // default implementation of `MakeSplitProvider` when there is a single input
  // dataset.
  virtual Status InputDatasets(std::vector<const DatasetBase*>* inputs) const;

  // Indicates whether the dataset depends on any external state which would
  // prevent it from being serializable. If so, the method returns
  // `errors::FailedPrecondition` with a message that identifies the external
  // state. Otherwise, the method returns `OkStatus()`.
  virtual Status CheckExternalState() const = 0;

  // Indicates whether the dataset is compatible with random access.
  Status CheckRandomAccessCompatible(const int64 index) const;

  // Return the element at a particular index for a randomly accessible dataset.
  virtual Status Get(OpKernelContext* ctx, int64 index,
                     std::vector<Tensor>* out_tensors) const;

  // Return a finalized version of the dataset.  The returned DatasetBase is
  // unowned and lives for as long as this dataset.
  virtual StatusOr<DatasetBase*> Finalize(
      OpKernelContext* ctx,
      std::function<StatusOr<core::RefCountPtr<DatasetBase>>()>
          make_finalized_dataset) const;

  // Wrapper around a GraphDefBuilder which provides support for serializing
  // Datasets as GraphDefs.
  class DatasetGraphDefBuilder : public GraphDefBuilderWrapper {
   public:
    explicit DatasetGraphDefBuilder(GraphDefBuilder* b)
        : GraphDefBuilderWrapper(b) {}
    Status AddInputDataset(SerializationContext* ctx,
                           const DatasetBase* dataset, Node** output);
    Status AddDatasetOrTensor(SerializationContext* ctx, const Tensor& val,
                              Node** output);
    Status AddIdentity(SerializationContext* ctx,
                       const std::string& name_prefix, Node** input,
                       Node** output);

   private:
    Status AddDatasetOrTensorHelper(SerializationContext* ctx,
                                    const Tensor& val, Node** output);
    Status AddResourceHelper(SerializationContext* ctx, const Tensor& val,
                             Node** output);
  };

 protected:
  friend class CapturedFunction;

  // Serializes the dataset into a `GraphDef`, which has two uses:
  //
  // 1) To perform static input pipeline optimizations, tf.data serializes the
  // dataset graph, applies graph rewrites, and then deserializes the graph.
  // If a subclass of `DatasetBase` does not implement this method, then it will
  // be excluded from static optimizations (and so will any upstream datasets).
  //
  // 2) To save the dataset so that it can restore at a later point (possibly in
  // different environment). If a subclass of `DatasetBase` does not implement
  // this method, then this migration will not be possible.
  virtual Status AsGraphDefInternal(SerializationContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    Node** node) const = 0;

  virtual std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const = 0;

  void set_options(const Options& options) { options_ = options; }

 private:
  // Computes and stores the cardinality of a given dataset.
  Status ComputeCardinality();

  // Computes the number of source datasets feeding into this dataset. A source
  // dataset is a leaf in the subtree of dataset inputs.
  Status ComputeNumSources();

  // Merges options from inputs to this dataset. If there is a conflict in a
  // field value, the options set on this dataset takes precedence over those in
  // the inputs. The order of precedence on the inputs is in the same order as
  // how they appear for this dataset.
  Status MergeOptionsFromInputs();

  const string type_string_;
  const string node_name_;
  Metadata metadata_;
  Options options_;
  mutable mutex mu_;
  mutable mutex cardinality_mu_;
  mutable core::RefCountPtr<DatasetBase> finalized_dataset_;
  //  The number of source datasets feeding into the dataset. A source dataset
  //  is a leaf in the subtree of dataset inputs.
  int64_t num_sources_ = -1;
  mutable int64_t cardinality_ TF_GUARDED_BY(cardinality_mu_) =
      kUnknownCardinality;
};

// Represents an iterator that is associated with a particular dataset.
class DatasetBaseIterator : public IteratorBase {
 public:
  struct BaseParams {
    // Owns one reference on the shared dataset object.
    const DatasetBase* dataset;

    // Identifies the sequence of iterators leading up to this iterator.
    const string prefix;
  };

  explicit DatasetBaseIterator(const BaseParams& params);

  ~DatasetBaseIterator() override;

  virtual const DatasetBase* dataset() const { return params_.dataset; }

  const DataTypeVector& output_dtypes() const override {
    return params_.dataset->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return params_.dataset->output_shapes();
  }

  const string& prefix() const override { return params_.prefix; }

  // Returns a name to be used for the TraceMe event.
  //
  // NOTE: TraceMe supports passing key-value pairs of "arguments" using the
  // following format "name#arg_1=value_,...,arg_n=value_n".
  string BuildTraceMeName();

  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) final;

  Status GetNext(IteratorContext&& ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
    return GetNext(&ctx, out_tensors, end_of_sequence);
  }

  Status Skip(IteratorContext* ctx, int num_to_skip, bool* end_of_sequence,
              int* num_skipped) final;

  Status Save(SerializationContext* ctx, IteratorStateWriter* writer) final {
    VLOG(2) << "Attempting to save checkpoints on iterator (prefix: "
            << prefix() << ") from " << dataset()->DebugString();
    return IteratorBase::Save(ctx, writer);
  }

 protected:
  Status Restore(IteratorContext* ctx, IteratorStateReader* reader) final {
    VLOG(2) << "Attempting to restore checkpoints on iterator (prefix: "
            << prefix() << ") from " << dataset()->DebugString();
    return IteratorBase::Restore(ctx, reader);
  }

  // Internal implementation of GetNext that is wrapped in tracing logic.
  //
  // See the docstring of `GetNext` method regaring the contract for
  // `out_tensors` and `end_of_sequence`. Implementations may assume that
  // `*out_tensors` is empty.
  virtual Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) = 0;

  // Internal implementation of Skip that is wrapped in tracing logic
  virtual Status SkipInternal(IteratorContext* ctx, int num_to_skip,
                              bool* end_of_sequence, int* num_skipped);

  string full_name(const string& name) const {
    return FullName(params_.prefix, name);
  }

  // Returns a map of key-value pairs to included in the TraceMe string.
  virtual TraceMeMetadata GetTraceMeMetadata() const { return {}; }

  // By default we model iterators using an unknown node, which acts as
  // pass-through with respect to performance modeling.
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeUnknownNode(std::move(args));
  }

  // When modeling is enabled, this method disables autotuning for the given
  // iterator (and the transitive closure of its inputs).
  void DisableAutotune(IteratorContext* ctx, IteratorBase* iterator) {
    if (iterator->node_) {
      iterator->node_->set_autotune(false);
    }
  }

  // When modeling is enabled, this method enables autotuning for the given
  // iterator (and the transitive closure of its inputs).
  void EnableAutotune(IteratorContext* ctx, IteratorBase* iterator) {
    if (iterator->node_) {
      iterator->node_->set_autotune(true);
    }
  }

  // When modeling is enabled, this method records the fact that this iterator
  // has dequeued an element from an internal buffer.
  void RecordBufferDequeue(IteratorContext* ctx,
                           const std::vector<Tensor>& element) {
    if (collect_resource_usage(ctx)) {
      node_->record_buffer_event(-GetAllocatedBytes(element), -1);
      DCHECK_GE(node_->buffered_elements(), 0);
    }
  }

  // When modeling is enabled, this method records the fact that this iterator
  // has enqueued an element in an internal buffer.
  void RecordBufferEnqueue(IteratorContext* ctx,
                           const std::vector<Tensor>& element) {
    if (collect_resource_usage(ctx)) {
      node_->record_buffer_event(GetAllocatedBytes(element), 1);
    }
  }

  // When modeling is enabled, this method records the fact that this iterator
  // has produced an element and its size in bytes.
  void RecordElement(IteratorContext* ctx, std::vector<Tensor>* out_tensors) {
    if (collect_resource_usage(ctx)) {
      int64_t num_bytes = GetAllocatedBytes(*out_tensors);
      node_->record_element();
      node_->record_bytes_produced(num_bytes);
      if (node_->output()) {
        node_->output()->record_bytes_consumed(num_bytes);
      }
    }
  }

  // When modeling is enabled, this method records the fact that a thread of
  // this iterator has started work.
  void RecordStart(IteratorContext* ctx) {
    if (collect_resource_usage(ctx)) {
      int64_t now_nanos = EnvTime::NowNanos();
      node_->record_start(now_nanos);
    }
  }

  // When modeling is enabled, this method records the fact that a thread of
  // this iterator has stopped work.
  void RecordStop(IteratorContext* ctx) {
    if (collect_resource_usage(ctx)) {
      int64_t now_nanos = EnvTime::NowNanos();
      node_->record_stop(now_nanos);
    }
  }

  // Returns whether work is currently being recorded, i.e. whether we are
  // currently between a `RecordStart` and a `RecordStop`.
  bool IsRecording(IteratorContext* ctx) {
    return node_ && node_->is_recording();
  }

 private:
  bool collect_resource_usage(IteratorContext* ctx) {
    return ctx->model() && node_;
  }

  string traceme_metadata_;
  BaseParams params_;
};

// Represents an iterator that is associated with a particular dataset
// with a particular type.
template <class DatasetType>
class DatasetIterator : public DatasetBaseIterator {
 public:
  struct Params {
    // Borrowed pointer to the dataset.
    const DatasetType* dataset;

    // Identifies the sequence of iterators leading up to this iterator.
    const string prefix;
  };

  explicit DatasetIterator(const Params& params)
      : DatasetBaseIterator({params.dataset, params.prefix}),
        typed_dataset_(params.dataset) {}

  // The dataset from which this iterator was created.
  const DatasetType* dataset() const final { return typed_dataset_; }

 private:
  const DatasetType* const typed_dataset_;  // Not owned.
};

template <typename T>
Status ParseScalarArgument(OpKernelContext* ctx,
                           const StringPiece& argument_name, T* output) {
  const Tensor* argument_t;
  TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
  if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
    return errors::InvalidArgument(argument_name, " must be a scalar");
  }
  *output = argument_t->scalar<T>()();
  return OkStatus();
}

template <typename T>
Status ParseVectorArgument(OpKernelContext* ctx,
                           const StringPiece& argument_name,
                           std::vector<T>* output) {
  const Tensor* argument_t;
  TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
  if (!TensorShapeUtils::IsVector(argument_t->shape())) {
    return errors::InvalidArgument(argument_name, " must be a vector");
  }
  int size = argument_t->vec<T>().size();
  output->reserve(size);
  for (int i = 0; i < size; ++i) {
    output->push_back(argument_t->vec<T>()(i));
  }
  return OkStatus();
}

// Encapsulates the work required to plug a DatasetBase into the core TensorFlow
// graph execution engine.
class DatasetOpKernel : public OpKernel {
 public:
  explicit DatasetOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    if (ctx->HasAttr(kMetadata)) {
      std::string serialized_metadata;
      OP_REQUIRES_OK(ctx, ctx->GetAttr(kMetadata, &serialized_metadata));
      OP_REQUIRES(ctx, metadata_.ParseFromString(serialized_metadata),
                  errors::InvalidArgument(absl::StrCat(
                      "Could not parse the 'metadata' attribute.")));
    }
  }

  void Compute(OpKernelContext* ctx) final;

  // Checks whether the given op is a tf.data operation.
  //
  // NOTE: The check uses a heuristic and can produce both false positives and
  // false negatives. In particular, tf.data operations are expected to use
  // names that end with "Dataset" or "DatasetV[0-9]+".
  static bool IsDatasetOp(const OpDef& op_def);

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;

 protected:
  // Subclasses should implement this method. It will be called during Compute
  // execution.
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase** output) = 0;

 private:
  Metadata metadata_;
};

// Encapsulates the work required to plug unary Datasets into the core
// TensorFlow graph execution engine.
class UnaryDatasetOpKernel : public DatasetOpKernel {
 public:
  explicit UnaryDatasetOpKernel(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase** output) = 0;
};

// Encapsulates the work required to plug binary Datasets into the core
// TensorFlow graph execution engine.
class BinaryDatasetOpKernel : public DatasetOpKernel {
 public:
  explicit BinaryDatasetOpKernel(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase* another_input,
                           DatasetBase** output) = 0;
};

// A simple background worker that executes closures asynchronously and without
// blocking.
//
// A `BackgroundWorker` is used to offload blocking work from an `AsyncOpKernel`
// to avoid blocking an executor thread that may be required by the blocking
// work.
//
// NOTE(mrry): We do not use a regular `tensorflow::thread::ThreadPool` for this
// purpose because its current implementation (in Eigen) uses a finite-length
// queue and will block the caller when full. This can lead to deadlock under
// heavy load. Since the number of concurrent work items in each user of a
// `BackgroundWorker` is at most one per op invocation, the dynamic allocation
// overhead is tolerable.
class BackgroundWorker {
 public:
  BackgroundWorker(Env* env, const char* name);

  ~BackgroundWorker();

  void Schedule(std::function<void()> work_item);

 private:
  void WorkerLoop();

  Env* const env_;
  const char* const name_;

  std::unique_ptr<Thread> thread_;
  mutex mu_;
  condition_variable cond_var_;
  bool cancelled_ TF_GUARDED_BY(mu_) = false;
  std::deque<std::function<void()>> work_queue_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DATASET_H_
