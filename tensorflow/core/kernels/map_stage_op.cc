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

#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace {

// Partial Ordering Comparator for Tensor keys containing scalar int64's
struct KeyTensorLess {
  bool operator()(const Tensor& lhs, const Tensor& rhs) const {
    return std::less<int64_t>{}(lhs.scalar<int64_t>()(),
                                rhs.scalar<int64_t>()());
  }
};

// Key Equality operator for Tensor keys containing scalar int64's
struct KeyTensorEqual {
  bool operator()(const Tensor& lhs, const Tensor& rhs) const {
    return std::equal_to<int64_t>{}(lhs.scalar<int64_t>()(),
                                    rhs.scalar<int64_t>()());
  }
};

// Hash for Tensor keys containing scalar int64's
struct KeyTensorHash {
  std::size_t operator()(const Tensor& key) const {
    return std::hash<int64_t>{}(key.scalar<int64_t>()());
  }
};

// Primary template.
template <bool Ordered, typename Data>
struct MapTraits;

// Partial specialization for ordered.
template <typename Data>
struct MapTraits<true, Data> {
  using KeyType = Tensor;
  using DataType = Data;
  using MapType = std::map<KeyType, Data, KeyTensorLess>;
};

// Partial specialization for unordered.
template <typename Data>
struct MapTraits<false, Data> {
  using KeyType = Tensor;
  using DataType = Data;
  using MapType =
      std::unordered_map<KeyType, Data, KeyTensorHash, KeyTensorEqual>;
};

// Wrapper around map/unordered_map.
template <bool Ordered>
class StagingMap : public ResourceBase {
 public:
  // Public typedefs
  using Tuple = std::vector<Tensor>;
  using OptionalTensor = std::optional<Tensor>;
  using OptionalTuple = std::vector<OptionalTensor>;

  using MapType = typename MapTraits<Ordered, OptionalTuple>::MapType;
  using KeyType = typename MapTraits<Ordered, OptionalTuple>::KeyType;

  using IncompleteType = typename MapTraits<false, OptionalTuple>::MapType;

 private:
  // Private variables
  DataTypeVector dtypes_ TF_GUARDED_BY(mu_);
  std::size_t capacity_ TF_GUARDED_BY(mu_);
  std::size_t memory_limit_ TF_GUARDED_BY(mu_);
  std::size_t current_bytes_ TF_GUARDED_BY(mu_);
  tensorflow::mutex mu_;
  tensorflow::condition_variable not_empty_;
  tensorflow::condition_variable full_;
  IncompleteType incomplete_ TF_GUARDED_BY(mu_);
  MapType map_ TF_GUARDED_BY(mu_);

 private:
  // private methods

  // If map is configured for bounded capacity, notify
  // waiting inserters that space is now available
  void notify_inserters_if_bounded() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (has_capacity() || has_memory_limit()) {
      // Notify all inserters. The removal of an element
      // may make memory available for many inserters
      // to insert new elements
      full_.notify_all();
    }
  }

  // Notify all removers waiting to extract values
  // that data is now available
  void notify_removers() {
    // Notify all removers. This is because they are
    // waiting for specific keys to appear in the map
    // so we don't know which one to wake up.
    not_empty_.notify_all();
  }

  bool has_capacity() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return capacity_ > 0;
  }

  bool has_memory_limit() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return memory_limit_ > 0;
  }

  bool would_exceed_memory_limit(std::size_t bytes) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return has_memory_limit() && bytes + current_bytes_ > memory_limit_;
  }

  bool is_capacity_full() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return has_capacity() && map_.size() >= capacity_;
  }

  // Get number of bytes in the tuple
  std::size_t get_tuple_bytes(const Tuple& tuple) {
    return std::accumulate(tuple.begin(), tuple.end(),
                           static_cast<std::size_t>(0),
                           [](const std::size_t& lhs, const Tensor& rhs) {
                             return lhs + rhs.TotalBytes();
                           });
  }

  // Get number of bytes in the incomplete tuple
  std::size_t get_tuple_bytes(const OptionalTuple& tuple) {
    return std::accumulate(
        tuple.begin(), tuple.end(), static_cast<std::size_t>(0),
        [](const std::size_t& lhs, const OptionalTensor& rhs) {
          return (lhs + rhs.has_value()) ? rhs.value().TotalBytes() : 0;
        });
  }

  // Check that the index is within bounds
  absl::Status check_index(const Tensor& key, std::size_t index)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (index >= dtypes_.size()) {
      return absl::Status(errors::InvalidArgument(
          "Index '", index, "' for key '", key.scalar<int64_t>()(),
          "' was out of bounds '", dtypes_.size(), "'."));
    }

    return absl::OkStatus();
  }

  absl::Status copy_or_move_tensors(OptionalTuple* map_tuple, const Tensor& key,
                                    const Tensor& indices, Tuple* output,
                                    bool copy = false)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto findices = indices.flat<int>();

    // Return values at specified indices
    for (std::size_t i = 0; i < findices.dimension(0); ++i) {
      std::size_t index = findices(i);

      TF_RETURN_IF_ERROR(check_index(key, index));

      // Insist on a value present at the specified index
      if (!(*map_tuple)[index].has_value()) {
        return absl::Status(errors::InvalidArgument(
            "Tensor at index '", index, "' for key '", key.scalar<int64_t>()(),
            "' has already been removed."));
      }

      // Copy the contained tensor and
      // remove from the OptionalTuple
      output->push_back((*map_tuple)[index].value());

      // Clear out the entry if we're not copying (moving)
      if (!copy) {
        (*map_tuple)[index].reset();
      }
    }

    return absl::OkStatus();
  }

  // Check that the optional value at the specified index
  // is uninitialized
  absl::Status check_index_uninitialized(const Tensor& key, std::size_t index,
                                         const OptionalTuple& tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (tuple[index].has_value()) {
      return errors::InvalidArgument("The tensor for index '", index,
                                     "' for key '", key.scalar<int64_t>()(),
                                     "' was already initialized '",
                                     dtypes_.size(), "'.");
    }

    return absl::OkStatus();
  }

  // Check that the indices are strictly ordered
  absl::Status check_index_ordering(const Tensor& indices) {
    if (indices.NumElements() == 0) {
      return errors::InvalidArgument("Indices are empty");
    }

    auto findices = indices.flat<int>();

    for (std::size_t i = 0; i < findices.dimension(0) - 1; ++i) {
      if (findices(i) < findices(i + 1)) {
        continue;
      }

      return errors::InvalidArgument("Indices are not strictly ordered");
    }

    return absl::OkStatus();
  }

  // Check bytes are within memory limits memory limits
  absl::Status check_memory_limit(std::size_t bytes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (has_memory_limit() && bytes > memory_limit_) {
      return errors::ResourceExhausted(
          "Attempted to insert tensors with combined size of '", bytes,
          "' bytes into Staging Area with a memory limit of '", memory_limit_,
          "'.");
    }

    return absl::OkStatus();
  }

  // Insert incomplete data into the Barrier
  absl::Status put_incomplete(const KeyType& key, const Tensor& indices,
                              OptionalTuple* tuple,
                              tensorflow::mutex_lock* lock)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto findices = indices.flat<int>();

    // Search for the key in our incomplete set
    auto it = incomplete_.find(key);

    // Check that the tuple fits within the memory limit
    std::size_t tuple_bytes = get_tuple_bytes(*tuple);
    TF_RETURN_IF_ERROR(check_memory_limit(tuple_bytes));

    // Wait until we don't exceed the memory limit
    while (would_exceed_memory_limit(tuple_bytes)) {
      full_.wait(*lock);
    }

    // This key isn't present in the incomplete set
    // Create OptionalTuple and insert
    if (it == incomplete_.end()) {
      OptionalTuple empty(dtypes_.size());

      // Initialize empty tuple with given dta
      for (std::size_t i = 0; i < findices.dimension(0); ++i) {
        std::size_t index = findices(i);
        TF_RETURN_IF_ERROR(check_index(key, index));

        // Assign tuple at this index
        empty[index] = std::move((*tuple)[i]);
      }

      // Insert into incomplete map
      incomplete_.insert({key, std::move(empty)});

      // Increment size
      current_bytes_ += tuple_bytes;
    }
    // Found an entry in the incomplete index
    // Update with given data and insert complete entries
    // into the main map
    else {
      // Reference existing incomplete tuple
      OptionalTuple& present = it->second;

      // Assign given data
      for (std::size_t i = 0; i < findices.dimension(0); ++i) {
        std::size_t index = findices(i);
        TF_RETURN_IF_ERROR(check_index(key, index));
        TF_RETURN_IF_ERROR(check_index_uninitialized(key, index, present));

        // Assign tuple at this index
        present[index] = std::move((*tuple)[i]);
      }

      // Increment size
      current_bytes_ += tuple_bytes;

      // Do we have values at all tuple elements?
      bool complete =
          std::all_of(present.begin(), present.end(),
                      [](const OptionalTensor& v) { return v.has_value(); });

      // If so, put the tuple in the actual map
      if (complete) {
        OptionalTuple insert_tuple = std::move(it->second);

        // Remove from incomplete
        incomplete_.erase(it);

        TF_RETURN_IF_ERROR(put_complete(key, &insert_tuple));
      }
    }

    return absl::OkStatus();
  }

  // Does the insertion into the actual staging area
  absl::Status put_complete(const KeyType& key, OptionalTuple* tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // Insert key and tuples into the map
    map_.insert({key, std::move(*tuple)});

    notify_removers();

    return absl::OkStatus();
  }

 public:
  // public methods
  explicit StagingMap(const DataTypeVector& dtypes, std::size_t capacity,
                      std::size_t memory_limit)
      : dtypes_(dtypes),
        capacity_(capacity),
        memory_limit_(memory_limit),
        current_bytes_(0) {}

  absl::Status put(KeyType* key, const Tensor* indices, OptionalTuple* tuple) {
    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    // Handle incomplete inserts
    if (indices->NumElements() != dtypes_.size()) {
      return put_incomplete(*key, *indices, tuple, &lock);
    }

    std::size_t tuple_bytes = get_tuple_bytes(*tuple);
    // Check that tuple_bytes fits within the memory limit
    TF_RETURN_IF_ERROR(check_memory_limit(tuple_bytes));

    // Wait until there's space for insertion.
    while (would_exceed_memory_limit(tuple_bytes) || is_capacity_full()) {
      full_.wait(lock);
    }

    // Do the put operation
    TF_RETURN_IF_ERROR(put_complete(*key, tuple));

    // Update the current size
    current_bytes_ += tuple_bytes;

    return absl::OkStatus();
  }

  absl::Status get(const KeyType* key, const Tensor* indices, Tuple* tuple) {
    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    typename MapType::iterator it;

    // Wait until the element with the requested key is present
    while ((it = map_.find(*key)) == map_.end()) {
      not_empty_.wait(lock);
    }

    TF_RETURN_IF_ERROR(
        copy_or_move_tensors(&it->second, *key, *indices, tuple, true));

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    return absl::OkStatus();
  }

  absl::Status pop(const KeyType* key, const Tensor* indices, Tuple* tuple) {
    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    typename MapType::iterator it;

    // Wait until the element with the requested key is present
    while ((it = map_.find(*key)) == map_.end()) {
      not_empty_.wait(lock);
    }

    TF_RETURN_IF_ERROR(
        copy_or_move_tensors(&it->second, *key, *indices, tuple));

    // Remove entry if all the values have been consumed
    if (!std::any_of(
            it->second.begin(), it->second.end(),
            [](const OptionalTensor& tensor) { return tensor.has_value(); })) {
      map_.erase(it);
    }

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    notify_inserters_if_bounded();

    return absl::OkStatus();
  }

  absl::Status popitem(KeyType* key, const Tensor* indices, Tuple* tuple) {
    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    // Wait until map is not empty
    while (this->map_.empty()) {
      not_empty_.wait(lock);
    }

    // Move from the first element and erase it

    auto it = map_.begin();

    TF_RETURN_IF_ERROR(
        copy_or_move_tensors(&it->second, *key, *indices, tuple));

    *key = it->first;

    // Remove entry if all the values have been consumed
    if (!std::any_of(
            it->second.begin(), it->second.end(),
            [](const OptionalTensor& tensor) { return tensor.has_value(); })) {
      map_.erase(it);
    }

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    notify_inserters_if_bounded();

    return absl::OkStatus();
  }

  absl::Status clear() {
    tensorflow::mutex_lock lock(mu_);
    map_.clear();
    incomplete_.clear();
    current_bytes_ = 0;

    notify_inserters_if_bounded();

    return absl::OkStatus();
  }

  std::size_t incomplete_size() {
    tensorflow::mutex_lock lock(mu_);
    return incomplete_.size();
  }

  std::size_t size() {
    tensorflow::mutex_lock lock(mu_);
    return map_.size();
  }

  string DebugString() const override { return "StagingMap"; }
};

template <bool Ordered>
absl::Status GetStagingMap(OpKernelContext* ctx, const NodeDef& ndef,
                           StagingMap<Ordered>** map) {
  auto rm = ctx->resource_manager();
  ContainerInfo cinfo;

  // Lambda for creating the Staging Area
  auto create_fn = [&ndef](StagingMap<Ordered>** ret) -> absl::Status {
    DataTypeVector dtypes;
    int64_t capacity;
    int64_t memory_limit;
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "dtypes", &dtypes));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "memory_limit", &memory_limit));
    *ret = new StagingMap<Ordered>(dtypes, capacity, memory_limit);
    return absl::OkStatus();
  };

  TF_RETURN_IF_ERROR(cinfo.Init(rm, ndef, true /* use name() */));
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<StagingMap<Ordered>>(
      cinfo.container(), cinfo.name(), map, create_fn));
  return absl::OkStatus();
}

template <bool Ordered>
class MapStageOp : public OpKernel {
 public:
  explicit MapStageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::OptionalTuple tuple;

    const Tensor* key_tensor;
    const Tensor* indices_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values_tensor));
    OP_REQUIRES(ctx, key_tensor->NumElements() > 0,
                errors::InvalidArgument("key must not be empty"));

    OP_REQUIRES(ctx, key_tensor->NumElements() == 1,
                errors::InvalidArgument(
                    "key must be an int64 scalar, got tensor with shape: ",
                    key_tensor->shape()));

    // Create copy for insertion into Staging Area
    Tensor key(*key_tensor);

    // Create the tuple to store
    for (std::size_t i = 0; i < values_tensor.size(); ++i) {
      tuple.push_back(values_tensor[i]);
    }

    // Store the tuple in the map
    OP_REQUIRES_OK(ctx, map->put(&key, indices_tensor, &tuple));
  }
};

REGISTER_KERNEL_BUILDER(Name("MapStage").Device(DEVICE_CPU), MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage").Device(DEVICE_CPU),
                        MapStageOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapStage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapStageOp<true>);

template <bool Ordered>
class MapUnstageOp : public OpKernel {
 public:
  explicit MapUnstageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor* key_tensor;
    const Tensor* indices_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(key_tensor->shape()),
                errors::InvalidArgument("key must be an int64 scalar: ",
                                        key_tensor->shape().DebugString()));

    OP_REQUIRES_OK(ctx, map->pop(key_tensor, indices_tensor, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == indices_tensor->NumElements(),
        errors::InvalidArgument("output/indices size mismatch: ", tuple.size(),
                                " vs. ", indices_tensor->NumElements()));

    for (std::size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstage").Device(DEVICE_CPU),
                        MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").Device(DEVICE_CPU),
                        MapUnstageOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapUnstage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageOp<true>);

template <bool Ordered>
class MapPeekOp : public OpKernel {
 public:
  explicit MapPeekOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor* key_tensor;
    const Tensor* indices_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(key_tensor->shape()),
                errors::InvalidArgument("key must be an int64 scalar: ",
                                        key_tensor->shape().DebugString()));
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES_OK(ctx, map->get(key_tensor, indices_tensor, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == indices_tensor->NumElements(),
        errors::InvalidArgument("output/indices size mismatch: ", tuple.size(),
                                " vs. ", indices_tensor->NumElements()));

    for (std::size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapPeek").Device(DEVICE_CPU), MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").Device(DEVICE_CPU),
                        MapPeekOp<true>);

REGISTER_KERNEL_BUILDER(
    Name("MapPeek").HostMemory("key").HostMemory("indices").Device(
        DEVICE_DEFAULT),
    MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapPeekOp<true>);

template <bool Ordered>
class MapUnstageNoKeyOp : public OpKernel {
 public:
  explicit MapUnstageNoKeyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Pop a random (key, value) off the map
    typename StagingMap<Ordered>::KeyType key;
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor* indices_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES_OK(ctx, map->popitem(&key, indices_tensor, &tuple));

    // Allocate a key tensor and assign the key as the first output
    ctx->set_output(0, key);

    // Set the rest of the outputs to the tuple Tensors
    OP_REQUIRES(
        ctx, tuple.size() == indices_tensor->NumElements(),
        errors::InvalidArgument("output/indices size mismatch: ", tuple.size(),
                                " vs. ", indices_tensor->NumElements()));

    for (std::size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i + 1, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").Device(DEVICE_CPU),
                        MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").Device(DEVICE_CPU),
                        MapUnstageNoKeyOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageNoKeyOp<true>);

template <bool Ordered>
class MapSizeOp : public OpKernel {
 public:
  explicit MapSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Allocate size output tensor
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(map->size());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_CPU), MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_CPU),
                        MapSizeOp<true>);

REGISTER_KERNEL_BUILDER(
    Name("MapSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(
    Name("OrderedMapSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapSizeOp<true>);

template <bool Ordered>
class MapIncompleteSizeOp : public OpKernel {
 public:
  explicit MapIncompleteSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Allocate size output tensor
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(map->incomplete_size());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapIncompleteSize").Device(DEVICE_CPU),
                        MapIncompleteSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapIncompleteSize").Device(DEVICE_CPU),
                        MapIncompleteSizeOp<true>);

REGISTER_KERNEL_BUILDER(
    Name("MapIncompleteSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapIncompleteSizeOp<false>);
REGISTER_KERNEL_BUILDER(
    Name("OrderedMapIncompleteSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapIncompleteSizeOp<true>);

template <bool Ordered>
class MapClearOp : public OpKernel {
 public:
  explicit MapClearOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    OP_REQUIRES_OK(ctx, map->clear());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_CPU), MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_CPU),
                        MapClearOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_DEFAULT),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_DEFAULT),
                        MapClearOp<true>);

}  // namespace
}  // namespace tensorflow
