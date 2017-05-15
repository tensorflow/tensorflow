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

#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {

struct KeyTensorLess {
  bool operator()(const Tensor & lhs, const Tensor & rhs) const {
    return std::less<int64>{}(lhs.scalar<int64>()(),
                              rhs.scalar<int64>()());
  }
};

struct KeyTensorEqual {
  bool operator()(const Tensor & lhs, const Tensor & rhs) const {
    return std::equal_to<int64>{}(lhs.scalar<int64>()(),
                                  rhs.scalar<int64>()());
  }
};

struct KeyTensorHash {
  std::size_t operator()(const Tensor & key) const {
    return std::hash<int64>{}(key.scalar<int64>()());
  }
};


// General Template Definition
template <bool Ordered, typename Data>
struct MapTraits {};

// Partially specialise for ordered
template <typename Data>
struct MapTraits<true, Data>
{
  typedef Tensor key_type;
  typedef Data data_type;
  typedef std::map<key_type, Data, KeyTensorLess> map_type;
};

// Partially specialise for unordered
template <typename Data>
struct MapTraits<false, Data>
{
  typedef Tensor key_type;
  typedef Data data_type;
  typedef std::unordered_map<key_type, Data,
                            KeyTensorHash, KeyTensorEqual> map_type;
};

// Wrapper around map/unordered_map
template <bool Ordered>
class StagingMap : public ResourceBase
{
public:
  // Public typedefs
  typedef std::vector<Tensor> Tuple;
  typedef gtl::optional<Tensor> OptionalTensor;
  typedef std::vector<OptionalTensor> IncompleteTuple;

  typedef MapTraits<Ordered, Tuple> map_traits;
  typedef typename map_traits::map_type map_type;
  typedef typename map_traits::key_type key_type;

  typedef MapTraits<false, IncompleteTuple> incomplete_traits;
  typedef typename incomplete_traits::map_type incomplete_type;

private:
  // Private variables
  DataTypeVector dtypes_;
  int capacity_;
  int memory_limit_;
  int current_bytes_;
  mutex mu_;
  condition_variable not_empty_;
  condition_variable full_;
  incomplete_type incomplete_ GUARDED_BY(mu_);
  map_type map_ GUARDED_BY(mu_);

private:
  // private methods

  // If map is configured for bounded capacity, notify
  // waiting inserters that space is now available
  void notify_inserters_if_bounded(mutex_lock & l)
  {
    {
      l.unlock();
      full_.notify_one();
    }
  }

  // Notify any removers waiting to extract values
  // that data is now available
  void notify_removers(mutex_lock & l)
  {
      l.unlock();
      not_empty_.notify_one();
  }

  bool is_bounded()
    { return capacity_ > 0 || memory_limit_ > 0; }

  bool would_exceed_memory_limit(int bytes)
    { return bytes + current_bytes_ > memory_limit_; }

  bool is_capacity_full()
    { return map_.size() >= capacity_; }

  // Get number of bytes in the tuple
  int get_tuple_bytes(const Tuple & tuple)
  {
    return std::accumulate(tuple.begin(), tuple.end(), int(0),
      [](const int & lhs, const Tensor & rhs) {
        return lhs + rhs.TotalBytes();
    });
  }


public:
  // public methods
  explicit StagingMap(const DataTypeVector & dtypes,
          int capacity, int memory_limit) :
      dtypes_(dtypes),
      capacity_(capacity),
      memory_limit_(memory_limit),
      current_bytes_(0) {}

  Status put(key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    int tuple_bytes = get_tuple_bytes(*tuple);

    // Sanity check so that we don't block for ever below
    if(memory_limit_ > 0 && tuple_bytes > memory_limit_) {
      return Status(errors::ResourceExhausted("Attempted to insert "
        "tensors with combined size of '", tuple_bytes, "' bytes into "
        "Staging Area with a memory limit of '", memory_limit_, "'."));
    }

    // If map capacity is bounded wait until map is not full
    if(is_bounded()) {
      full_.wait(l, [tuple_bytes, this]() {
        // If there's a memory limit, check if there's space for insertion
        bool memory_limit_valid = memory_limit_ > 0 ?
              !would_exceed_memory_limit(tuple_bytes) : true;
        // If we're configured for capacity check if there's space for insertion
        bool capacity_valid = capacity_ > 0 ? !is_capacity_full() : true;

        // Stop waiting upon success for both conditions
        return memory_limit_valid && capacity_valid;
      });
    }

    // Update bytes in the Staging Area
    current_bytes_ += tuple_bytes;

    // Insert key and tuples into the map
    map_.insert({*key, std::move(*tuple)});

    notify_removers(l);

    return Status::OK();
  }

  Status get(const key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    typename map_type::const_iterator it;

    // Wait until the element with the requested key is present
    not_empty_.wait(l, [&, this]() {
      it = map_.find(*key);
      return it != map_.end();
    });

    // Copy tensors into the tuple
    for(const auto & tensor : it->second)
      { tuple->push_back(tensor); }

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    return Status::OK();
  }

  Status pop(const key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    typename map_type::iterator it;

    // Wait until the element with the requested key is present
    not_empty_.wait(l, [&, this]() {
      it = map_.find(*key);
      return it != this->map_.end();
    });

    // Move from the entry as its erased anyway
    *tuple = std::move(it->second);

    // Remove
    map_.erase(it);

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    notify_inserters_if_bounded(l);

    return Status::OK();
  }

  Status popitem(key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    // Wait until map is not empty
    not_empty_.wait(l, [this]() { return !this->map_.empty(); });

    // Move from the first element and erase it
    *tuple = std::move(map_.begin()->second);
    *key = map_.begin()->first;
    map_.erase(map_.begin());

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    notify_inserters_if_bounded(l);

    return Status::OK();
  }

  Status clear()
  {
    mutex_lock l(mu_);
    map_.clear();
    current_bytes_ = 0;

    notify_inserters_if_bounded(l);

    return Status::OK();
  }

  size_t size()
  {
    // Lock the map and return the size
    mutex_lock l(mu_);
    return map_.size();
  }

  string DebugString()
  {
    return "StagingMap";
  }
};

template <bool Ordered>
Status GetStagingMap(OpKernelContext* ctx,
                    const NodeDef& ndef,
                    StagingMap<Ordered>** map)
{
  auto rm = ctx->resource_manager();
  ContainerInfo cinfo;

  // Lambda for creating the Staging Area
  auto create_fn = [&ndef](StagingMap<Ordered>** ret) -> Status
  {
    DataTypeVector dtypes;
    int capacity;
    int memory_limit;
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "dtypes", &dtypes));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "memory_limit", &memory_limit));
    *ret = new StagingMap<Ordered>(dtypes, capacity, memory_limit);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(cinfo.Init(rm, ndef, true /* use name() */));
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<StagingMap<Ordered>>(
                        cinfo.container(), cinfo.name(),
                        map, create_fn));
  return Status::OK();
}

template <bool Ordered>
class MapStageOp : public OpKernel
{
 public:
  explicit MapStageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor * key_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values_tensor));

    // Create copy for insertion into Staging Area
    Tensor key(*key_tensor);

    // Create the tuple to store
    for (int i = 0; i < values_tensor.size(); ++i) {
      tuple.push_back(values_tensor[i]);
    }

    // Store the tuple in the map
    OP_REQUIRES_OK(ctx, map->put(&key, &tuple));
  }
};

REGISTER_KERNEL_BUILDER(Name("MapStage").Device(DEVICE_CPU),
                      MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage").Device(DEVICE_CPU),
                      MapStageOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapStage").HostMemory("key")
                      .Device(DEVICE_GPU), MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage").HostMemory("key")
                      .Device(DEVICE_GPU), MapStageOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapStage").HostMemory("key")
                      .Device(DEVICE_SYCL), MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage").HostMemory("key")
                      .Device(DEVICE_SYCL), MapStageOp<true>);

#endif // TENSORFLOW_USE_SYCL

template <bool Ordered>
class MapUnstageOp : public OpKernel
{
 public:
  explicit MapUnstageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor * key_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, map->pop(key_tensor, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));
    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstage").Device(DEVICE_CPU),
                            MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").Device(DEVICE_CPU),
                            MapUnstageOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapUnstage").HostMemory("key")
                            .Device(DEVICE_GPU), MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").HostMemory("key")
                            .Device(DEVICE_GPU), MapUnstageOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapUnstage").HostMemory("key")
                            .Device(DEVICE_SYCL), MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").HostMemory("key")
                            .Device(DEVICE_SYCL), MapUnstageOp<true>);
#endif // TENSORFLOW_USE_SYCL

template <bool Ordered>
class MapPeekOp : public OpKernel
{
 public:
  explicit MapPeekOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor * key_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, map->get(key_tensor, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));
    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapPeek").Device(DEVICE_CPU),
                      MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").Device(DEVICE_CPU),
                      MapPeekOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapPeek").HostMemory("key")
                      .Device(DEVICE_GPU), MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").HostMemory("key")
                      .Device(DEVICE_GPU), MapPeekOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapPeek").HostMemory("key")
                      .Device(DEVICE_SYCL), MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").HostMemory("key")
                      .Device(DEVICE_SYCL), MapPeekOp<true>);
#endif // TENSORFLOW_USE_SYCL



template <bool Ordered>
class MapUnstageNoKeyOp : public OpKernel
{
 public:
  explicit MapUnstageNoKeyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Pop a random (key, value) off the map
    typename StagingMap<Ordered>::key_type key;
    typename StagingMap<Ordered>::Tuple tuple;

    OP_REQUIRES_OK(ctx, map->popitem(&key, &tuple));

    // Allocate a key tensor and assign the key as the first output
    ctx->set_output(0, key);

    // Set the rest of the outputs to the tuple Tensors
    OP_REQUIRES(ctx,
      tuple.size() == (size_t)ctx->num_outputs()-1,
      errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                              " vs. ", ctx->num_outputs()-1));
    for (size_t i = 0; i < tuple.size(); ++i)
    {
      ctx->set_output(i+1, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").Device(DEVICE_CPU),
                      MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").Device(DEVICE_CPU),
                      MapUnstageNoKeyOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_GPU), MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_GPU), MapUnstageNoKeyOp<true>);

#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_SYCL), MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_SYCL), MapUnstageNoKeyOp<true>);
#endif // TENSORFLOW_USE_SYCL


template <bool Ordered>
class MapSizeOp : public OpKernel
{
 public:
  explicit MapSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override
  {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Allocate size output tensor
    Tensor * size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}),
                                                     &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(map->size());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_CPU),
                        MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_CPU),
                        MapSizeOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_GPU)
                        .HostMemory("size"), MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_GPU)
                        .HostMemory("size"), MapSizeOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_SYCL)
                        .HostMemory("size"), MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_SYCL)
                        .HostMemory("size"), MapSizeOp<true>);
#endif // TENSORFLOW_USE_SYCL

template <bool Ordered>
class MapClearOp : public OpKernel
{
 public:
  explicit MapClearOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override
  {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    OP_REQUIRES_OK(ctx, map->clear());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_CPU),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_CPU),
                        MapClearOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_GPU),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_GPU),
                        MapClearOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_SYCL),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_SYCL),
                        MapClearOp<true>);
#endif // TENSORFLOW_USE_SYCL

}  // namespace

}  // namespace tensorflow
