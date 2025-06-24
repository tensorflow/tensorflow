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

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/thread_annotations.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

// Implement a simple hash table as a Resource using ref-counting.
// This demonstrates a Stateful Op for a general Create/Read/Update/Delete
// (CRUD) style use case.  To instead make an op for a specific lookup table
// case, it is preferable to follow the implementation style of
// the kernels for TensorFlow's tf.lookup internal ops which use
// LookupInterface.
template <class K, class V>
class SimpleHashTableResource : public ::tensorflow::ResourceBase {
 public:
  Status Insert(const Tensor& key, const Tensor& value) {
    const K key_val = key.flat<K>()(0);
    const V value_val = value.flat<V>()(0);

    mutex_lock l(mu_);
    table_[key_val] = value_val;
    return OkStatus();
  }

  Status Find(const Tensor& key, Tensor* value, const Tensor& default_value) {
    // Note that tf_shared_lock could be used instead of mutex_lock
    // in ops that do not not modify data protected by a mutex, but
    // go/totw/197 recommends using exclusive lock instead of a shared
    // lock when the lock is not going to be held for a significant amount
    // of time.
    mutex_lock l(mu_);

    const V default_val = default_value.flat<V>()(0);
    const K key_val = key.flat<K>()(0);
    auto value_val = value->flat<V>();
    value_val(0) = gtl::FindWithDefault(table_, key_val, default_val);
    return OkStatus();
  }

  Status Remove(const Tensor& key) {
    mutex_lock l(mu_);

    const K key_val = key.flat<K>()(0);
    if (table_.erase(key_val) != 1) {
      return errors::NotFound("Key for remove not found: ", key_val);
    }
    return OkStatus();
  }

  // Save all key, value pairs to tensor outputs to support SavedModel
  Status Export(OpKernelContext* ctx) {
    mutex_lock l(mu_);
    int64_t size = table_.size();
    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("values", TensorShape({size}), &values));
    auto keys_data = keys->flat<K>();
    auto values_data = values->flat<V>();
    int64_t i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it->first;
      values_data(i) = it->second;
    }
    return OkStatus();
  }

  // Load all key, value pairs from tensor inputs to support SavedModel
  Status Import(const Tensor& keys, const Tensor& values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();

    mutex_lock l(mu_);
    table_.clear();
    for (int64_t i = 0; i < key_values.size(); ++i) {
      gtl::InsertOrUpdate(&table_, key_values(i), value_values(i));
    }
    return OkStatus();
  }

  // Create a debug string with the content of the map if this is small,
  // or some example data if this is large, handling both the cases where the
  // hash table has many entries and where the entries are long strings.
  std::string DebugString() const override { return DebugString(3); }
  std::string DebugString(int num_pairs) const {
    std::string rval = "SimpleHashTable {";
    size_t count = 0;
    const size_t max_kv_str_len = 100;
    mutex_lock l(mu_);
    for (const auto& pair : table_) {
      if (count >= num_pairs) {
        strings::StrAppend(&rval, "...");
        break;
      }
      std::string kv_str = strings::StrCat(pair.first, ": ", pair.second);
      strings::StrAppend(&rval, kv_str.substr(0, max_kv_str_len));
      if (kv_str.length() > max_kv_str_len) strings::StrAppend(&rval, " ...");
      strings::StrAppend(&rval, ", ");
      count += 1;
    }
    strings::StrAppend(&rval, "}");
    return rval;
  }

 private:
  mutable mutex mu_;
  absl::flat_hash_map<K, V> table_ TF_GUARDED_BY(mu_);
};

template <class K, class V>
class SimpleHashTableCreateOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableCreateOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor handle_tensor;
    AllocatorAttributes attr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                           &handle_tensor, attr));
    handle_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle(
            new SimpleHashTableResource<K, V>(), ctx->device()->name(),
            /*dtypes_and_shapes=*/{}, ctx->stack_trace());
    ctx->set_output(0, handle_tensor);
  }

 private:
  // Just to be safe, avoid accidentally copying the kernel.
  SimpleHashTableCreateOpKernel(const SimpleHashTableCreateOpKernel&) = delete;
  void operator=(const SimpleHashTableCreateOpKernel&) = delete;
};

// GetResource retrieves a Resource using a handle from the first
// input in "ctx" and saves it in "resource" without increasing
// the reference count for that resource.
template <class K, class V>
Status GetResource(OpKernelContext* ctx,
                   SimpleHashTableResource<K, V>** resource) {
  const Tensor& handle_tensor = ctx->input(0);
  const ResourceHandle& handle = handle_tensor.scalar<ResourceHandle>()();
  typedef SimpleHashTableResource<K, V> resource_type;
  TF_ASSIGN_OR_RETURN(*resource, handle.GetResource<resource_type>());
  return OkStatus();
}

template <class K, class V>
class SimpleHashTableFindOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableFindOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    DataTypeVector expected_outputs = {DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // Note that ctx->input(0) is the Resource handle
    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);
    TensorShape output_shape = default_value.shape();
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("value", output_shape, &out));
    OP_REQUIRES_OK(ctx, resource->Find(key, out, default_value));
  }
};

template <class K, class V>
class SimpleHashTableInsertOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableInsertOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // Note that ctx->input(0) is the Resource handle
    const Tensor& key = ctx->input(1);
    const Tensor& value = ctx->input(2);
    OP_REQUIRES_OK(ctx, resource->Insert(key, value));
  }
};

template <class K, class V>
class SimpleHashTableRemoveOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableRemoveOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    // Note that ctx->input(0) is the Resource handle
    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, resource->Remove(key));
  }
};

template <class K, class V>
class SimpleHashTableExportOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableExportOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DataTypeVector expected_inputs = {DT_RESOURCE};
    DataTypeVector expected_outputs = {DataTypeToEnum<K>::v(),
                                       DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    OP_REQUIRES_OK(ctx, resource->Export(ctx));
  }
};

template <class K, class V>
class SimpleHashTableImportOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableImportOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    SimpleHashTableResource<K, V>* resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &resource));
    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES(
        ctx, keys.shape() == values.shape(),
        errors::InvalidArgument("Shapes of keys and values are not the same: ",
                                keys.shape().DebugString(), " for keys, ",
                                values.shape().DebugString(), "for values."));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = resource->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, resource->Import(keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(resource->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// The "Name" used by REGISTER_KERNEL_BUILDER is defined by REGISTER_OP,
// see simple_hash_table_op.cc.
#define REGISTER_KERNEL(key_dtype, value_dtype)               \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableCreate")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableCreateOpKernel<key_dtype, value_dtype>); \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableFind")                    \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableFindOpKernel<key_dtype, value_dtype>);   \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableInsert")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableInsertOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableRemove")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableRemoveOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableExport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableExportOpKernel<key_dtype, value_dtype>)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples>SimpleHashTableImport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<key_dtype>("key_dtype")             \
          .TypeConstraint<value_dtype>("value_dtype"),        \
      SimpleHashTableImportOpKernel<key_dtype, value_dtype>);

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int32, tstring);
REGISTER_KERNEL(int64_t, double);
REGISTER_KERNEL(int64_t, float);
REGISTER_KERNEL(int64_t, int32);
REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int64_t, tstring);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64_t);
REGISTER_KERNEL(tstring, tstring);
#undef REGISTER_KERNEL

}  // namespace custom_op_examples
}  // namespace tensorflow
