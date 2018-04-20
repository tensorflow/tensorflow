/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_LOOKUP_TABLE_OP_H_
#define TENSORFLOW_KERNELS_LOOKUP_TABLE_OP_H_

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Lookup table op that supports different table implementations specified by
// the 'Container' template. Container must be derived from LookupInterface. The
// key and value are of the templated type "key_dtype" and "value_dtype"
// respectively.
template <class Container, class key_dtype, class value_dtype>
class LookupTableOp : public OpKernel {
 public:
  // ctx is not owned by this class.
  explicit LookupTableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    OP_REQUIRES_OK(ctx, ctx->allocate_persistent(tensorflow::DT_STRING,
                                                 tensorflow::TensorShape({2}),
                                                 &table_handle_, nullptr));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  // ctx is not owned by this function.
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    if (!table_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator = [ctx, this](lookup::LookupInterface** ret) {
      lookup::LookupInterface* container = new Container(ctx, this);
      if (!ctx->status().ok()) {
        container->Unref();
        return ctx->status();
      }
      if (ctx->track_allocations()) {
        ctx->record_persistent_memory_allocation(
            container->MemoryUsed() + table_handle_.AllocatedBytes());
      }
      *ret = container;
      return Status::OK();
    };

    lookup::LookupInterface* table = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()
                       ->template LookupOrCreate<lookup::LookupInterface>(
                           cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, lookup::CheckTableDataTypes(
                            *table, DataTypeToEnum<key_dtype>::v(),
                            DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      Tensor* handle;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
      handle->scalar<ResourceHandle>()() =
          MakeResourceHandle<lookup::LookupInterface>(ctx, cinfo_.container(),
                                                      cinfo_.name());
    } else {
      if (!table_handle_set_) {
        auto h = table_handle_.AccessTensor(ctx)->template flat<string>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      ctx->set_output_ref(0, &mu_, table_handle_.AccessTensor(ctx));
    }
    table_handle_set_ = true;
  }

  ~LookupTableOp() override {
    // If the table object was not shared, delete it.
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(
          cinfo_.resource_manager()->template Delete<lookup::LookupInterface>(
              cinfo_.container(), cinfo_.name()));
    }
  }

 private:
  mutex mu_;
  PersistentTensor table_handle_ GUARDED_BY(mu_);
  bool table_handle_set_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(LookupTableOp);
};

namespace lookup {

// Ensure that the compiler cannot elide a copy into a local, for
// bounds checking on source tensors that might be updated asynchronously for
// integral types. However non-integer variables are not allowed and therefore
// the local copy is unnecessary.
template <typename T>
T SubtleMustCopyIfIntegral(const T& value) {
  return internal::SubtleMustCopy(value);
}

inline const string& SubtleMustCopyIfIntegral(const string& value) {
  return value;
}

inline const float SubtleMustCopyIfIntegral(const float value) { return value; }

inline const double SubtleMustCopyIfIntegral(const double value) {
  return value;
}

inline const Variant& SubtleMustCopyIfIntegral(const Variant& value) {
  return value;
}

// Lookup table that wraps an unordered_map, where the key and value data type
// is specified.
//
// This table is recommended for any variations to key values.
//
// For look up, the table is required to be initialized (allocated
// and populated). Once the table is marked as initialized it becomes read-only.
//
// Sample use case:
//
// HashTable<int64, int64> table;  // int64 -> int64.
// table.Prepare(10); // Prepare the underlying data structure, the number of
//                    // elements is required by interface, but not used.
// // Populate the table, elements could be added in one or multiple calls.
// table.Insert(key_tensor, value_tensor); // Populate the table.
// ...
// table.set_is_initialized();
//
// table.Find(in_t, &out_t, default_t)
//
template <class K, class V>
class HashTable : public InitializableLookupTable {
 public:
  HashTable(OpKernelContext* ctx, OpKernel* kernel) {}

  size_t size() const override {
    // return the size of the table only if it's initialized, otherwise 0.
    if (!is_initialized_) {
      return 0;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return table_ ? table_->size() : 0;
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

 protected:
  Status DoPrepare(size_t unused) override {
    if (is_initialized_) {
      return errors::Aborted("HashTable already initialized.");
    }
    if (!table_) {
      table_ = std::unique_ptr<std::unordered_map<K, V>>(
          new std::unordered_map<K, V>());
    }
    return Status::OK();
  };

  Status DoLazyPrepare(std::function<int64(void)> unused) override {
    constexpr size_t kUnusedSize = 0;
    return DoPrepare(kUnusedSize);
  }

  Status DoInsert(const Tensor& keys, const Tensor& values) override {
    if (!table_) {
      return errors::FailedPrecondition("HashTable is not prepared.");
    }

    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();
    for (int64 i = 0; i < key_values.size(); ++i) {
      const K key = SubtleMustCopyIfIntegral(key_values(i));
      const V value = SubtleMustCopyIfIntegral(value_values(i));
      const V& previous_value = gtl::LookupOrInsert(table_.get(), key, value);
      if (previous_value != value) {
        return errors::FailedPrecondition(
            "HashTable has different value for same key. Key ", key, " has ",
            previous_value, " and trying to add value ", value);
      }
    }
    return Status::OK();
  }

  Status DoFind(const Tensor& key, Tensor* value,
                const Tensor& default_value) override {
    const V default_val = default_value.flat<V>()(0);
    const auto key_values = key.flat<K>();
    auto value_values = value->flat<V>();

    for (int64 i = 0; i < key_values.size(); ++i) {
      value_values(i) = gtl::FindWithDefault(
          *table_, SubtleMustCopyIfIntegral(key_values(i)), default_val);
    }
    return Status::OK();
  }

  int64 MemoryUsed() const override {
    if (table_) {
      const int64 num_elements = table_->size();
      return num_elements * (sizeof(K) + sizeof(V));
    } else {
      return 0;
    }
  }

 private:
  std::unique_ptr<std::unordered_map<K, V>> table_;
};

}  // namespace lookup

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_LOOKUP_TABLE_OP_H_
