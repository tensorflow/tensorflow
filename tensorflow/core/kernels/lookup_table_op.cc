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

#include "tensorflow/core/kernels/lookup_table_op.h"
#define EIGEN_USE_THREADS

#include <string>
#include <type_traits>
#include <utility>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace lookup {
namespace {

// Ensure that the compiler cannot elide a copy into a local, for
// bounds checking on source tensors that might be updated asynchronously for
// integral types. However non-integer variables are not allowed and therefore
// the local copy is unnecessary.
template <typename T>
T SubtleMustCopyUnlessStringOrFloat(const T& value) {
  return internal::SubtleMustCopy(value);
}

const string& SubtleMustCopyUnlessStringOrFloat(const string& value) {
  return value;
}

const float SubtleMustCopyUnlessStringOrFloat(const float value) {
  return value;
}

}  // namespace

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

  Status DoInsert(const Tensor& keys, const Tensor& values) override {
    if (!table_) {
      return errors::FailedPrecondition("HashTable is not prepared.");
    }

    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();
    for (int64 i = 0; i < key_values.size(); ++i) {
      const K key = SubtleMustCopyUnlessStringOrFloat(key_values(i));
      const V value = SubtleMustCopyUnlessStringOrFloat(value_values(i));
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
          *table_, SubtleMustCopyUnlessStringOrFloat(key_values(i)),
          default_val);
    }
    return Status::OK();
  }

 private:
  std::unique_ptr<std::unordered_map<K, V>> table_;
};

// Lookup table that wraps an unordered_map, where the key and value data type
// is specified. Each individual value must be a scalar. If vector values are
// required, use MutableHashTableOfTensors.
//
// This table is mutable and thread safe - Insert can be called at any time.
//
// Sample use case:
//
// MutableHashTableOfScalars<int64, int64> table;  // int64 -> int64.
// // Populate the table, elements could be added in one or multiple calls.
// table.Insert(key_tensor, value_tensor); // Populate the table.
//
// table.Find(in_t, &out_t, default_t)
//
template <class K, class V>
class MutableHashTableOfScalars final : public LookupInterface {
 public:
  MutableHashTableOfScalars(OpKernelContext* ctx, OpKernel* kernel) {}

  size_t size() const override {
    mutex_lock l(mu_);
    return table_.size();
  }

  Status Find(const Tensor& key, Tensor* value,
              const Tensor& default_value) override {
    const V default_val = default_value.flat<V>()(0);
    const auto key_values = key.flat<K>();
    auto value_values = value->flat<V>();

    mutex_lock l(mu_);
    for (int64 i = 0; i < key_values.size(); ++i) {
      value_values(i) = gtl::FindWithDefault(
          table_, SubtleMustCopyUnlessStringOrFloat(key_values(i)),
          default_val);
    }

    return Status::OK();
  }

  Status DoInsert(bool clear, const Tensor& keys, const Tensor& values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();

    mutex_lock l(mu_);
    if (clear) {
      table_.clear();
    }
    for (int64 i = 0; i < key_values.size(); ++i) {
      const K key = SubtleMustCopyUnlessStringOrFloat(key_values(i));
      const V value = SubtleMustCopyUnlessStringOrFloat(value_values(i));
      gtl::InsertOrUpdate(&table_, key, value);
    }
    return Status::OK();
  }

  Status Insert(const Tensor& keys, const Tensor& values) override {
    return DoInsert(false, keys, values);
  }

  Status ImportValues(const Tensor& keys, const Tensor& values) override {
    return DoInsert(true, keys, values);
  }

  Status ExportValues(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    int64 size = table_.size();

    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("values", TensorShape({size}), &values));

    auto keys_data = keys->flat<K>();
    auto values_data = values->flat<V>();
    int64 i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it->first;
      values_data(i) = it->second;
    }
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape value_shape() const override { return TensorShape(); }

 private:
  // TODO(andreasst): consider using a read/write lock or a concurrent map
  mutable mutex mu_;
  std::unordered_map<K, V> table_ GUARDED_BY(mu_);
};

// Lookup table that wraps an unordered_map. Behaves identical to
// MutableHashTableOfScalars except that each value must be a vector.
template <class K, class V>
class MutableHashTableOfTensors final : public LookupInterface {
 public:
  MutableHashTableOfTensors(OpKernelContext* ctx, OpKernel* kernel) {
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
  }

  size_t size() const override {
    mutex_lock l(mu_);
    return table_.size();
  }

  Status Find(const Tensor& key, Tensor* value,
              const Tensor& default_value) override {
    const auto default_flat = default_value.flat<V>();
    const auto key_values = key.flat<K>();
    auto value_values = value->flat_inner_dims<V, 2>();
    int64 value_dim = value_shape_.dim_size(0);

    mutex_lock l(mu_);
    for (int64 i = 0; i < key_values.size(); ++i) {
      ValueArray* value_vec = gtl::FindOrNull(
          table_, SubtleMustCopyUnlessStringOrFloat(key_values(i)));
      if (value_vec != nullptr) {
        for (int64 j = 0; j < value_dim; j++) {
          value_values(i, j) = value_vec->at(j);
        }
      } else {
        for (int64 j = 0; j < value_dim; j++) {
          value_values(i, j) = default_flat(j);
        }
      }
    }

    return Status::OK();
  }

  Status DoInsert(bool clear, const Tensor& keys, const Tensor& values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat_inner_dims<V, 2>();
    int64 value_dim = value_shape_.dim_size(0);

    mutex_lock l(mu_);
    if (clear) {
      table_.clear();
    }
    for (int64 i = 0; i < key_values.size(); ++i) {
      const K key = SubtleMustCopyUnlessStringOrFloat(key_values(i));
      ValueArray value_vec;
      for (int64 j = 0; j < value_dim; j++) {
        V value = value_values(i, j);
        value_vec.push_back(value);
      }
      gtl::InsertOrUpdate(&table_, key, value_vec);
    }
    return Status::OK();
  }

  Status Insert(const Tensor& keys, const Tensor& values) override {
    return DoInsert(false, keys, values);
  }

  Status ImportValues(const Tensor& keys, const Tensor& values) override {
    return DoInsert(true, keys, values);
  }

  Status ExportValues(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    int64 size = table_.size();
    int64 value_dim = value_shape_.dim_size(0);

    Tensor* keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({size}), &keys));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({size, value_dim}), &values));

    auto keys_data = keys->flat<K>();
    auto values_data = values->matrix<V>();
    int64 i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      K key = it->first;
      ValueArray value = it->second;
      keys_data(i) = key;
      for (int64 j = 0; j < value_dim; j++) {
        values_data(i, j) = value[j];
      }
    }
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape value_shape() const override { return value_shape_; }

 private:
  TensorShape value_shape_;
  // TODO(andreasst): consider using a read/write lock or a concurrent map
  mutable mutex mu_;
  typedef gtl::InlinedVector<V, 4> ValueArray;
  std::unordered_map<K, ValueArray> table_ GUARDED_BY(mu_);
};

}  // namespace lookup

// Table lookup op. Perform the lookup operation on the given table.
class LookupTableFindOp : public OpKernel {
 public:
  explicit LookupTableFindOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {DT_STRING_REF, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckFindArguments(key, default_value));

    TensorShape output_shape = key.shape();
    output_shape.AppendShape(table->value_shape());
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, table->Find(key, out, default_value));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableFind").Device(DEVICE_CPU),
                        LookupTableFindOp);

// Table insert op.
class LookupTableInsertOp : public OpKernel {
 public:
  explicit LookupTableInsertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {DT_STRING_REF, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensors(keys, values));
    OP_REQUIRES_OK(ctx, table->Insert(keys, values));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableInsert").Device(DEVICE_CPU),
                        LookupTableInsertOp);

// Op that returns the size of the given table.
class LookupTableSizeOp : public OpKernel {
 public:
  explicit LookupTableSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
    out->flat<int64>().setConstant(table->size());
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableSize").Device(DEVICE_CPU),
                        LookupTableSizeOp);

// Op that outputs tensors of all keys and all values.
class LookupTableExportOp : public OpKernel {
 public:
  explicit LookupTableExportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableExport").Device(DEVICE_CPU),
                        LookupTableExportOp);

// Clear the table and insert data.
class LookupTableImportOp : public OpKernel {
 public:
  explicit LookupTableImportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {DT_STRING_REF, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensors(keys, values));
    OP_REQUIRES_OK(ctx, table->ImportValues(keys, values));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableImport").Device(DEVICE_CPU),
                        LookupTableImportOp);

// Register the HashTable op with the currently supported key and value types.
#define REGISTER_KERNEL(key_dtype, value_dtype)                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("HashTable")                                                   \
          .Device(DEVICE_CPU)                                             \
          .TypeConstraint<key_dtype>("key_dtype")                         \
          .TypeConstraint<value_dtype>("value_dtype"),                    \
      LookupTableOp<lookup::HashTable<key_dtype, value_dtype>, key_dtype, \
                    value_dtype>)

REGISTER_KERNEL(string, int64);
REGISTER_KERNEL(int64, string);

#undef REGISTER_KERNEL

// Register the MutableHashTable op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MutableHashTable")                                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      LookupTableOp<lookup::MutableHashTableOfScalars<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)

REGISTER_KERNEL(string, float);
REGISTER_KERNEL(string, int64);
REGISTER_KERNEL(int64, string);

#undef REGISTER_KERNEL

// Register the MutableHashTableOfTensors op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MutableHashTableOfTensors")                                        \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      LookupTableOp<lookup::MutableHashTableOfTensors<key_dtype, value_dtype>, \
                    key_dtype, value_dtype>)

REGISTER_KERNEL(string, float);
REGISTER_KERNEL(string, int64);
REGISTER_KERNEL(int64, string);

#undef REGISTER_KERNEL

}  // namespace tensorflow
