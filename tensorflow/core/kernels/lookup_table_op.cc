/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace lookup {
namespace {

// Ensure that the compiler cannot elide a copy into a local, for
// bounds checking on source tensors that might be updated asynchronously for
// integral types. However strings variables are not allowed and therefore the
// local copy is unnecessary.
template <typename T>
T SubtleMustCopyUnlessString(const T& value) {
  return internal::SubtleMustCopy(value);
}

const string& SubtleMustCopyUnlessString(const string& value) { return value; }

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
    for (int i = 0; i < key_values.size(); ++i) {
      const K key = SubtleMustCopyUnlessString(key_values(i));
      const V value = SubtleMustCopyUnlessString(value_values(i));
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

    for (int i = 0; i < key_values.size(); ++i) {
      value_values(i) = gtl::FindWithDefault(
          *table_, SubtleMustCopyUnlessString(key_values(i)), default_val);
    }
    return Status::OK();
  }

 private:
  std::unique_ptr<std::unordered_map<K, V>> table_;
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

    const Tensor& input = ctx->input(1);

    const Tensor& default_value = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(default_value.shape()),
                errors::InvalidArgument("Default value must be a scalar, not ",
                                        default_value.shape().DebugString()));

    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", input.shape(), &out));

    OP_REQUIRES_OK(ctx, table->Find(input, out, default_value));
  }
};

REGISTER_KERNEL_BUILDER(Name("LookupTableFind").Device(DEVICE_CPU),
                        LookupTableFindOp);

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

}  // namespace tensorflow
