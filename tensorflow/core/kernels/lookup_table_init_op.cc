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

#define EIGEN_USE_THREADS

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lookup {

// Iterator to initialize tables given 'keys' and 'values' tensors.
//
// The two tensors are returned in the first iteration. It doesn't loop
// over each element of the tensor since insertions in the lookup table can
// process batches.
class KeyValueTensorIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  // keys and values are not owned by the iterator.
  explicit KeyValueTensorIterator(const Tensor* keys, const Tensor* values)
      : keys_(keys), values_(values), valid_(true), status_(Status::OK()) {
    TensorShape key_shape = keys_->shape();
    if (!key_shape.IsSameSize(values_->shape())) {
      valid_ = false;
      status_ = errors::InvalidArgument(
          "keys and values should have the same dimension.",
          key_shape.DebugString(), " vs ", values_->shape().DebugString());
    }
    if (key_shape.num_elements() == 0) {
      valid_ = false;
      status_ =
          errors::InvalidArgument("keys and values cannot be empty tensors.");
    }
  }

  bool Valid() const override { return valid_; }

  void Next() override {
    valid_ = false;
    status_ = errors::OutOfRange("No more data.");
  }

  const Tensor& keys() const override { return *keys_; }

  const Tensor& values() const override { return *values_; }

  Status status() const override { return status_; }

  int64 total_size() const {
    return keys_ == nullptr ? -1 : keys_->NumElements();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(KeyValueTensorIterator);

  const Tensor* keys_;    // Doesn't own it.
  const Tensor* values_;  // Doesn't own it.
  bool valid_;            // true if the iterator points to an existing range.
  Status status_;
};

}  // namespace lookup

// Kernel to initialize a look table given a key and value tensors.
// After this operation, the table becomes read-only.
class InitializeTableOp : public OpKernel {
 public:
  explicit InitializeTableOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK(ctx,
                   GetInitializableLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {DT_STRING_REF, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(keys.shape()),
                errors::InvalidArgument("Keys must be a vector, but received ",
                                        keys.shape().DebugString()));

    const Tensor& values = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(values.shape()),
        errors::InvalidArgument("Values must be a vector, but received ",
                                values.shape().DebugString()));

    OP_REQUIRES(ctx, keys.NumElements() == values.NumElements(),
                errors::InvalidArgument(
                    "Keys and values must have the same size ",
                    keys.NumElements(), " vs ", values.NumElements()));

    lookup::KeyValueTensorIterator iter(&keys, &values);
    OP_REQUIRES_OK(ctx, table->Initialize(iter));
  }

 private:
  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("InitializeTable").Device(DEVICE_CPU),
                        InitializeTableOp);

}  // namespace tensorflow
