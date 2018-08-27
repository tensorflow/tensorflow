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
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/lookup_table_init_op.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

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

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(keys.shape()),
        errors::InvalidArgument("Keys must be a vector, but received shape",
                                keys.shape().DebugString()));

    const Tensor& values = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(values.shape()),
        errors::InvalidArgument("Values must be a vector, but received shape",
                                values.shape().DebugString()));

    OP_REQUIRES(ctx, keys.NumElements() == values.NumElements(),
                errors::InvalidArgument(
                    "Keys and values must have the same size ",
                    keys.NumElements(), " vs ", values.NumElements()));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }

 private:
  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("InitializeTable").Device(DEVICE_CPU),
                        InitializeTableOp);
REGISTER_KERNEL_BUILDER(Name("InitializeTableV2").Device(DEVICE_CPU),
                        InitializeTableOp);

// Kernel to initialize a lookup table from a text file.
//
// After this operation, the table becomes read-only.
class InitializeTableFromTextFileOp : public OpKernel {
 public:
  explicit InitializeTableFromTextFileOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_index", &key_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value_index", &value_index_));
    string delimiter;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delimiter", &delimiter));
    OP_REQUIRES(ctx, delimiter.size() == 1,
                errors::InvalidArgument("delimiter should be only 1 char"));
    delimiter_ = delimiter[0];
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK(ctx,
                   GetInitializableLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, DT_STRING};
    DataTypeVector expected_outputs = {};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& vocab_filename_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(vocab_filename_tensor.shape()),
        errors::InvalidArgument("filename should be a single string, but got ",
                                vocab_filename_tensor.shape().DebugString()));

    string vocab_filename = vocab_filename_tensor.scalar<string>()();
    OP_REQUIRES(ctx, !vocab_filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, lookup::InitializeTableFromTextFile(
                            vocab_filename, vocab_size_, delimiter_, key_index_,
                            value_index_, ctx->env(), table));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }

 private:
  mutex mu_;
  int64 vocab_size_;
  char delimiter_;
  int64 key_index_;
  int64 value_index_;

  TF_DISALLOW_COPY_AND_ASSIGN(InitializeTableFromTextFileOp);
};

REGISTER_KERNEL_BUILDER(Name("InitializeTableFromTextFile").Device(DEVICE_CPU),
                        InitializeTableFromTextFileOp);
REGISTER_KERNEL_BUILDER(
    Name("InitializeTableFromTextFileV2").Device(DEVICE_CPU),
    InitializeTableFromTextFileOp);

}  // namespace tensorflow
