/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_TABLE_RESOURCE_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_TABLE_RESOURCE_UTILS_H_

#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/lookup_tables/lookup_table_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace tables {

// Parent class for tables with support for multithreaded synchronization.
template <typename HeterogeneousKeyType, typename ValueType>
class LookupTableWithSynchronization
    : public LookupTableInterface<HeterogeneousKeyType, ValueType> {
 public:
  // By convention, it is assumed that the OpKernel which creates this
  // resource is bound to an op whose first input is an in64 tensor list with a
  // first element whose boolean value indicates whether a mutex should be
  // exported to synchronize access to state. If the tensor list is empty,
  // the status in ctx is set to InvalidArgument and this object is in an
  // undefined state.
  LookupTableWithSynchronization(OpKernelContext* ctx, OpKernel* kernel) {
    OpInputList table_int64_args;
    OP_REQUIRES_OK(ctx, ctx->input_list("table_int64_args", &table_int64_args));
    if (table_int64_args.size() == 0) {
      ctx->SetStatus(errors::InvalidArgument(
          "table_int64_args should not be empty. Set the first element "
          "to 1 to enable synchronized table use and to 0 otherwise."));
      return;
    }
    if (ctx->input(0).scalar<int64>()() != 0) {
      mutex_ = absl::make_unique<mutex>();
    }
  }

  // Mutex for synchronizing access to unsynchronized methods.
  mutex* GetMutex() const override { return mutex_.get(); }

 private:
  // Use this for locking.
  mutable std::unique_ptr<mutex> mutex_;
};

// Parent class for tables which can be constructed with arbitrary
// lookup fallbacks.
// Since LookupTableInterface::LookupKey assumes that all keys can be mapped
// to values, LookupTableWithFallbackInterface allows clients to implement
// two-stage lookups. If the first key lookup fails, clients can choose
// to perform a fallback lookup using an externally supplied table.
template <typename HeterogeneousKeyType, typename ValueType,
          typename FallbackTableRegisteredType =
              LookupTableInterface<HeterogeneousKeyType, ValueType>>
class LookupTableWithFallbackInterface
    : public LookupTableWithSynchronization<HeterogeneousKeyType, ValueType> {
 public:
  LookupTableWithFallbackInterface(OpKernelContext* ctx, OpKernel* kernel,
                                   FallbackTableRegisteredType* fallback_table)
      : LookupTableWithSynchronization<HeterogeneousKeyType, ValueType>(ctx,
                                                                        kernel),
        fallback_table_(fallback_table) {}

  // Clients are required to fail when ctx is set to a not-OK status in
  // the constructor so this dereference is safe.
  const FallbackTableRegisteredType& fallback_table() const {
    return *fallback_table_;
  }

  ~LookupTableWithFallbackInterface() override {
    if (fallback_table_ != nullptr) {
      fallback_table_->Unref();
    }
  }

 private:
  FallbackTableRegisteredType* fallback_table_;
};

}  // namespace tables
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_TABLE_RESOURCE_UTILS_H_
