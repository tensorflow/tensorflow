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

#ifndef TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_TABLE_OP_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_TABLE_OP_UTILS_H_

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/meta/type_traits.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/tensor_flag_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tables {

// Create resources of type ContainerBase using the static method
// Functor::AllocateContainer(OpKernelConstruction*, OpKernel*,
// ContainerBase**)
// If the resource has already been created it will be looked up.
template <class ContainerBase, typename Functor>
class ResourceConstructionOp : public OpKernel {
 public:
  explicit ResourceConstructionOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    if (!table_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator = [ctx,
                    this](ContainerBase** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      ContainerBase* container;
      auto status = Functor::AllocateContainer(ctx, this, &container);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        container->Unref();
        return status;
      }
      if (ctx->track_allocations()) {
        ctx->record_persistent_memory_allocation(container->MemoryUsed());
      }
      *ret = container;
      return Status::OK();
    };

    ContainerBase* container_base = nullptr;
    OP_REQUIRES_OK(
        ctx, cinfo_.resource_manager()->template LookupOrCreate<ContainerBase>(
                 cinfo_.container(), cinfo_.name(), &container_base, creator));
    core::ScopedUnref unref_me(container_base);

    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() = MakeResourceHandle<ContainerBase>(
        ctx, cinfo_.container(), cinfo_.name());
    table_handle_set_ = true;
  }

  ~ResourceConstructionOp() override {
    // If the table object was not shared, delete it.
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<ContainerBase>(cinfo_.container(),
                                                cinfo_.name())
               .ok()) {
        // Do nothing; the resource may have been deleted by session resets.
      }
    }
  }

 private:
  mutex mu_;
  bool table_handle_set_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceConstructionOp);
};

// Create resources of type ContainerBase using the static method
// Functor::AllocateContainer(OpKernelConstruction*, OpKernel*,
// FallbackTableBaseType*, ContainerBase**)
// If the resource has already been created it will be looked up.
// Container must decrease the reference count of the FallbackTableBaseType*
// constructor argument before its destructor completes.
template <class ContainerBase, class Functor,
          class FallbackTableBaseType = ContainerBase>
class TableWithFallbackConstructionOp : public OpKernel {
 public:
  explicit TableWithFallbackConstructionOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList table_int64_args;
    OP_REQUIRES_OK(ctx, ctx->input_list("table_int64_args", &table_int64_args));
    if (ctx->num_inputs() == table_int64_args.size()) {
      ctx->SetStatus(errors::InvalidArgument(
          "Expected op to have a resource input after the table_int64_args "
          "input but no such input found."));
      return;
    }

    FallbackTableBaseType* fallback_table = nullptr;
    {
      const Tensor& table_handle = ctx->input(table_int64_args.size());
      ResourceHandle handle(table_handle.scalar<ResourceHandle>()());
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->Lookup(handle.container(),
                                               handle.name(), &fallback_table));
    }
    mutex_lock l(mu_);

    if (!table_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator = [ctx, this, fallback_table](
                       ContainerBase** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // container construction logic can't be merged with
      // ResourceConstructionOp because Container constructor requires an
      // input which can only be constructed if the resource manager
      // internal lock is not already held.
      ContainerBase* container;
      auto status =
          Functor::AllocateContainer(ctx, this, fallback_table, &container);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        container->Unref();
        return status;
      }
      if (ctx->track_allocations()) {
        ctx->record_persistent_memory_allocation(container->MemoryUsed());
      }
      *ret = container;
      return Status::OK();
    };

    ContainerBase* table = nullptr;
    OP_REQUIRES_OK(
        ctx, cinfo_.resource_manager()->template LookupOrCreate<ContainerBase>(
                 cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() = MakeResourceHandle<ContainerBase>(
        ctx, cinfo_.container(), cinfo_.name());
    table_handle_set_ = true;
  }

  ~TableWithFallbackConstructionOp() override {
    // If the table object was not shared, delete it.
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<ContainerBase>(cinfo_.container(),
                                                cinfo_.name())
               .ok()) {
        // Do nothing; the resource may have been deleted by session resets.
      }
    }
  }

 private:
  mutex mu_;
  bool table_handle_set_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(TableWithFallbackConstructionOp);
};

// Used to insert tensors into a container.
template <class Container, class InsertKeyTensorType,
          class InsertValueTensorType>
class HeterogeneousLookupTableInsertOrAssignOp : public OpKernel {
 public:
  explicit HeterogeneousLookupTableInsertOrAssignOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OpInputList table_int64_args;
    OP_REQUIRES_OK(ctx, ctx->input_list("table_int64_args", &table_int64_args));
    const size_t tensor_index_offset = table_int64_args.size();
    const Tensor& keys = ctx->input(tensor_index_offset + 1);
    const Tensor& values = ctx->input(tensor_index_offset + 2);
    if (ABSL_PREDICT_FALSE(keys.NumElements() != values.NumElements())) {
      ctx->SetStatus(errors::InvalidArgument(
          "keys and values do not have the same number of elements: ",
          keys.NumElements(), " vs ", values.NumElements()));
      return;
    }

    const Tensor& table_handle = ctx->input(tensor_index_offset);
    ResourceHandle handle(table_handle.scalar<ResourceHandle>()());
    Container* table;
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(handle.container(),
                                                        handle.name(), &table));
    core::ScopedUnref unref_me(table);

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    auto* mutex = table->GetMutex();
    if (mutex != nullptr) {
      mutex_lock lock(*mutex);
      OP_REQUIRES_OK(ctx, TensorInsert(keys, values, table));
    } else {
      OP_REQUIRES_OK(ctx, TensorInsert(keys, values, table));
    }
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }

 private:
  // Non-variant InsertKeyTensorType which is the same as Container::key_type.
  // No need to static_cast.
  template <typename SfinaeArg = InsertKeyTensorType>
  absl::enable_if_t<
      IsValidDataType<SfinaeArg>::value &&
          std::is_same<SfinaeArg, typename Container::key_type>::value,
      Status>
  TensorInsert(const Tensor& keys, const Tensor& values,
               Container* table) const {
    const auto keys_flat = keys.flat<SfinaeArg>();
    const auto values_flat = values.flat<InsertValueTensorType>();
    return table->BatchInsertOrAssign(
        absl::MakeSpan(keys_flat.data(), keys_flat.size()),
        absl::MakeSpan(values_flat.data(), values_flat.size()));
  }

  // Non-variant InsertKeyTensorType which is otherwise convertible to
  // Container::key_type.
  template <typename SfinaeArg = InsertKeyTensorType>
  absl::enable_if_t<
      IsValidDataType<SfinaeArg>::value &&
          !std::is_same<SfinaeArg, typename Container::key_type>::value &&
          std::is_convertible<SfinaeArg, typename Container::key_type>::value,
      Status>
  TensorInsert(const Tensor& keys, const Tensor& values,
               Container* table) const {
    const auto keys_flat = keys.flat<InsertKeyTensorType>();
    std::vector<typename Container::key_type> keys_vec;
    const auto keys_size = keys_flat.size();
    keys_vec.reserve(keys_size);
    for (size_t i = 0; i < keys_size; ++i) {
      keys_vec.push_back(
          static_cast<typename Container::key_type>(keys_flat(i)));
    }
    const auto values_flat = values.flat<InsertValueTensorType>();
    return table->BatchInsertOrAssign(
        keys_vec, absl::MakeSpan(values_flat.data(), values_flat.size()));
  }

  // Variant InsertKeyTensorType; the wrapped type is convertible to
  // Container::key_type.
  template <typename SfinaeArg = InsertKeyTensorType>
  absl::enable_if_t<
      !IsValidDataType<SfinaeArg>::value &&
          std::is_convertible<typename SfinaeArg::value_type,
                              typename Container::key_type>::value,
      Status>
  TensorInsert(const Tensor& keys, const Tensor& values,
               Container* table) const {
    const auto keys_flat = keys.flat<Variant>();
    std::vector<typename Container::key_type> keys_vec;
    keys_vec.reserve(keys_flat.size());
    for (size_t i = 0; i < keys_flat.size(); ++i) {
      keys_vec.emplace_back(
          *keys_flat(i).get<typename SfinaeArg::value_type>());
    }
    const auto values_flat = values.flat<InsertValueTensorType>();
    return table->BatchInsertOrAssign(
        keys_vec, absl::MakeSpan(values_flat.data(), values_flat.size()));
  }
};

// Used for tensor lookups.
template <class Container, class LookupKeyTensorType, class ValueTensorType>
class HeterogeneousLookupTableFindOp : public OpKernel {
 public:
  explicit HeterogeneousLookupTableFindOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OpInputList table_int64_args;
    {
      auto status = ctx->input_list("table_int64_args", &table_int64_args);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        ctx->SetStatus(status);
        return;
      }
    }
    // We lookup tensors using positional indices because that's more
    // efficient than looking up their string names.
    const Tensor& prefetch_lookahead_t = ctx->input(0);
    const size_t tensor_index_offset = table_int64_args.size();
    const Tensor& keys = ctx->input(tensor_index_offset + 1);
    const Tensor& num_threads = ctx->input(tensor_index_offset + 2);

    TensorShape output_shape = keys.shape();
    Tensor* out;
    {
      auto status = ctx->allocate_output(0, output_shape, &out);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        ctx->SetStatus(status);
        return;
      }
    }

    int64 num_threads_scalar;
    if (TensorShapeUtils::IsScalar(num_threads.shape())) {
      num_threads_scalar = num_threads.template scalar<int64>()();
    } else {
      // Scans through rows of num_threads and returns second entry of first
      // row whose first entry is <= the number of keys to process.
      // This allows the user to control parallelism as a function of
      // the number of keys to lookup.
      num_threads_scalar = tensor_flag_utils::FindConfigValueForKey<int64, int>(
          num_threads.template matrix<int64>(), keys.dim_size(0));
    }
    const int64 num_keys_per_thread =
        num_threads_scalar > 0
            ? std::max(1ll, keys.dim_size(0) / num_threads_scalar)
            : keys.dim_size(0);

    const int64 prefetch_lookahead = prefetch_lookahead_t.scalar<int64>()();

    const Tensor& table_handle = ctx->input(tensor_index_offset);
    ResourceHandle handle(table_handle.scalar<ResourceHandle>()());
    Container* table;
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(handle.container(),
                                                        handle.name(), &table));
    core::ScopedUnref unref_me(table);

    auto* mutex = table->GetMutex();
    auto* threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    if (mutex != nullptr) {
      // There are many subtle problems with using reader locks so we opt for a
      // writer lock here.
      mutex_lock lock(*mutex);
      OP_REQUIRES_OK(
          ctx, TensorLookup(*table, prefetch_lookahead, num_keys_per_thread,
                            keys, out, threadpool));
    } else {
      OP_REQUIRES_OK(
          ctx, TensorLookup(*table, prefetch_lookahead, num_keys_per_thread,
                            keys, out, threadpool));
    }
  }

 private:
  // keys and *values arguments to TensorLookup must have the same number of
  // elements. This is guaranteed above.

  // 'Simple' types below are types which are not natively supported in TF.
  // Simple LookupKeyTensorType which is the same as Container::key_type.
  template <typename SfinaeArg = LookupKeyTensorType>
  absl::enable_if_t<
      IsValidDataType<SfinaeArg>::value &&
          std::is_same<SfinaeArg, typename Container::key_type>::value,
      Status>
  TensorLookup(Container& table, int64 prefetch_lookahead,
               int64 num_keys_per_thread, const Tensor& keys, Tensor* values,
               thread::ThreadPool* threadpool) const {
    const auto keys_flat = keys.flat<LookupKeyTensorType>();
    const auto keys_size = keys_flat.size();
    auto key_span = absl::MakeSpan(keys_flat.data(), keys_size);
    auto value_span = absl::MakeSpan(values->flat<ValueTensorType>().data(),
                                     values->NumElements());
    return MultithreadedTensorLookup(table, prefetch_lookahead,
                                     num_keys_per_thread, key_span, value_span,
                                     threadpool);
  }

  // Try to implicitly convert all other simple LookupKeyTensorTypes to
  // Container::key_type.
  template <typename SfinaeArg = LookupKeyTensorType>
  absl::enable_if_t<
      IsValidDataType<SfinaeArg>::value &&
          !std::is_same<SfinaeArg, typename Container::key_type>::value,
      Status>
  TensorLookup(Container& table, int64 prefetch_lookahead,
               int64 num_keys_per_thread, const Tensor& keys, Tensor* values,
               thread::ThreadPool* threadpool) const {
    const auto keys_flat = keys.flat<LookupKeyTensorType>();
    std::vector<typename Container::key_type> keys_vec;
    const auto keys_size = keys_flat.size();
    keys_vec.reserve(keys_size);
    for (size_t i = 0; i < keys_size; ++i) {
      keys_vec.emplace_back(keys_flat(i));
    }
    absl::Span<typename Container::key_type> key_span(keys_vec);
    auto value_span = absl::MakeSpan(values->flat<ValueTensorType>().data(),
                                     values->NumElements());
    return MultithreadedTensorLookup(table, prefetch_lookahead,
                                     num_keys_per_thread, key_span, value_span,
                                     threadpool);
  }

  // Non-simple LookupKeyTensorType. We'll try an implicit conversion to
  // Container::key_type.
  template <typename VariantSubType = LookupKeyTensorType>
  absl::enable_if_t<!IsValidDataType<VariantSubType>::value, Status>
  TensorLookup(Container& table, int64 prefetch_lookahead,
               int64 num_keys_per_thread, const Tensor& keys, Tensor* values,
               thread::ThreadPool* threadpool) const {
    const auto keys_flat = keys.flat<Variant>();
    std::vector<typename Container::key_type> keys_vec;
    const auto keys_size = keys_flat.size();
    keys_vec.reserve(keys_size);
    for (size_t i = 0; i < keys_size; ++i) {
      keys_vec.emplace_back(
          *keys_flat(i).get<typename VariantSubType::value_type>());
    }
    absl::Span<typename Container::key_type> key_span(keys_vec);
    auto value_span = absl::MakeSpan(values->flat<ValueTensorType>().data(),
                                     values->NumElements());
    return MultithreadedTensorLookup(table, prefetch_lookahead,
                                     num_keys_per_thread, key_span, value_span,
                                     threadpool);
  }

  // Wrapper around table.BatchLookup which permits sharding across cores.
  template <typename K, typename V>
  Status MultithreadedTensorLookup(Container& table, int64 prefetch_lookahead,
                                   int64 num_keys_per_thread,
                                   absl::Span<K> keys, absl::Span<V> values,
                                   thread::ThreadPool* threadpool) const {
    mutex temp_mutex;  // Protect status.
    Status status;
    auto lookup_keys = [&, this](int64 begin, int64 end) {
      auto temp_status = table.BatchLookup(keys.subspan(begin, end - begin),
                                           values.subspan(begin, end - begin),
                                           prefetch_lookahead);
      if (ABSL_PREDICT_FALSE(!temp_status.ok())) {
        mutex_lock lock(temp_mutex);
        status.Update(temp_status);
      }
    };
    threadpool->TransformRangeConcurrently(num_keys_per_thread /* block_size */,
                                           keys.size(), lookup_keys);
    return status;
  }
};

// Op that returns the size of a container.
template <class Container>
class ContainerSizeOp : public OpKernel {
 public:
  explicit ContainerSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& container_handle = ctx->input(0);
    ResourceHandle handle(container_handle.scalar<ResourceHandle>()());
    Container* container;
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(
                            handle.container(), handle.name(), &container));
    core::ScopedUnref unref_me(container);
    OP_REQUIRES_OK(ctx, container->SizeStatus());

    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));

    auto* mutex = container->GetMutex();
    if (mutex != nullptr) {
      tf_shared_lock lock(*mutex);
      out->scalar<int64>()() = container->UnsafeSize();
    } else {
      out->scalar<int64>()() = container->UnsafeSize();
    }
  }
};

}  // namespace tables
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_TABLE_OP_UTILS_H_
