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

#ifndef TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_OP_KERNEL_TEMPLATES_H_
#define TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_OP_KERNEL_TEMPLATES_H_

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

// Create resources of type ResourceType and AliasesToRegister using
// Functor::AllocateContainer(OpKernelConstruction*, OpKernel*,
// ResourceType**). ResourceType = Functor::resource_type.
// No-op for resources which have already been created.
template <typename Functor, typename... AliasesToRegister>
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
                    this](ResourceType** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      ResourceType* resource = nullptr;
      auto status = Functor::AllocateContainer(ctx, this, &resource);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        // Ideally resource is non-null only if status is OK but we try
        // to compensate here.
        if (resource != nullptr) {
          resource->Unref();
        }
        return status;
      }
      if (ctx->track_allocations()) {
        ctx->record_persistent_memory_allocation(resource->MemoryUsed());
      }
      *ret = resource;
      return Status::OK();
    };

    // Register the ResourceType alias.
    ResourceType* resource = nullptr;
    core::ScopedUnref unref_me(resource);
    OP_REQUIRES_OK(
        ctx,
        cinfo_.resource_manager()->template LookupOrCreate<ResourceType, true>(
            cinfo_.container(), cinfo_.name(), &resource, creator));

    // Put a handle to resource in the output tensor (the other aliases will
    // have the same handle).
    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() = MakeResourceHandle<ResourceType>(
        ctx, cinfo_.container(), cinfo_.name());
    table_handle_set_ = true;

    // Create other alias resources.
    Status status;
    int dummy[sizeof...(AliasesToRegister)] = {
        (status.Update(RegisterAlias<AliasesToRegister>(resource)), 0)...};
    (void)dummy;
    OP_REQUIRES_OK(ctx, status);
  }

  ~ResourceConstructionOp() override {
    // If the table object was not shared, delete it.
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<ResourceType>(cinfo_.container(),
                                               cinfo_.name())
               .ok()) {
        // Do nothing; the resource may have been deleted by session resets.
      }
      // Attempt to delete other resource aliases.
      Status dummy_status;
      int dummy[sizeof...(AliasesToRegister)] = {
          (dummy_status.Update(DeleteAlias<AliasesToRegister>()), 0)...};
      (void)dummy;
    }
  }

 private:
  using ResourceType = typename Functor::resource_type;
  template <typename T>
  Status RegisterAlias(ResourceType* resource) {
    auto creator = [resource](T** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      *ret = resource;
      return Status::OK();
    };

    T* alias_resource = nullptr;
    core::ScopedUnref unref_me(alias_resource);
    return cinfo_.resource_manager()->template LookupOrCreate<T, true>(
        cinfo_.container(), cinfo_.name(), &alias_resource, creator);
  }

  template <typename T>
  Status DeleteAlias() {
    return cinfo_.resource_manager()->template Delete<T>(cinfo_.container(),
                                                         cinfo_.name());
  }

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
template <typename Functor, typename... AliasesToRegister>
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

    // Look up the fallback table.
    FallbackTableBaseType* fallback_table = nullptr;
    {
      const Tensor& table_handle = ctx->input(table_int64_args.size());
      ResourceHandle handle(table_handle.scalar<ResourceHandle>()());
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->Lookup<FallbackTableBaseType, true>(
                   handle.container(), handle.name(), &fallback_table));
    }
    mutex_lock l(mu_);

    if (!table_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator = [ctx, this, fallback_table](
                       ResourceType** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // container construction logic can't be merged with
      // ResourceConstructionOp because Container constructor requires an
      // input which can only be constructed if the resource manager
      // internal lock is not already held.
      ResourceType* resource = nullptr;
      auto status =
          Functor::AllocateContainer(ctx, this, fallback_table, &resource);
      if (ABSL_PREDICT_FALSE(!status.ok())) {
        // Ideally resource is non-null only if status is OK but we try
        // to compensate here.
        if (resource != nullptr) {
          resource->Unref();
        }
        return status;
      }
      if (ctx->track_allocations()) {
        ctx->record_persistent_memory_allocation(resource->MemoryUsed());
      }
      *ret = resource;
      return Status::OK();
    };

    // Register the ResourceType alias.
    ResourceType* table = nullptr;
    core::ScopedUnref unref_me(table);
    OP_REQUIRES_OK(
        ctx,
        cinfo_.resource_manager()->template LookupOrCreate<ResourceType, true>(
            cinfo_.container(), cinfo_.name(), &table, creator));

    // Put a handle to resource in the output tensor (the other aliases will
    // have the same handle).
    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() = MakeResourceHandle<ResourceType>(
        ctx, cinfo_.container(), cinfo_.name());
    table_handle_set_ = true;

    // Create other alias resources.
    Status status;
    int dummy[sizeof...(AliasesToRegister)] = {
        (status.Update(RegisterAlias<AliasesToRegister>(table)), 0)...};
    (void)dummy;
    OP_REQUIRES_OK(ctx, status);
  }

  ~TableWithFallbackConstructionOp() override {
    // If the table object was not shared, delete it.
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<ResourceType>(cinfo_.container(),
                                               cinfo_.name())
               .ok()) {
        // Do nothing; the resource may have been deleted by session resets.
      }
      // Attempt to delete other resource aliases.
      Status dummy_status;
      int dummy[sizeof...(AliasesToRegister)] = {
          (dummy_status.Update(DeleteAlias<AliasesToRegister>()), 0)...};
      (void)dummy;
    }
  }

 private:
  using ResourceType = typename Functor::resource_type;
  using FallbackTableBaseType = typename Functor::fallback_table_type;

  template <typename T>
  Status RegisterAlias(ResourceType* resource) {
    auto creator = [resource](T** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      *ret = resource;
      return Status::OK();
    };

    T* alias_resource = nullptr;
    core::ScopedUnref unref_me(alias_resource);
    return cinfo_.resource_manager()->template LookupOrCreate<T, true>(
        cinfo_.container(), cinfo_.name(), &alias_resource, creator);
  }

  template <typename T>
  Status DeleteAlias() {
    return cinfo_.resource_manager()->template Delete<T>(cinfo_.container(),
                                                         cinfo_.name());
  }

  mutex mu_;
  bool table_handle_set_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(TableWithFallbackConstructionOp);
};

// Lookup a table of type ResourceAlias and insert the passed in keys and
// values tensors using Functor::TensorInsert(keys, values, table).
template <typename Functor,
          typename ResourceAlias = typename Functor::resource_type>
class LookupTableInsertOp : public OpKernel {
 public:
  explicit LookupTableInsertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OpInputList table_int64_args;
    OP_REQUIRES_OK(ctx, ctx->input_list("table_int64_args", &table_int64_args));
    const size_t tensor_index_offset = table_int64_args.size();
    // Business logic for checking tensor shapes, etc, is delegated to the
    // Functor.
    const Tensor& keys = ctx->input(tensor_index_offset + 1);
    const Tensor& values = ctx->input(tensor_index_offset + 2);

    const Tensor& table_handle = ctx->input(tensor_index_offset);
    ResourceHandle handle(table_handle.scalar<ResourceHandle>()());
    ResourceAlias* table;
    core::ScopedUnref unref_me(table);
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup<ResourceAlias, true>(
                            handle.container(), handle.name(), &table));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    auto* mutex = table->GetMutex();
    if (mutex != nullptr) {
      mutex_lock lock(*mutex);
      OP_REQUIRES_OK(ctx, Functor::TensorInsert(keys, values, table));
    } else {
      OP_REQUIRES_OK(ctx, Functor::TensorInsert(keys, values, table));
    }
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LookupTableInsertOp);
};

// Lookup a table of type ResourceAlias and look up the passed in keys using
// Functor::TensorLookup(
//     table, keys, prefetch_lookahead, num_keys_per_thread, threadpool, out).
template <typename Functor,
          typename ResourceAlias = typename Functor::resource_type>
class LookupTableFindOp : public OpKernel {
 public:
  explicit LookupTableFindOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    ResourceAlias* table;
    core::ScopedUnref unref_me(table);
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup<ResourceAlias, true>(
                            handle.container(), handle.name(), &table));

    auto* mutex = table->GetMutex();
    auto* threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    if (mutex != nullptr) {
      // There are many subtle problems with using reader locks so we opt for a
      // writer lock here.
      mutex_lock lock(*mutex);
      OP_REQUIRES_OK(
          ctx, Functor::TensorLookup(*table, keys, prefetch_lookahead,
                                     num_keys_per_thread, threadpool, out));
    } else {
      OP_REQUIRES_OK(
          ctx, Functor::TensorLookup(*table, keys, prefetch_lookahead,
                                     num_keys_per_thread, threadpool, out));
    }
  }
};

// Lookup a container of type ResourceAlias and return its size using
// Functor::Size(container, &size).
template <typename Functor,
          typename ResourceAlias = typename Functor::resource_type>
class ContainerSizeOp : public OpKernel {
 public:
  explicit ContainerSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& container_handle = ctx->input(0);
    ResourceHandle handle(container_handle.scalar<ResourceHandle>()());
    ResourceAlias* container;
    core::ScopedUnref unref_me(container);
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup<ResourceAlias, true>(
                            handle.container(), handle.name(), &container));

    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));

    auto* mutex = container->GetMutex();
    if (mutex != nullptr) {
      tf_shared_lock lock(*mutex);
      OP_REQUIRES_OK(ctx, Functor::Size(*container, &out->scalar<uint64>()()));
    } else {
      OP_REQUIRES_OK(ctx, Functor::Size(*container, &out->scalar<uint64>()()));
    }
  }
};

}  // namespace tables
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_OP_KERNEL_TEMPLATES_H_
