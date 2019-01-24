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

#include "absl/base/thread_annotations.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tables {

// Lookup or create resources of type Container but treat them as having type
// ContainerBase for the purpose of dynamic dispatching.
// Container must have constructor Container(OpKernelContext*, OpKernel*)
template <class ContainerBase, class ContainerChild>
class ResourceConstructionOp : public OpKernel {
 public:
  explicit ResourceConstructionOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    static_assert(std::is_base_of<ContainerBase, ContainerChild>::value,
                  "ContainerChild is not derived from ContainerBase");
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
      ContainerBase* container = new ContainerChild(ctx, this);
      if (!ctx->status().ok()) {
        container->Unref();
        return ctx->status();
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
  PersistentTensor table_handle_ GUARDED_BY(mu_);
  bool table_handle_set_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceConstructionOp);
};

// Lookup or create resources of type Container but treat them as having type
// ContainerBase for the purpose of dynamic dispatching.
// Container must have constructor
// Container(OpKernelContext*, OpKernel*, FallbackTableBaseType*)
// Container must decrease the reference count of the FallbackTableBaseType*
// constructor argument before its destructor completes.
template <class ContainerBase, class ContainerChild,
          class FallbackTableBaseType = ContainerBase>
class TableWithFallbackConstructionOp : public OpKernel {
 public:
  explicit TableWithFallbackConstructionOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    static_assert(std::is_base_of<ContainerBase, ContainerChild>::value,
                  "ContainerChild is not derived from ContainerBase");
    OP_REQUIRES_OK(ctx, ctx->allocate_persistent(tensorflow::DT_STRING,
                                                 tensorflow::TensorShape({2}),
                                                 &table_handle_, nullptr));
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
      ContainerBase* container = new ContainerChild(ctx, this, fallback_table);
      if (!ctx->status().ok()) {
        container->Unref();
        return ctx->status();
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
  PersistentTensor table_handle_ GUARDED_BY(mu_);
  bool table_handle_set_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(TableWithFallbackConstructionOp);
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
      out->scalar<int64>()() = container->size();
    } else {
      out->scalar<int64>()() = container->size();
    }
  }
};

}  // namespace tables
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_TABLE_OP_UTILS_H_
