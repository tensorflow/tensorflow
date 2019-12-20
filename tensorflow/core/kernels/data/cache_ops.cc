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
#include "tensorflow/core/kernels/data/cache_ops.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {
namespace {

const char kMemoryCache[] = "MemoryCache";

}  // namespace

string MemoryCache::DebugString() const { return kMemoryCache; }

void MemoryCache::Complete(std::vector<std::vector<Tensor>>&& cache) {
  mutex_lock l(mu_);
  if (!completed_) {
    cache_ = std::move(cache);
    completed_ = true;
  }
}

bool MemoryCache::IsCompleted() {
  tf_shared_lock l(mu_);
  return completed_;
}

void MemoryCache::Reset() {
  mutex_lock l(mu_);
  completed_ = false;
  cache_.clear();
}

const std::vector<Tensor>& MemoryCache::at(int64 index) {
  tf_shared_lock l(mu_);
  DCHECK(index < cache_.size());
  return cache_[index];
}

size_t MemoryCache::size() {
  tf_shared_lock l(mu_);
  return cache_.size();
}

AnonymousMemoryCacheHandleOp::AnonymousMemoryCacheHandleOp(
    OpKernelConstruction* ctx)
    : AnonymousResourceOp<MemoryCache>(ctx) {}

void AnonymousMemoryCacheHandleOp::Compute(OpKernelContext* ctx) {
  AnonymousResourceOp<MemoryCache>::Compute(ctx);
}

string AnonymousMemoryCacheHandleOp::name() { return kMemoryCache; }

Status AnonymousMemoryCacheHandleOp::CreateResource(
    OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, MemoryCache** resource) {
  *resource = new MemoryCache();
  return Status::OK();
}

void DeleteMemoryCacheOp::Compute(OpKernelContext* ctx) {
  const ResourceHandle& handle = ctx->input(0).flat<ResourceHandle>()(0);
  // The resource is guaranteed to exist because the variant tensor wrapping the
  // deleter is provided as an unused input to this op, which guarantees that it
  // has not run yet.
  OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete(handle));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("AnonymousMemoryCache").Device(DEVICE_CPU),
                        AnonymousMemoryCacheHandleOp);

REGISTER_KERNEL_BUILDER(Name("DeleteMemoryCache").Device(DEVICE_CPU),
                        DeleteMemoryCacheOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
