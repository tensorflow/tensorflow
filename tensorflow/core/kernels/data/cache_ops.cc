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

#include "tensorflow/core/data/dataset_utils.h"
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

constexpr char kMemoryCache[] = "MemoryCache";

}  // namespace

string MemoryCacheManager::DebugString() const { return kMemoryCache; }

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

const std::vector<Tensor>& MemoryCache::at(int64_t index) {
  tf_shared_lock l(mu_);
  DCHECK(index < cache_.size());
  return cache_[index];
}

size_t MemoryCache::size() {
  tf_shared_lock l(mu_);
  return cache_.size();
}

const std::vector<std::vector<Tensor>>& MemoryCache::data() {
  tf_shared_lock l(mu_);
  return cache_;
}

AnonymousMemoryCacheHandleOp::AnonymousMemoryCacheHandleOp(
    OpKernelConstruction* ctx)
    : AnonymousResourceOp<MemoryCacheManager>(ctx,
                                              /* ref_counting */ true,
                                              /* return_deleter */ true) {}

string AnonymousMemoryCacheHandleOp::name() { return kMemoryCache; }

Status AnonymousMemoryCacheHandleOp::CreateResource(
    OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, MemoryCacheManager** manager) {
  *manager = new MemoryCacheManager();
  return OkStatus();
}

void DeleteMemoryCacheOp::Compute(OpKernelContext* ctx) {
  const ResourceHandle& handle = ctx->input(0).flat<ResourceHandle>()(0);
  // The resource might have been already deleted by the dataset.
  Status s = ctx->resource_manager()->Delete(handle);
  if (!errors::IsNotFound(s)) {
    OP_REQUIRES_OK(ctx, s);
  }
}

namespace {

REGISTER_KERNEL_BUILDER(Name("AnonymousMemoryCache").Device(DEVICE_CPU),
                        AnonymousMemoryCacheHandleOp);

REGISTER_KERNEL_BUILDER(Name("DeleteMemoryCache").Device(DEVICE_CPU),
                        DeleteMemoryCacheOp);

REGISTER_KERNEL_BUILDER(Name("DummyMemoryCache").Device(DEVICE_CPU),
                        DummyResourceOp<MemoryCache>);

}  // namespace
}  // namespace data
}  // namespace tensorflow
