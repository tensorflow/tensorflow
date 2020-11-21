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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_CACHE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_CACHE_OPS_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"

namespace tensorflow {
namespace data {

// A thread-safe data structure for caching dataset elements.
//
// The expected use is that a single `MemoryWriterIterator` populates the
// cache with dataset elements. Once all elements are cached, the cache can
// be used by one or more `MemoryReaderIterator`s.
class MemoryCache {
 public:
  MemoryCache() = default;

  // Marks the cache as completed.
  void Complete(std::vector<std::vector<Tensor>>&& cache);

  // Returns whether the cache is completed.
  bool IsCompleted();

  // Resets the cache.
  void Reset();

  // Returns the element at the given index.
  const std::vector<Tensor>& at(int64 index);

  // Returns the size of the cache.
  size_t size();

  // Returns a reference to the cache's data. The returned reference will be
  // invalidated by any call to Reset().
  const std::vector<std::vector<Tensor>>& data();

 private:
  mutex mu_;
  // Determines whether all elements of the dataset have been cached.
  bool completed_ TF_GUARDED_BY(mu_) = false;
  std::vector<std::vector<Tensor>> cache_ TF_GUARDED_BY(mu_);
};

// A resource wrapping a shared instance of a memory cache.
class MemoryCacheManager : public ResourceBase {
 public:
  MemoryCacheManager() : cache_(std::make_shared<MemoryCache>()) {}

  string DebugString() const override;

  std::shared_ptr<MemoryCache> get() { return cache_; }

 private:
  std::shared_ptr<MemoryCache> cache_;
};

// Creates an instance of cache resource and transfers ownership to the caller.
class AnonymousMemoryCacheHandleOp
    : public AnonymousResourceOp<MemoryCacheManager> {
 public:
  explicit AnonymousMemoryCacheHandleOp(OpKernelConstruction* ctx);

 private:
  string name() override;
  Status CreateResource(OpKernelContext* ctx,
                        std::unique_ptr<FunctionLibraryDefinition> flib_def,
                        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                        FunctionLibraryRuntime* lib,
                        MemoryCacheManager** manager) override;
};

// Deletes an instance of cache resource.
class DeleteMemoryCacheOp : public OpKernel {
 public:
  explicit DeleteMemoryCacheOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_CACHE_OPS_H_
