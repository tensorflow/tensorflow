/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_SYMMETRIC_MEMORY_RESOURCES_H_
#define XLA_STREAM_EXECUTOR_GPU_SYMMETRIC_MEMORY_RESOURCES_H_

#include <memory>

#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
class CollectiveMemoryCache;
}  // namespace xla::gpu

namespace stream_executor::gpu {

class SymmetricMemoryResources : public StreamExecutor::Resource {
 public:
  SymmetricMemoryResources();
  ~SymmetricMemoryResources() override;

  xla::gpu::CollectiveMemoryCache* collective_memory_cache() const {
    return collective_memory_cache_.get();
  }

 private:
  std::unique_ptr<xla::gpu::CollectiveMemoryCache> collective_memory_cache_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_SYMMETRIC_MEMORY_RESOURCES_H_
