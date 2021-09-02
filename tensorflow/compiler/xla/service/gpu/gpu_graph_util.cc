/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_graph_util.h"

namespace xla {
namespace gpu {

bool MutexedGraphExecCache::AddToCache(BufferAllocations::KeyType key,
                                       void* gpu_exec_graph) {
  tensorflow::mutex_lock lock(exec_graph_cache_mu_);
  bool has_max_cache_size_reached = false;
  gpu_exec_graphs_.push_front(gpu_exec_graph);
  if (gpu_exec_graphs_.size() > cache_size_.load()) {
    has_max_cache_size_reached = true;
    auto* exec_graph =
        reinterpret_cast<stream_executor::gpu::GpuGraphExecHandle*>(
            &gpu_exec_graphs_.back());
    using stream_executor::gpu::GpuDriver;
    GpuDriver::DestroyExecutableGraph(gpu_context_, exec_graph);
    gpu_exec_graphs_.pop_back();
  }
  gpu_key_to_exec_graphs_map_[key] = gpu_exec_graphs_.begin();
  return has_max_cache_size_reached;
}

void* MutexedGraphExecCache::GetExecGraph(BufferAllocations::KeyType key) {
  tensorflow::mutex_lock lock(exec_graph_cache_mu_);
  if (gpu_key_to_exec_graphs_map_.find(key) !=
      gpu_key_to_exec_graphs_map_.end()) {
    auto it = std::find(gpu_exec_graphs_.begin(), gpu_exec_graphs_.end(),
                        *(gpu_key_to_exec_graphs_map_[key]));
    if (it == gpu_exec_graphs_.end()) {
      gpu_key_to_exec_graphs_map_.erase(key);
      return nullptr;
    }
    auto gpu_exec_graph = *(gpu_key_to_exec_graphs_map_[key]);
    gpu_exec_graphs_.remove(gpu_exec_graph);
    gpu_exec_graphs_.push_front(gpu_exec_graph);
    gpu_key_to_exec_graphs_map_[key] = gpu_exec_graphs_.begin();
    return gpu_exec_graph;
  }
  return nullptr;
}

void MutexedGraphExecCache::SetCacheSize(int64 cache_size) {
  cache_size_.store(cache_size);
}

void MutexedGraphExecCache::SetGpuContext(
    stream_executor::gpu::GpuContext* gpu_context) {
  tensorflow::mutex_lock lock(exec_graph_cache_mu_);
  gpu_context_ = gpu_context;
}

size_t MutexedGraphExecCache::GetCurrentCacheSize() {
  tensorflow::mutex_lock lock(exec_graph_cache_mu_);
  return gpu_exec_graphs_.size();
}

void MutexedGraphExecCache::Initialize(
    stream_executor::gpu::GpuContext* gpu_context) {
  if (!is_initialized_) {
    is_initialized_ = true;
    SetGpuContext(gpu_context);
    SetCacheSize(GpuExecGraphCacheSize());
  }
}

}  // namespace gpu
}  // namespace xla