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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_DEVICE_INFO_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_DEVICE_INFO_H_

#include "tensorflow/compiler/xla/stream_executor/device_description.pb.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// The information contained in these structures is also contained in
// se::DeviceDescription, but separating these out lets us write code that does
// not depend on stream executor.
struct GpuDeviceInfo {
  int threads_per_block_limit;
  int threads_per_warp;
  int shared_memory_per_block;
  int shared_memory_per_core;
  int threads_per_core_limit;
  int core_count;
  int64_t fpus_per_core;
  int block_dim_limit_x;
  int block_dim_limit_y;
  int block_dim_limit_z;
  int64_t memory_bandwidth;
  int64_t l2_cache_size;
  float clock_rate_ghz;
  int64_t device_memory_size;

  stream_executor::GpuDeviceInfoProto ToProto() const {
    stream_executor::GpuDeviceInfoProto proto;
    proto.set_threads_per_block_limit(threads_per_block_limit);
    proto.set_threads_per_warp(threads_per_warp);
    proto.set_shared_memory_per_block(shared_memory_per_block);
    proto.set_shared_memory_per_core(shared_memory_per_core);
    proto.set_threads_per_core_limit(threads_per_core_limit);
    proto.set_core_count(core_count);
    proto.set_fpus_per_core(fpus_per_core);
    proto.set_block_dim_limit_x(block_dim_limit_x);
    proto.set_block_dim_limit_y(block_dim_limit_y);
    proto.set_block_dim_limit_z(block_dim_limit_z);
    proto.set_memory_bandwidth(memory_bandwidth);
    proto.set_l2_cache_size(l2_cache_size);
    proto.set_clock_rate_ghz(clock_rate_ghz);
    proto.set_device_memory_size(device_memory_size);
    return proto;
  }

  GpuDeviceInfo() = default;
  explicit GpuDeviceInfo(const stream_executor::GpuDeviceInfoProto& proto) {
    threads_per_block_limit = proto.threads_per_block_limit();
    threads_per_warp = proto.threads_per_warp();
    shared_memory_per_block = proto.shared_memory_per_block();
    shared_memory_per_core = proto.shared_memory_per_core();
    threads_per_core_limit = proto.threads_per_core_limit();
    core_count = proto.core_count();
    fpus_per_core = proto.fpus_per_core();
    block_dim_limit_x = proto.block_dim_limit_x();
    block_dim_limit_y = proto.block_dim_limit_y();
    block_dim_limit_z = proto.block_dim_limit_z();
    memory_bandwidth = proto.memory_bandwidth();
    l2_cache_size = proto.l2_cache_size();
    clock_rate_ghz = proto.clock_rate_ghz();
    device_memory_size = proto.device_memory_size();
  }
};

GpuDeviceInfo GetGpuDeviceInfo(
    const stream_executor::StreamExecutor* stream_exec);
GpuDeviceInfo GetGpuDeviceInfo(const stream_executor::Platform* platform);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_DEVICE_INFO_H_
