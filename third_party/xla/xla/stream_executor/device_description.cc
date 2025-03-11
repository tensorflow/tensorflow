/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/device_description.h"

#include <cstdint>
#include <string>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/lib/math/math_util.h"
#include "tsl/platform/logging.h"

namespace stream_executor {

DeviceDescription::DeviceDescription(const GpuDeviceInfoProto &proto)
    : block_dim_limit_(BlockDim(proto.block_dim_limit_x(),
                                proto.block_dim_limit_y(),
                                proto.block_dim_limit_z())),
      threads_per_core_limit_(proto.threads_per_core_limit()),
      threads_per_block_limit_(proto.threads_per_block_limit()),
      threads_per_warp_(proto.threads_per_warp()),
      registers_per_core_limit_(proto.registers_per_core_limit()),
      registers_per_block_limit_(proto.registers_per_block_limit()),
      device_memory_size_(proto.device_memory_size()),
      l2_cache_size_(proto.l2_cache_size()),
      memory_bandwidth_(proto.memory_bandwidth()),
      shared_memory_per_core_(proto.shared_memory_per_core()),
      shared_memory_per_block_(proto.shared_memory_per_block()),
      shared_memory_per_block_optin_(proto.shared_memory_per_block_optin()),
      clock_rate_ghz_(proto.clock_rate_ghz()),
      gpu_compute_capability_(
          proto.has_cuda_compute_capability()
              ? GpuComputeCapability(stream_executor::CudaComputeCapability(
                    proto.cuda_compute_capability()))
              : GpuComputeCapability(stream_executor::RocmComputeCapability(
                    proto.rocm_compute_capability()))),
      core_count_(proto.core_count()),
      fpus_per_core_(proto.fpus_per_core()) {}

GpuDeviceInfoProto DeviceDescription::ToGpuProto() const {
  stream_executor::GpuDeviceInfoProto proto;
  if (auto *ptr = std::get_if<stream_executor::CudaComputeCapability>(
          &gpu_compute_capability_))
    *proto.mutable_cuda_compute_capability() = ptr->ToProto();
  if (auto *ptr = std::get_if<stream_executor::RocmComputeCapability>(
          &gpu_compute_capability_))
    *proto.mutable_rocm_compute_capability() = ptr->ToProto();

  proto.set_threads_per_block_limit(threads_per_block_limit_);
  proto.set_threads_per_warp(threads_per_warp_);
  proto.set_shared_memory_per_block(shared_memory_per_block_);
  proto.set_shared_memory_per_block_optin(shared_memory_per_block_optin_);
  proto.set_shared_memory_per_core(shared_memory_per_core_);
  proto.set_threads_per_core_limit(threads_per_core_limit_);
  proto.set_core_count(core_count_);
  proto.set_fpus_per_core(fpus_per_core_);
  proto.set_block_dim_limit_x(block_dim_limit().x);
  proto.set_block_dim_limit_y(block_dim_limit().y);
  proto.set_block_dim_limit_z(block_dim_limit().z);
  proto.set_memory_bandwidth(memory_bandwidth_);
  proto.set_l2_cache_size(l2_cache_size_);
  proto.set_clock_rate_ghz(clock_rate_ghz_);
  proto.set_device_memory_size(device_memory_size_);
  proto.set_registers_per_core_limit(registers_per_core_limit_);
  proto.set_registers_per_block_limit(registers_per_block_limit_);
  return proto;
}

std::string DeviceDescription::ToString() const {
  return ToGpuProto().DebugString();
}

const GpuComputeCapability &DeviceDescription::gpu_compute_capability() const {
  return gpu_compute_capability_;
}

CudaComputeCapability DeviceDescription::cuda_compute_capability() const {
  if (auto *ptr =
          std::get_if<CudaComputeCapability>(&gpu_compute_capability_)) {
    return *ptr;
  }
  // Fallback for backwards compatibility.
  return CudaComputeCapability{-1, -1};
}

RocmComputeCapability DeviceDescription::rocm_compute_capability() const {
  if (auto *ptr =
          std::get_if<RocmComputeCapability>(&gpu_compute_capability_)) {
    return *ptr;
  }
  return RocmComputeCapability{};
}

bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim) {
  const int64_t total_threads = thread_dim.x * thread_dim.y * thread_dim.z;
  const int64_t threads_per_block_limit =
      device_description.threads_per_block_limit();
  if (total_threads > threads_per_block_limit) {
    VLOG(2) << "exceeded total-thread-per-block limit: " << total_threads
            << " vs limit " << threads_per_block_limit;
    return false;
  }

  const auto &limit = device_description.thread_dim_limit();
  bool ok = thread_dim.x <= limit.x && thread_dim.y <= limit.y &&
            thread_dim.z <= limit.z;
  if (!ok) {
    VLOG(2) << "thread dim " << thread_dim.ToString()
            << " exceeds limit constraints of " << limit.ToString();
  }
  return ok;
}

void CalculateDimensionality(const DeviceDescription &device_description,
                             int64_t element_count, int64_t *threads_per_block,
                             int64_t *block_count) {
  *threads_per_block = device_description.threads_per_block_limit();
  *block_count = tsl::MathUtil::CeilOfRatio(element_count, *threads_per_block);
  if (*block_count == 1) {
    CHECK_LE(element_count, *threads_per_block);
    *threads_per_block = element_count;
  }
}

}  // namespace stream_executor
