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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

ExecutionUnitDescriptionProto ExecutionUnitDescription::ToProto() const {
  ExecutionUnitDescriptionProto proto;
  for (const auto& [type, info] : rate_infos_) {
    auto& entry = (*proto.mutable_rate_infos())[type];
    entry.set_clock_rate_ghz(info.clock_rate_ghz);
    entry.set_units_per_core(info.units_per_core);
    entry.set_ops_per_clock(info.ops_per_clock);
  }
  return proto;
}

absl::StatusOr<ExecutionUnitDescription> ExecutionUnitDescription::FromProto(
    const ExecutionUnitDescriptionProto& proto) {
  ExecutionUnitDescription desc;
  for (const auto& [type_int, info_proto] : proto.rate_infos()) {
    if (!xla::PrimitiveType_IsValid(type_int)) {
      VLOG(2) << "Invalid PrimitiveType encountered, ExecutionUnitDescription "
                 "might be malformed: "
              << type_int;
      continue;
    }
    desc.SetRateInfo(
        static_cast<xla::PrimitiveType>(type_int),
        RateInfo{info_proto.units_per_core(), info_proto.clock_rate_ghz(),
                 info_proto.ops_per_clock()});
  }
  return desc;
}

absl::StatusOr<DeviceDescription> DeviceDescription::FromProto(
    const GpuDeviceInfoProto& proto) {
  DeviceDescription device_description;
  device_description.block_dim_limit_ =
      BlockDim(proto.block_dim_limit_x(), proto.block_dim_limit_y(),
               proto.block_dim_limit_z());
  device_description.threads_per_core_limit_ = proto.threads_per_core_limit();
  device_description.threads_per_block_limit_ = proto.threads_per_block_limit();
  device_description.threads_per_warp_ = proto.threads_per_warp();
  device_description.registers_per_core_limit_ =
      proto.registers_per_core_limit();
  device_description.registers_per_block_limit_ =
      proto.registers_per_block_limit();
  device_description.device_memory_size_ = proto.device_memory_size();
  device_description.l2_cache_size_ = proto.l2_cache_size();
  device_description.memory_bandwidth_ = proto.memory_bandwidth();
  device_description.shared_memory_per_core_ = proto.shared_memory_per_core();
  device_description.shared_memory_per_block_ = proto.shared_memory_per_block();
  device_description.shared_memory_per_block_optin_ =
      proto.shared_memory_per_block_optin();
  device_description.clock_rate_ghz_ = proto.clock_rate_ghz();

  if (proto.has_cuda_compute_capability()) {
    TF_ASSIGN_OR_RETURN(
        device_description.gpu_compute_capability_,
        CudaComputeCapability::FromProto(proto.cuda_compute_capability()));
  }
  if (proto.has_rocm_compute_capability()) {
    device_description.gpu_compute_capability_ =
        RocmComputeCapability(proto.rocm_compute_capability());
  }
  if (proto.has_oneapi_compute_capability()) {
    device_description.gpu_compute_capability_ =
        OneAPIComputeCapability(proto.oneapi_compute_capability());
  }
  device_description.core_count_ = proto.core_count();
  device_description.fpus_per_core_ = proto.fpus_per_core();

  if (proto.has_scalar_unit_description()) {
    TF_ASSIGN_OR_RETURN(
        device_description.scalar_unit_description_,
        ExecutionUnitDescription::FromProto(proto.scalar_unit_description()));
  }
  if (proto.has_matrix_unit_description()) {
    TF_ASSIGN_OR_RETURN(
        device_description.matrix_unit_description_,
        ExecutionUnitDescription::FromProto(proto.matrix_unit_description()));
  }

  return device_description;
}

GpuDeviceInfoProto DeviceDescription::ToGpuProto() const {
  stream_executor::GpuDeviceInfoProto proto;
  if (auto* ptr = gpu_compute_capability_.cuda_compute_capability()) {
    *proto.mutable_cuda_compute_capability() = ptr->ToProto();
  }
  if (auto* ptr = gpu_compute_capability_.rocm_compute_capability()) {
    *proto.mutable_rocm_compute_capability() = ptr->ToProto();
  }
  if (auto* ptr = gpu_compute_capability_.oneapi_compute_capability()) {
    *proto.mutable_oneapi_compute_capability() = ptr->ToProto();
  }

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
  if (scalar_unit_description_.has_value()) {
    *proto.mutable_scalar_unit_description() =
        scalar_unit_description_->ToProto();
  }
  if (matrix_unit_description_.has_value()) {
    *proto.mutable_matrix_unit_description() =
        matrix_unit_description_->ToProto();
  }
  return proto;
}

std::string DeviceDescription::ToString() const {
  return ToGpuProto().DebugString();
}

bool DeviceDescription::operator==(const DeviceDescription& other) const {
  return name_ == other.name_ && device_vendor_ == other.device_vendor_ &&
         platform_version_ == other.platform_version_ &&
         driver_version_ == other.driver_version_ &&
         runtime_version_ == other.runtime_version_ &&
         compile_time_toolkit_version_ == other.compile_time_toolkit_version_ &&
         dnn_version_ == other.dnn_version_ && model_str_ == other.model_str_ &&
         pci_bus_id_ == other.pci_bus_id_ && numa_node_ == other.numa_node_ &&
         core_count_ == other.core_count_ &&
         fpus_per_core_ == other.fpus_per_core_ &&
         thread_dim_limit_ == other.thread_dim_limit_ &&
         block_dim_limit_ == other.block_dim_limit_ &&
         threads_per_block_limit_ == other.threads_per_block_limit_ &&
         threads_per_core_limit_ == other.threads_per_core_limit_ &&
         threads_per_warp_ == other.threads_per_warp_ &&
         registers_per_core_limit_ == other.registers_per_core_limit_ &&
         registers_per_block_limit_ == other.registers_per_block_limit_ &&
         device_address_bits_ == other.device_address_bits_ &&
         device_memory_size_ == other.device_memory_size_ &&
         l2_cache_size_ == other.l2_cache_size_ &&
         memory_bandwidth_ == other.memory_bandwidth_ &&
         pcie_bandwidth_ == other.pcie_bandwidth_ &&
         clock_rate_ghz_ == other.clock_rate_ghz_ &&
         ecc_enabled_ == other.ecc_enabled_ &&
         gpu_compute_capability_ == other.gpu_compute_capability_ &&
         shared_memory_per_core_ == other.shared_memory_per_core_ &&
         shared_memory_per_block_ == other.shared_memory_per_block_ &&
         shared_memory_per_block_optin_ ==
             other.shared_memory_per_block_optin_ &&
         interconnect_info_ == other.interconnect_info_;
}

const GpuComputeCapability &DeviceDescription::gpu_compute_capability() const {
  return gpu_compute_capability_;
}

CudaComputeCapability DeviceDescription::cuda_compute_capability() const {
  if (auto* ptr = gpu_compute_capability_.cuda_compute_capability()) {
    return *ptr;
  }
  // Fallback for backwards compatibility.
  return CudaComputeCapability{-1, -1};
}

RocmComputeCapability DeviceDescription::rocm_compute_capability() const {
  if (auto* ptr = gpu_compute_capability_.rocm_compute_capability()) {
    return *ptr;
  }
  return RocmComputeCapability{};
}

OneAPIComputeCapability DeviceDescription::oneapi_compute_capability() const {
  if (auto* ptr = gpu_compute_capability_.oneapi_compute_capability()) {
    return *ptr;
  }
  return OneAPIComputeCapability{};
}

bool ThreadDimOk(const DeviceDescription& device_description,
                 const ThreadDim& thread_dim) {
  const int64_t total_threads = thread_dim.x * thread_dim.y * thread_dim.z;
  const int64_t threads_per_block_limit =
      device_description.threads_per_block_limit();
  if (total_threads > threads_per_block_limit) {
    VLOG(2) << "exceeded total-thread-per-block limit: " << total_threads
            << " vs limit " << threads_per_block_limit;
    return false;
  }

  const auto& limit = device_description.thread_dim_limit();
  bool ok = thread_dim.x <= limit.x && thread_dim.y <= limit.y &&
            thread_dim.z <= limit.z;
  if (!ok) {
    VLOG(2) << "thread dim " << thread_dim.ToString()
            << " exceeds limit constraints of " << limit.ToString();
  }
  return ok;
}

void CalculateDimensionality(const DeviceDescription& device_description,
                             int64_t element_count, int64_t* threads_per_block,
                             int64_t* block_count) {
  *threads_per_block = device_description.threads_per_block_limit();
  *block_count = tsl::MathUtil::CeilOfRatio(element_count, *threads_per_block);
  if (*block_count == 1) {
    CHECK_LE(element_count, *threads_per_block);
    *threads_per_block = element_count;
  }
}

GpuComputeCapabilityProto GpuComputeCapability::ToProto() const {
  GpuComputeCapabilityProto proto;
  if (IsCuda()) {
    *proto.mutable_cuda_compute_capability() =
        cuda_compute_capability()->ToProto();
  } else if (IsOneAPI()) {
    *proto.mutable_oneapi_compute_capability() =
        oneapi_compute_capability()->ToProto();
  } else {
    *proto.mutable_rocm_compute_capability() =
        rocm_compute_capability()->ToProto();
  }
  return proto;
}

absl::StatusOr<GpuComputeCapability> GpuComputeCapability::FromProto(
    const GpuComputeCapabilityProto& proto) {
  if (proto.has_cuda_compute_capability()) {
    TF_ASSIGN_OR_RETURN(
        CudaComputeCapability cuda_compute_capability,
        CudaComputeCapability::FromProto(proto.cuda_compute_capability()));
    return GpuComputeCapability(cuda_compute_capability);
  }

  if (proto.has_rocm_compute_capability()) {
    return GpuComputeCapability(
        RocmComputeCapability::FromProto(proto.rocm_compute_capability()));
  }

  if (proto.has_oneapi_compute_capability()) {
    return GpuComputeCapability(
        OneAPIComputeCapability::FromProto(proto.oneapi_compute_capability()));
  }

  return absl::InvalidArgumentError(
      "The serialized GpuComputeCapability has no compute capability set.");
}
}  // namespace stream_executor
