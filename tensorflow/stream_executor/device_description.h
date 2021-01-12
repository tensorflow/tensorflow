/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Describes the underlying platform for a StreamExecutor; e.g. OpenCL or CUDA
// device and platform properties. Also contains convenience functions for
// checking/calculating launch dimensionality based on device properties.

#ifndef TENSORFLOW_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
#define TENSORFLOW_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_

#include <map>
#include <memory>
#include "absl/base/macros.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {
namespace internal {
class DeviceDescriptionBuilder;
}  // namespace internal

// Data that describes the execution target of the StreamExecutor, in terms of
// important logical parameters. These include dimensionality limits and
// physical parameters of interest, such as number of cores present on the
// device.
//
// Thread-safe: immutable post-initialization.
class DeviceDescription {
 public:
  // Returns the platform being run on; this value is primarily intended for
  // printing, and comes out something like "OpenCL 1.2" or "Compute Capability
  // 3.5".
  const std::string &platform_version() const { return platform_version_; }

  // Returns the driver version interfacing with the underlying platform. Vendor
  // dependent format.
  const std::string &driver_version() const { return driver_version_; }

  // Return the runtime version, if one is provided by the underlying platform.
  // Vendor dependent format / usefulness.
  const std::string &runtime_version() const { return runtime_version_; }

  // Returns the name that the device reports. Vendor dependent.
  const std::string &name() const { return name_; }

  // Returns the PCI bus identifier for this device, of the form
  // [domain]:[bus]:[device].[function]
  const std::string &pci_bus_id() const { return pci_bus_id_; }

  // Returns the NUMA node associated with this device, for use in
  // determining socket locality. If the NUMA node could not be determined, -1
  // is returned.
  int numa_node() const { return numa_node_; }

  // Number of cores (traditional notion of core; i.e. an SM on an NVIDIA device
  // or an AMD Compute Unit.
  int core_count() const { return core_count_; }

  // Returns the limit on the thread dimensionality values in each of the
  // respective dimensions. These limits affect what constitutes a legitimate
  // kernel launch request.
  const ThreadDim &thread_dim_limit() const { return thread_dim_limit_; }

  // Returns the limit on the block dimensionality values in each of the
  // respective dimensions. These limits may affect what constitutes a
  // legitimate kernel launch request.
  const BlockDim &block_dim_limit() const { return block_dim_limit_; }

  // Returns the limit on the total number of threads that can be launched in a
  // single block; i.e. the limit on x * y * z dimensions of a ThreadDim.
  // This limit affects what constitutes a legitimate kernel launch request.
  const int64 &threads_per_block_limit() const {
    return threads_per_block_limit_;
  }

  // Returns the limit on the total number of threads that can be simultaneously
  // launched on a given multiprocessor.
  const int64 &threads_per_core_limit() const {
    return threads_per_core_limit_;
  }

  // Returns the number of threads per warp/wavefront.
  const int64 &threads_per_warp() const { return threads_per_warp_; }

  // Returns the limit on the total number of registers per core.
  const int64 &registers_per_core_limit() const {
    return registers_per_core_limit_;
  }

  // Returns the limit on the total number of registers that can be
  // simultaneously used by a block.
  const int64 &registers_per_block_limit() const {
    return registers_per_block_limit_;
  }

  // Returns the number of address bits available to kernel code running on the
  // platform. This affects things like the maximum allocation size and perhaps
  // types used in kernel code such as size_t.
  const int64 &device_address_bits() const { return device_address_bits_; }

  // Returns the device memory size in bytes.
  int64 device_memory_size() const { return device_memory_size_; }

  // Returns the device's memory bandwidth in bytes/sec.  (This is for
  // reads/writes to/from the device's own memory, not for transfers between the
  // host and device.)
  int64 memory_bandwidth() const { return memory_bandwidth_; }

  // Returns the device's core clock rate in GHz.
  float clock_rate_ghz() const { return clock_rate_ghz_; }

  // Returns whether ECC is enabled.
  bool ecc_enabled() const { return ecc_enabled_; }

  // Returns the device vendor string, e.g., "NVIDIA Corporation", "Advanced
  // Micro Devices, Inc.", or "GenuineIntel".
  const std::string &device_vendor() const { return device_vendor_; }

  // Returns the CUDA compute capability if we're running on the CUDA platform.
  // If a CUDA compute capability is not available, the major version will be
  // zero, and the return value will be false.
  bool cuda_compute_capability(int *major, int *minor) const;

  // Returns the AMDGPU ISA version if we're running on the ROCm platform.
  // If the information is not available, the version is not modified,
  // and the return value will be false.
  bool rocm_amdgpu_isa_version(int *version) const;

  // Returns the
  // * AMDGPU GCN Architecture Name if we're running on the ROCm platform.
  // * kUndefinedString otherwise
  const std::string rocm_amdgpu_gcn_arch_name() const {
    return rocm_amdgpu_gcn_arch_name_;
  }

  // Returns the maximum amount of shared memory present on a single core
  // (i.e. Streaming Multiprocessor on NVIDIA GPUs; Compute Unit for OpenCL
  // devices). Note that some devices, such as NVIDIA's have a configurable
  // partitioning between shared memory and L1 cache.
  int64 shared_memory_per_core() const { return shared_memory_per_core_; }

  // Returns the maximum amount of shared memory available for a single block.
  int64 shared_memory_per_block() const { return shared_memory_per_block_; }

  // TODO(leary): resident blocks per core will be useful.

  // Convenience typedef for the string-based DeviceDescription mapping.
  typedef std::map<std::string, std::string> Map;

  // Returns a mapping from readable names to readable values that describe the
  // device. This is useful for things like printing.
  std::unique_ptr<Map> ToMap() const;

  // For string values that are not available via the underlying platform, this
  // value will be provided.
  static const char *kUndefinedString;

 private:
  friend class internal::DeviceDescriptionBuilder;

  DeviceDescription();

  // For description of the following members, see the corresponding accessor
  // above.
  //
  // N.B. If another field is added, update ToMap() above.
  std::string device_vendor_;
  std::string platform_version_;
  std::string driver_version_;
  std::string runtime_version_;
  std::string pci_bus_id_;
  std::string name_;

  ThreadDim thread_dim_limit_;
  BlockDim block_dim_limit_;

  int64 threads_per_core_limit_;
  int64 threads_per_block_limit_;
  int64 threads_per_warp_;

  int64 registers_per_core_limit_;
  int64 registers_per_block_limit_;

  int64 device_address_bits_;
  int64 device_memory_size_;
  int64 memory_bandwidth_;

  // Shared memory limits on a given device.
  int64 shared_memory_per_core_;
  int64 shared_memory_per_block_;

  float clock_rate_ghz_;

  // CUDA "CC" major value, -1 if not available.
  int cuda_compute_capability_major_;
  int cuda_compute_capability_minor_;

  // ROCM AMDGPU ISA version, 0 if not available.
  int rocm_amdgpu_isa_version_;

  // ROCm AMDGPU GCN Architecture name, "" if not available.
  std::string rocm_amdgpu_gcn_arch_name_;

  int numa_node_;
  int core_count_;
  bool ecc_enabled_;

  SE_DISALLOW_COPY_AND_ASSIGN(DeviceDescription);
};

namespace internal {

// Helper class the builds a device description, given that it has a large
// number of fields that would be easily confused in constructor form.
class DeviceDescriptionBuilder {
 public:
  DeviceDescriptionBuilder();

  // For descriptions of the following fields, see comments on the corresponding
  // DeviceDescription::* accessors above.

  void set_device_vendor(const std::string &value) {
    device_description_->device_vendor_ = value;
  }
  void set_platform_version(const std::string &value) {
    device_description_->platform_version_ = value;
  }
  void set_driver_version(const std::string &value) {
    device_description_->driver_version_ = value;
  }
  void set_runtime_version(const std::string &value) {
    device_description_->runtime_version_ = value;
  }
  void set_pci_bus_id(const std::string &value) {
    device_description_->pci_bus_id_ = value;
  }
  void set_name(const std::string &value) {
    device_description_->name_ = value;
  }

  void set_thread_dim_limit(const ThreadDim &value) {
    device_description_->thread_dim_limit_ = value;
  }
  void set_block_dim_limit(const BlockDim &value) {
    device_description_->block_dim_limit_ = value;
  }

  void set_threads_per_core_limit(int64 value) {
    device_description_->threads_per_core_limit_ = value;
  }
  void set_threads_per_block_limit(int64 value) {
    device_description_->threads_per_block_limit_ = value;
  }
  void set_threads_per_warp(int64 value) {
    device_description_->threads_per_warp_ = value;
  }

  void set_registers_per_core_limit(int64 value) {
    device_description_->registers_per_core_limit_ = value;
  }
  void set_registers_per_block_limit(int64 value) {
    device_description_->registers_per_block_limit_ = value;
  }

  void set_device_address_bits(int64 value) {
    device_description_->device_address_bits_ = value;
  }
  void set_device_memory_size(int64 value) {
    device_description_->device_memory_size_ = value;
  }
  void set_memory_bandwidth(int64 value) {
    device_description_->memory_bandwidth_ = value;
  }

  void set_shared_memory_per_core(int64 value) {
    device_description_->shared_memory_per_core_ = value;
  }
  void set_shared_memory_per_block(int64 value) {
    device_description_->shared_memory_per_block_ = value;
  }

  void set_clock_rate_ghz(float value) {
    device_description_->clock_rate_ghz_ = value;
  }

  void set_cuda_compute_capability(int major, int minor) {
    device_description_->cuda_compute_capability_major_ = major;
    device_description_->cuda_compute_capability_minor_ = minor;
  }

  void set_rocm_amdgpu_isa_version(int version) {
    device_description_->rocm_amdgpu_isa_version_ = version;
  }

  void set_rocm_amdgpu_gcn_arch_name(const std::string &gcn_arch_name) {
    device_description_->rocm_amdgpu_gcn_arch_name_ = gcn_arch_name;
  }

  void set_numa_node(int value) { device_description_->numa_node_ = value; }
  void set_core_count(int value) { device_description_->core_count_ = value; }
  void set_ecc_enabled(bool value) {
    device_description_->ecc_enabled_ = value;
  }

  // Returns a built DeviceDescription with ownership transferred to the
  // caller. There are currently no restrictions on which fields must be set in
  // order to build the descriptor.
  //
  // Once the description is built, this builder object should be discarded.
  std::unique_ptr<DeviceDescription> Build() {
    return std::move(device_description_);
  }

 private:
  std::unique_ptr<DeviceDescription> device_description_;

  SE_DISALLOW_COPY_AND_ASSIGN(DeviceDescriptionBuilder);
};

}  // namespace internal

// Returns whether the given thread_dim is acceptable given the limits described
// in device_description. For detailed reasons for failing the predicate, enable
// VLOG(2) for this module.
bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim);

// Equivalent to ceil(double(element_count) / threads_per_block).
ABSL_DEPRECATED("Use MathUtil::CeilOfRatio directly instead.")
int64 DivideCeil(int64 x, int64 y);

// Calculate the number of threads/blocks required to process element_count
// elements. Note that you can still end up with more threads than
// element_count due to rounding, so kernels often start with an "is this
// thread id in the element_count range?" test.
void CalculateDimensionality(const DeviceDescription &device_description,
                             int64 element_count, int64 *threads_per_block,
                             int64 *block_count);

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
