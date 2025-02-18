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

// Describes the underlying platform for a StreamExecutor; e.g. OpenCL or CUDA
// device and platform properties. Also contains convenience functions for
// checking/calculating launch dimensionality based on device properties.

#ifndef XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
#define XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {

// ROCm compute capability, as reported by the device description.
class RocmComputeCapability {
 public:
  // gcn_arch_name example --  gfx90a:sramecc+:xnack-
  // gfx_version is the "gfx90a" part of the gcn_arch_name
  explicit RocmComputeCapability(std::string gcn_arch_name)
      : gcn_arch_name_(std::move(gcn_arch_name)) {}

  explicit RocmComputeCapability(const RocmComputeCapabilityProto &proto)
      : gcn_arch_name_(proto.gcn_arch_name()) {}

  RocmComputeCapability() = default;

  std::string gcn_arch_name() const { return gcn_arch_name_; }

  std::string gfx_version() const {
    std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name_, ':');
    return tokens[0];
  }

  bool is_supported_gfx_version() const {
    return absl::c_count(kSupportedGfxVersions, gfx_version()) != 0;
  }

  std::string supported_gfx_versions_str() const {
    return absl::StrJoin(kSupportedGfxVersions, ", ");
  }

  bool gfx9_mi100() const { return gfx_version() == "gfx908"; }

  bool gfx9_mi200() const { return gfx_version() == "gfx90a"; }

  bool gfx9_mi300() const {
    static constexpr absl::string_view kList[] = {"gfx940", "gfx941", "gfx942"};
    return absl::c_count(kList, gfx_version()) != 0;
  }

  bool gfx9_mi100_or_later() const {
    static constexpr absl::string_view kList[] = {"gfx908", "gfx90a", "gfx940",
                                                  "gfx941", "gfx942"};
    return absl::c_count(kList, gfx_version()) != 0;
  }

  bool gfx9_mi200_or_later() const {
    static constexpr absl::string_view kList[] = {"gfx90a", "gfx940", "gfx941",
                                                  "gfx942"};
    return absl::c_count(kList, gfx_version()) != 0;
  }

  bool gfx10_rx68xx() const { return gfx_version() == "gfx1030"; }

  bool gfx10_rx69xx() const { return gfx_version() == "gfx1030"; }

  bool gfx11_rx7900() const { return gfx_version() == "gfx1100"; }

  bool gfx1200() const { return gfx_version() == "gfx1200"; }

  bool gfx1201() const { return gfx_version() == "gfx1201"; }

  bool has_nhwc_layout_support() const { return gfx9_mi100_or_later(); }

  bool has_bf16_dtype_support() const { return gfx9_mi100_or_later(); }

  bool has_fast_fp16_support() const {
    return gfx9_mi100_or_later() || gfx10_rx68xx() || gfx10_rx69xx() ||
           gfx11_rx7900();
  }

  bool has_mfma_instr_support() const { return gfx9_mi100_or_later(); }

  bool has_amd_matrix_core() const {
    return (gfx9_mi100_or_later() || gfx_version().find("gfx11") ||
            gfx_version().find("gfx12"));
  }

  bool has_fp16_atomics_support() const {
    // TODO(rocm): Check. This should be the same as has_fast_fp16_support().
    return gfx9_mi200_or_later();
  }

  bool fence_before_barrier() const {
    return gfx_version() != "gfx900" && gfx_version() != "gfx906";
  }

  bool has_hipblaslt() const {
    return gfx9_mi200_or_later() || gfx1200() || gfx1201();
  }

  bool has_fp8_support() const {
    return gfx9_mi300() || gfx1200() || gfx1201();
  }

  std::string ToString() const { return gcn_arch_name(); }

  RocmComputeCapabilityProto ToProto() const {
    RocmComputeCapabilityProto proto;
    proto.set_gcn_arch_name(gcn_arch_name_);
    return proto;
  }

  bool operator==(const RocmComputeCapability &other) const {
    return gcn_arch_name_ == other.gcn_arch_name_;
  }

 private:
  std::string gcn_arch_name_ = "gfx000";  // default to invalid arch.

  static constexpr absl::string_view kSupportedGfxVersions[]{
      "gfx900",                        // MI25
      "gfx906",                        // MI50 / MI60
      "gfx908",                        // MI100
      "gfx90a",                        // MI200
      "gfx940",  "gfx941",  "gfx942",  // MI300
      "gfx1030",                       // RX68xx / RX69xx
      "gfx1100",                       // RX7900
      "gfx1200", "gfx1201",
  };
};

using GpuComputeCapability =
    std::variant<CudaComputeCapability, RocmComputeCapability>;

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

  // Returns the driver version interfacing with the underlying platform.
  // Note for CUDA this returns the CUDA Toolkit version the driver ships with.
  SemanticVersion driver_version() const { return driver_version_; }

  // Returns the runtime version.
  SemanticVersion runtime_version() const { return runtime_version_; }

  // Returns the toolkit version that the application was compiled against.
  SemanticVersion compile_time_toolkit_version() const {
    return compile_time_toolkit_version_;
  }

  // Returns the name that the device reports. Vendor dependent.
  const std::string &name() const { return name_; }

  // Gets a human-readable description of the device, e.g. "nvidia GPU
  // supporting sm75 with 32GB RAM, 80 SMs, ...".  This is intended to be the
  // same if and only if two devices are "the same" (e.g. the same make/model of
  // GPU), though it may not completely succeed at this for all platforms.
  //
  // This string is not guaranteed to be stable between versions.  Please DO NOT
  // rely on it never changing.  (Within one version of the code, it won't
  // change, don't worry.)
  const std::string &model_str() const { return model_str_; }

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

  // Number of floating point operations one core (SM, compute unit) can execute
  // in parallel. Corresponds to the number of "CUDA cores" for NVIDIA devices.
  int fpus_per_core() const { return fpus_per_core_; }

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
  const int64_t &threads_per_block_limit() const {
    return threads_per_block_limit_;
  }

  // Returns the limit on the total number of threads that can be simultaneously
  // launched on a given multiprocessor.
  const int64_t &threads_per_core_limit() const {
    return threads_per_core_limit_;
  }

  // Returns the number of threads per warp/wavefront.
  constexpr int64_t threads_per_warp() const { return threads_per_warp_; }

  // Returns the limit on the total number of registers per core.
  const int64_t &registers_per_core_limit() const {
    return registers_per_core_limit_;
  }

  // Returns the limit on the total number of registers that can be
  // simultaneously used by a block.
  const int64_t &registers_per_block_limit() const {
    return registers_per_block_limit_;
  }

  // Returns the number of address bits available to kernel code running on the
  // platform. This affects things like the maximum allocation size and perhaps
  // types used in kernel code such as size_t.
  const int64_t &device_address_bits() const { return device_address_bits_; }

  // Returns the device memory size in bytes.
  int64_t device_memory_size() const { return device_memory_size_; }

  // Returns the L2 cache size in bytes.
  int64_t l2_cache_size() const { return l2_cache_size_; }

  // Returns the device's memory bandwidth in bytes/sec.  (This is for
  // reads/writes to/from the device's own memory, not for transfers between the
  // host and device.)
  int64_t memory_bandwidth() const { return memory_bandwidth_; }

  // Returns the device's core clock rate in GHz.
  float clock_rate_ghz() const { return clock_rate_ghz_; }

  // Returns whether ECC is enabled.
  bool ecc_enabled() const { return ecc_enabled_; }

  // Returns the device vendor string, e.g., "NVIDIA Corporation", "Advanced
  // Micro Devices, Inc.", or "GenuineIntel".
  const std::string &device_vendor() const { return device_vendor_; }

  // Returns the CUDA compute capability if we're running on the CUDA platform.
  // If a CUDA compute capability is not available, the major version will be
  // zero.
  CudaComputeCapability cuda_compute_capability() const;

  // Returns the ROCm compute capability if we're running on the ROCm platform.
  // If a ROCm compute capability is not available, the default gfx_arch will
  // be "gfx000" (which is an invalid gfx arch).
  RocmComputeCapability rocm_compute_capability() const;

  const GpuComputeCapability &gpu_compute_capability() const;

  // Returns the maximum amount of shared memory present on a single core
  // (i.e. Streaming Multiprocessor on NVIDIA GPUs; Compute Unit for OpenCL
  // devices). Note that some devices, such as NVIDIA's have a configurable
  // partitioning between shared memory and L1 cache.
  int64_t shared_memory_per_core() const { return shared_memory_per_core_; }

  // Returns the maximum amount of static shared memory
  // available for a single block.
  int64_t shared_memory_per_block() const { return shared_memory_per_block_; }

  // Returns the maximum amount of shared memory available for a single block
  // including the dynamically allocated one.
  int64_t shared_memory_per_block_optin() const {
    return shared_memory_per_block_optin_;
  }

  // L1 size varies because it can be dynamically
  // configured as shared memory; there is no easy way to query its actual size;
  // also we do not count what occupies cache, but rather claim that what is
  // much smaller than the cache size will likely stay in it.
  constexpr int64_t l1_cache_size_per_SM() const {
    return std::visit(
        [](const auto &capability) -> int64_t {
          if constexpr (std::is_same_v<std::decay_t<decltype(capability)>,
                                       RocmComputeCapability>) {
            // MI100 and MI200 has 16KB L1 cache per CU.
            if (capability.gfx9_mi100() || capability.gfx9_mi200()) {
              return 16 * 1024;
            }
            // MI300 has 32KB L1 cache per CU.
            if (capability.gfx9_mi300()) {
              return 32 * 1024;
            }
          }
          // Default return for other GPUs (e.g., RTX A6000).
          return 2 * 1024;
        },
        gpu_compute_capability_);
  }

  constexpr int64_t dram_to_l2_transaction_size_bytes() const {
    return std::visit(
        [](const auto &capability) -> int {
          if constexpr (std::is_same_v<std::decay_t<decltype(capability)>,
                                       RocmComputeCapability>) {
            // DRAM->L2 bus is 128 Byte width for MI300.
            if (capability.gfx9_mi300()) {
              return 128;
            }
          }
          // Cache line is 128B that is split into 4 sectors of 32B. Default
          // transaction size from DRAM -> L2 = 64 Bytes = 2 sectors, since
          // V100, but it can be also configured.
          // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21819-optimizing-applications-for-nvidia-ampere-gpu-architecture.pdf
          // (page 10).
          // return 64 Bytes by default.
          return 64;
        },
        gpu_compute_capability_);
  }

  constexpr int64_t memory_transactions_per_clock() const {
    return std::visit(
        [](const auto &capability) -> int {
          if constexpr (std::is_same_v<std::decay_t<decltype(capability)>,
                                       RocmComputeCapability>) {
            // 16 works well on MI300.
            if (capability.gfx9_mi300()) {
              return 16;
            }
          }
          // Default return for other GPUs.
          return 32;
        },
        gpu_compute_capability_);
  }

  GpuDeviceInfoProto ToGpuProto() const;

  std::string ToString() const;

  DeviceDescription() = default;
  explicit DeviceDescription(const GpuDeviceInfoProto &proto);

  // For string values that are not available via the underlying platform, this
  // value will be provided.
  static inline const char *const kUndefinedString = "<undefined>";

  void set_gpu_compute_capability(const GpuComputeCapability &c) {
    gpu_compute_capability_ = c;
  }

  void set_block_dim_limit_x(int64_t limit) { block_dim_limit_.x = limit; }

  void set_block_dim_limit_y(int64_t limit) { block_dim_limit_.y = limit; }

  void set_block_dim_limit_z(int64_t limit) { block_dim_limit_.z = limit; }

  void set_device_vendor(std::string value) {
    device_vendor_ = std::move(value);
  }
  void set_platform_version(std::string value) {
    platform_version_ = std::move(value);
  }
  void set_driver_version(const SemanticVersion &value) {
    driver_version_ = value;
  }
  void set_runtime_version(const SemanticVersion &value) {
    runtime_version_ = value;
  }
  void set_compile_time_toolkit_version(const SemanticVersion &value) {
    compile_time_toolkit_version_ = value;
  }
  void set_pci_bus_id(std::string value) { pci_bus_id_ = std::move(value); }
  void set_name(std::string value) { name_ = std::move(value); }
  void set_model_str(std::string value) { model_str_ = std::move(value); }

  void set_thread_dim_limit(const ThreadDim &value) {
    thread_dim_limit_ = value;
  }
  void set_block_dim_limit(const BlockDim &value) { block_dim_limit_ = value; }

  void set_threads_per_core_limit(int64_t value) {
    threads_per_core_limit_ = value;
  }
  void set_threads_per_block_limit(int64_t value) {
    threads_per_block_limit_ = value;
  }
  void set_threads_per_warp(int64_t value) { threads_per_warp_ = value; }

  void set_registers_per_core_limit(int64_t value) {
    registers_per_core_limit_ = value;
  }
  void set_registers_per_block_limit(int64_t value) {
    registers_per_block_limit_ = value;
  }

  void set_device_address_bits(int64_t value) { device_address_bits_ = value; }
  void set_device_memory_size(int64_t value) { device_memory_size_ = value; }
  void set_l2_cache_size(int64_t value) { l2_cache_size_ = value; }
  void set_memory_bandwidth(int64_t value) { memory_bandwidth_ = value; }

  void set_shared_memory_per_core(int64_t value) {
    shared_memory_per_core_ = value;
  }
  void set_shared_memory_per_block(int64_t value) {
    shared_memory_per_block_ = value;
  }
  void set_shared_memory_per_block_optin(int64_t value) {
    shared_memory_per_block_optin_ = value;
  }

  void set_clock_rate_ghz(float value) { clock_rate_ghz_ = value; }

  void set_cuda_compute_capability(int major, int minor) {
    gpu_compute_capability_ = CudaComputeCapability{major, minor};
  }

  void set_rocm_compute_capability(std::string gcn_arch_name) {
    gpu_compute_capability_ = RocmComputeCapability(std::move(gcn_arch_name));
  }

  void set_numa_node(int value) { numa_node_ = value; }
  void set_core_count(int value) { core_count_ = value; }
  void set_fpus_per_core(int value) { fpus_per_core_ = value; }
  void set_ecc_enabled(bool value) { ecc_enabled_ = value; }

 private:
  // For description of the following members, see the corresponding accessor
  // above.
  //
  // N.B. If another field is added, update ToMap() above.
  std::string device_vendor_ = kUndefinedString;
  std::string platform_version_ = kUndefinedString;
  std::string pci_bus_id_ = kUndefinedString;
  std::string name_ = kUndefinedString;
  std::string model_str_ = kUndefinedString;

  template <typename T>
  static constexpr T kUninitialized = T(-1);

  ThreadDim thread_dim_limit_{kUninitialized<uint64_t>,
                              kUninitialized<uint64_t>,
                              kUninitialized<uint64_t>};
  BlockDim block_dim_limit_{kUninitialized<uint64_t>, kUninitialized<uint64_t>,
                            kUninitialized<uint64_t>};

  int64_t threads_per_core_limit_ = kUninitialized<int64_t>;
  int64_t threads_per_block_limit_ = kUninitialized<int64_t>;
  int64_t threads_per_warp_ = kUninitialized<int64_t>;

  int64_t registers_per_core_limit_ = kUninitialized<int64_t>;
  int64_t registers_per_block_limit_ = kUninitialized<int64_t>;

  int64_t device_address_bits_ = kUninitialized<int64_t>;
  int64_t device_memory_size_ = kUninitialized<int64_t>;
  int64_t l2_cache_size_ = kUninitialized<int64_t>;
  int64_t memory_bandwidth_ = kUninitialized<int64_t>;

  // Shared memory limits on a given device.
  int64_t shared_memory_per_core_ = kUninitialized<int64_t>;
  int64_t shared_memory_per_block_ = kUninitialized<int64_t>;
  int64_t shared_memory_per_block_optin_ = kUninitialized<int64_t>;

  float clock_rate_ghz_ = kUninitialized<float>;

  GpuComputeCapability gpu_compute_capability_{};

  int numa_node_ = kUninitialized<int>;
  int core_count_ = kUninitialized<int>;
  int fpus_per_core_ = kUninitialized<int>;
  bool ecc_enabled_ = false;

  SemanticVersion driver_version_{0, 0, 0};
  SemanticVersion runtime_version_{0, 0, 0};
  SemanticVersion compile_time_toolkit_version_{0, 0, 0};
};

// Returns whether the given thread_dim is acceptable given the limits described
// in device_description. For detailed reasons for failing the predicate, enable
// VLOG(2) for this module.
bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim);

// Calculate the number of threads/blocks required to process element_count
// elements. Note that you can still end up with more threads than
// element_count due to rounding, so kernels often start with an "is this
// thread id in the element_count range?" test.
void CalculateDimensionality(const DeviceDescription &device_description,
                             int64_t element_count, int64_t *threads_per_block,
                             int64_t *block_count);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
