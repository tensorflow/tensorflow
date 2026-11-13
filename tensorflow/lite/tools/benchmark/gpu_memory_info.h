/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_GPU_MEMORY_INFO_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_GPU_MEMORY_INFO_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace tflite {
namespace benchmark {

// Point-in-time per-process GPU / accelerator memory snapshot (in bytes; -1 if
// unavailable).
struct GpuMemorySnapshot {
  static constexpr int64_t kUnavailable = -1;

  // Indicates whether obtaining GPU memory usage is supported on the platform.
  static bool IsSupported();

  bool HasGpu() const { return gpu_bytes >= 0; }

  // Total per-process GPU / accelerator memory footprint in bytes.
  int64_t gpu_bytes = kUnavailable;

  // Per-source breakdown of `gpu_bytes`, ordered largest-first.
  std::vector<std::pair<std::string, int64_t>> gpu_breakdown;
};

// Returns the current GPU memory usage snapshot from the system.
// `include_external_tools` is false on periodic background ticks and true only
// at untimed benchmark phase boundaries (e.g., to allow one-shot
// `dumpsys gpu --gpumem` queries on non-sandboxed Android CLI runs).
GpuMemorySnapshot GetGpuMemorySnapshot(bool include_external_tools = false);

namespace internal {

struct AndroidGpuMappings {
  int64_t kgsl_bytes = 0;
  int64_t drm_bytes = 0;
  bool has_mali = false;
  bool maps_accessible = false;
};

struct DmaBufInfo {
  int64_t total_bytes = 0;
  std::vector<std::pair<std::string, int64_t>> by_exporter;
  bool fdinfo_accessible = false;
};

struct DrmClientRegion {
  std::string name;
  int64_t resident_bytes = 0;
};

struct DrmClientInfo {
  std::string driver;
  std::string pdev;
  std::string client_id;
  std::vector<DrmClientRegion> regions;
};

struct AppleGpuLedger {
  int64_t graphics_bytes = 0;
  int64_t graphics_compressed_bytes = 0;
  int64_t neural_bytes = 0;
  int64_t neural_compressed_bytes = 0;
};

struct WindowsAdapterVideoMemory {
  std::string description;
  int64_t local_bytes = 0;
  int64_t non_local_bytes = 0;
};

// Platform-independent parsers exposed for unit testing across host OSes.
std::optional<int64_t> ParseProcStatusVmRssBytes(absl::string_view contents);
std::optional<int64_t> ParseSmapsRollupRssBytes(absl::string_view contents);
AndroidGpuMappings ParseProcMapsAndroidGpu(absl::string_view contents);
absl::flat_hash_map<std::string, std::string> ParseFdinfoFields(
    absl::string_view contents);
std::optional<DrmClientInfo> ParseDrmFdinfoClient(
    const absl::flat_hash_map<std::string, std::string>& fields);
int64_t ParseDumpsysGpuMemPidBytes(absl::string_view dumpsys_output, int pid);

GpuMemorySnapshot CombineAndroidGpuMemory(
    std::optional<int64_t> smaps_rss_bytes, std::optional<int64_t> vm_rss_bytes,
    const AndroidGpuMappings& mappings, const DmaBufInfo& dmabuf,
    int64_t external_gpumem_bytes = GpuMemorySnapshot::kUnavailable);

GpuMemorySnapshot CombineLinuxGpuMemory(
    const std::vector<DrmClientInfo>& drm_clients, bool fdinfo_accessible,
    std::optional<int64_t> nvml_process_bytes);

GpuMemorySnapshot CombineAppleGpuMemory(
    std::optional<AppleGpuLedger> ledger,
    std::optional<int64_t> metal_allocated_bytes);

GpuMemorySnapshot CombineWindowsGpuMemory(
    const std::vector<WindowsAdapterVideoMemory>& adapters);

#if defined(__linux__)
DmaBufInfo ReadAndroidDmaBufInfo(absl::string_view proc_self_dir);
std::vector<DrmClientInfo> ReadLinuxDrmClients(absl::string_view proc_self_dir,
                                               bool* fdinfo_accessible);
GpuMemorySnapshot ReadAndroidProcGpuMemorySnapshot(
    absl::string_view proc_self_dir,
    int64_t external_gpumem_bytes = GpuMemorySnapshot::kUnavailable);
GpuMemorySnapshot ReadLinuxProcGpuMemorySnapshot(
    absl::string_view proc_self_dir, std::optional<int64_t> nvml_process_bytes);
#endif  // defined(__linux__)

#if defined(__APPLE__)
std::optional<AppleGpuLedger> ReadAppleGpuLedger();
std::optional<int64_t> QueryAppleMetalAllocatedBytes();
GpuMemorySnapshot ReadAppleGpuMemorySnapshot();
#endif  // defined(__APPLE__)

#if defined(_WIN32)
std::vector<WindowsAdapterVideoMemory> ReadWindowsAdapterVideoMemory();
GpuMemorySnapshot ReadWindowsGpuMemorySnapshot();
#endif  // defined(_WIN32)

}  // namespace internal

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_GPU_MEMORY_INFO_H_
