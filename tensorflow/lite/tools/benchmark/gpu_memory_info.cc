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

#include "tensorflow/lite/tools/benchmark/gpu_memory_info.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // defined(__linux__)

#if defined(__ANDROID__)
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <atomic>
#endif  // defined(__ANDROID__)

#if defined(__linux__) && !defined(__ANDROID__)
#include "tensorflow/lite/shared_library.h"
#endif  // defined(__linux__) && !defined(__ANDROID__)

#if defined(__APPLE__)
#include <dlfcn.h>
#include <mach/mach.h>
#endif  // defined(__APPLE__)

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <dxgi1_4.h>
#include <windows.h>
#include <wrl/client.h>
#endif  // defined(_WIN32)

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

namespace tflite {
namespace benchmark {

namespace internal {
namespace {

std::optional<int64_t> ParseKbLineToBytes(absl::string_view line,
                                          absl::string_view key) {
  if (!absl::ConsumePrefix(&line, key) || !absl::ConsumePrefix(&line, ":")) {
    return std::nullopt;
  }
  line = absl::StripAsciiWhitespace(line);
  absl::ConsumeSuffix(&line, "kB");
  int64_t value_kb = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(line), &value_kb) ||
      value_kb < 0) {
    return std::nullopt;
  }
  return value_kb * 1024;
}

std::optional<int64_t> ParseDrmSizeBytes(absl::string_view value) {
  value = absl::StripAsciiWhitespace(value);
  int64_t multiplier = 1;
  if (absl::ConsumeSuffix(&value, "KiB") || absl::ConsumeSuffix(&value, "kB")) {
    multiplier = 1024;
  } else if (absl::ConsumeSuffix(&value, "MiB") ||
             absl::ConsumeSuffix(&value, "MB")) {
    multiplier = 1024 * 1024;
  } else if (absl::ConsumeSuffix(&value, "GiB") ||
             absl::ConsumeSuffix(&value, "GB")) {
    multiplier = 1024LL * 1024 * 1024;
  }
  int64_t count = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(value), &count) ||
      count < 0) {
    return std::nullopt;
  }
  return count * multiplier;
}

std::vector<std::pair<std::string, int64_t>> ExtractDrmRegions(
    const absl::flat_hash_map<std::string, std::string>& fields,
    absl::string_view prefix) {
  std::vector<std::pair<std::string, int64_t>> regions;
  for (const auto& [key, field_value] : fields) {
    if (!absl::StartsWith(key, prefix)) continue;
    if (std::optional<int64_t> bytes = ParseDrmSizeBytes(field_value)) {
      regions.emplace_back(key.substr(prefix.size()), *bytes);
    }
  }
  std::sort(regions.begin(), regions.end());
  return regions;
}

void SortBreakdownLargestFirst(
    std::vector<std::pair<std::string, int64_t>>* breakdown) {
  std::sort(breakdown->begin(), breakdown->end(),
            [](const auto& a, const auto& b) {
              return std::tie(b.second, a.first) < std::tie(a.second, b.first);
            });
}

#if defined(__linux__)
std::optional<std::string> ReadWholeProcFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) return std::nullopt;
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}
#endif  // defined(__linux__)

#if defined(__ANDROID__)
// Queries `/system/bin/dumpsys gpu --gpumem` without invoking a shell.
// Restricted to non-app UIDs (e.g., `adb shell` / `root` CLI runs) and called
// only at untimed benchmark phase boundaries so periodic sampling never forks.
// If unavailable or blocked by SELinux, disables future attempts immediately.
int64_t QueryAndroidDumpsysGpuMemBytes() {
  static std::atomic<bool> dumpsys_disabled{false};
  if (dumpsys_disabled.load(std::memory_order_relaxed)) {
    return GpuMemorySnapshot::kUnavailable;
  }

  constexpr uid_t kAndroidAppUidStart = 10000;
  constexpr uid_t kAndroidPerUserRange = 100000;
  if ((geteuid() % kAndroidPerUserRange) >= kAndroidAppUidStart ||
      access("/system/bin/dumpsys", X_OK) != 0) {
    dumpsys_disabled.store(true, std::memory_order_relaxed);
    return GpuMemorySnapshot::kUnavailable;
  }

  int pipefd[2];
  if (pipe2(pipefd, O_CLOEXEC) != 0) {
    return GpuMemorySnapshot::kUnavailable;
  }

  const pid_t child_pid = fork();
  if (child_pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    return GpuMemorySnapshot::kUnavailable;
  }

  if (child_pid == 0) {
    const int devnull = open("/dev/null", O_RDWR);
    if (devnull >= 0) {
      dup2(devnull, STDIN_FILENO);
      dup2(devnull, STDERR_FILENO);
      if (devnull > STDERR_FILENO) close(devnull);
    }
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);
    char* const argv[] = {const_cast<char*>("/system/bin/dumpsys"),
                          const_cast<char*>("gpu"),
                          const_cast<char*>("--gpumem"), nullptr};
    execv("/system/bin/dumpsys", argv);
    _exit(127);
  }

  close(pipefd[1]);
  std::string output;
  char chunk[1024];
  constexpr size_t kMaxOutputBytes = 256 * 1024;
  ssize_t n = 0;
  while ((n = read(pipefd[0], chunk, sizeof(chunk))) > 0) {
    if (output.size() + static_cast<size_t>(n) <= kMaxOutputBytes) {
      output.append(chunk, static_cast<size_t>(n));
    }
  }
  close(pipefd[0]);

  int status = 0;
  while (waitpid(child_pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    dumpsys_disabled.store(true, std::memory_order_relaxed);
    return GpuMemorySnapshot::kUnavailable;
  }

  const int64_t bytes =
      ParseDumpsysGpuMemPidBytes(output, static_cast<int>(getpid()));
  if (bytes < 0) {
    dumpsys_disabled.store(true, std::memory_order_relaxed);
  }
  return bytes;
}
#endif  // defined(__ANDROID__)

#if defined(__linux__) && !defined(__ANDROID__)
// Dynamically queries per-process NVIDIA framebuffer usage via NVML when
// libnvidia-ml.so is installed on the host.
std::optional<int64_t> QueryLinuxNvmlProcessBytes() {
  using NvmlReturn = int;
  constexpr NvmlReturn kNvmlSuccess = 0;
  constexpr NvmlReturn kNvmlErrorInsufficientSize = 7;
  constexpr uint64_t kNvmlValueNotAvailable = ~uint64_t{0};
  using NvmlDevice = struct NvmlDeviceSt*;
  struct NvmlProcessInfo {
    unsigned int pid;
    uint64_t used_gpu_memory;
    unsigned int gpu_instance_id;
    unsigned int compute_instance_id;
  };
  using InitFn = NvmlReturn (*)();
  using GetCountFn = NvmlReturn (*)(unsigned int*);
  using GetHandleFn = NvmlReturn (*)(unsigned int, NvmlDevice*);
  using GetMaxMigCountFn = NvmlReturn (*)(NvmlDevice, unsigned int*);
  using GetMigHandleFn = NvmlReturn (*)(NvmlDevice, unsigned int, NvmlDevice*);
  using GetProcessesFn =
      NvmlReturn (*)(NvmlDevice, unsigned int*, NvmlProcessInfo*);

  struct NvmlApi {
    GetProcessesFn get_compute = nullptr;
    GetProcessesFn get_graphics = nullptr;
    std::vector<NvmlDevice> devices;
    bool available() const {
      return !devices.empty() && get_compute != nullptr &&
             get_graphics != nullptr;
    }
  };

  static const NvmlApi* const api = []() -> NvmlApi* {
    auto* loaded = new NvmlApi();
    void* lib = SharedLibrary::LoadLibrary("libnvidia-ml.so.1");
    if (lib == nullptr) lib = SharedLibrary::LoadLibrary("libnvidia-ml.so");
    if (lib == nullptr) return loaded;

    auto init = reinterpret_cast<InitFn>(
        SharedLibrary::GetLibrarySymbol(lib, "nvmlInit_v2"));
    auto get_count = reinterpret_cast<GetCountFn>(
        SharedLibrary::GetLibrarySymbol(lib, "nvmlDeviceGetCount_v2"));
    auto get_handle = reinterpret_cast<GetHandleFn>(
        SharedLibrary::GetLibrarySymbol(lib, "nvmlDeviceGetHandleByIndex_v2"));
    auto get_max_mig = reinterpret_cast<GetMaxMigCountFn>(
        SharedLibrary::GetLibrarySymbol(lib, "nvmlDeviceGetMaxMigDeviceCount"));
    auto get_mig_handle =
        reinterpret_cast<GetMigHandleFn>(SharedLibrary::GetLibrarySymbol(
            lib, "nvmlDeviceGetMigDeviceHandleByIndex"));
    auto get_compute =
        reinterpret_cast<GetProcessesFn>(SharedLibrary::GetLibrarySymbol(
            lib, "nvmlDeviceGetComputeRunningProcesses_v3"));
    if (get_compute == nullptr) {
      get_compute =
          reinterpret_cast<GetProcessesFn>(SharedLibrary::GetLibrarySymbol(
              lib, "nvmlDeviceGetComputeRunningProcesses_v2"));
    }
    auto get_graphics =
        reinterpret_cast<GetProcessesFn>(SharedLibrary::GetLibrarySymbol(
            lib, "nvmlDeviceGetGraphicsRunningProcesses_v3"));
    if (get_graphics == nullptr) {
      get_graphics =
          reinterpret_cast<GetProcessesFn>(SharedLibrary::GetLibrarySymbol(
              lib, "nvmlDeviceGetGraphicsRunningProcesses_v2"));
    }
    if (init == nullptr || get_count == nullptr || get_handle == nullptr ||
        get_compute == nullptr || get_graphics == nullptr) {
      return loaded;
    }
    if (init() != kNvmlSuccess) {
      return loaded;
    }
    unsigned int device_count = 0;
    if (get_count(&device_count) != kNvmlSuccess || device_count == 0) {
      return loaded;
    }
    for (unsigned int i = 0; i < device_count; ++i) {
      NvmlDevice device = nullptr;
      if (get_handle(i, &device) != kNvmlSuccess || device == nullptr) continue;
      bool added_mig = false;
      unsigned int max_mig = 0;
      if (get_max_mig != nullptr && get_mig_handle != nullptr &&
          get_max_mig(device, &max_mig) == kNvmlSuccess && max_mig > 0) {
        for (unsigned int m = 0; m < max_mig; ++m) {
          NvmlDevice mig_device = nullptr;
          if (get_mig_handle(device, m, &mig_device) == kNvmlSuccess &&
              mig_device != nullptr) {
            loaded->devices.push_back(mig_device);
            added_mig = true;
          }
        }
      }
      if (!added_mig) {
        loaded->devices.push_back(device);
      }
    }
    loaded->get_compute = get_compute;
    loaded->get_graphics = get_graphics;
    return loaded;
  }();

  if (!api->available()) return std::nullopt;

  const unsigned int self_pid = static_cast<unsigned int>(getpid());
  const auto collect_processes =
      [&](GetProcessesFn fn, NvmlDevice device,
          std::vector<std::pair<unsigned int, uint64_t>>& pid_bytes) {
        unsigned int count = 0;
        NvmlReturn status = fn(device, &count, nullptr);
        if ((status != kNvmlSuccess && status != kNvmlErrorInsufficientSize) ||
            count == 0) {
          return;
        }
        std::vector<NvmlProcessInfo> infos(count + 8);
        count = static_cast<unsigned int>(infos.size());
        if (fn(device, &count, infos.data()) != kNvmlSuccess) return;
        for (unsigned int i = 0; i < count && i < infos.size(); ++i) {
          if (infos[i].used_gpu_memory == kNvmlValueNotAvailable) continue;
          bool found = false;
          for (auto& [pid, bytes] : pid_bytes) {
            if (pid == infos[i].pid) {
              // A process active in both compute and graphics lists reports its
              // full framebuffer usage in each list; take max() to avoid double
              // counting.
              bytes = std::max(bytes, infos[i].used_gpu_memory);
              found = true;
              break;
            }
          }
          if (!found) {
            pid_bytes.emplace_back(infos[i].pid, infos[i].used_gpu_memory);
          }
        }
      };

  int64_t total_nvml_bytes = 0;
  for (NvmlDevice device : api->devices) {
    std::vector<std::pair<unsigned int, uint64_t>> pid_bytes;
    collect_processes(api->get_compute, device, pid_bytes);
    collect_processes(api->get_graphics, device, pid_bytes);

    bool matched_self = false;
    for (const auto& [pid, bytes] : pid_bytes) {
      if (pid == self_pid) {
        total_nvml_bytes += static_cast<int64_t>(bytes);
        matched_self = true;
        break;
      }
    }
    if (matched_self) continue;

    // Inside a container with an isolated PID namespace, NVML reports host PIDs
    // that do not exist in the container's /proc. When exactly one such foreign
    // PID is active on the assigned GPU/MIG instance, it belongs to this
    // container's GPU context.
    unsigned int foreign_count = 0;
    uint64_t foreign_bytes = 0;
    for (const auto& [pid, bytes] : pid_bytes) {
      struct stat st;
      const std::string proc_path = absl::StrCat("/proc/", pid);
      if (stat(proc_path.c_str(), &st) != 0) {
        ++foreign_count;
        foreign_bytes = bytes;
      }
    }
    if (foreign_count == 1) {
      total_nvml_bytes += static_cast<int64_t>(foreign_bytes);
    }
  }
  return total_nvml_bytes;
}
#endif  // defined(__linux__) && !defined(__ANDROID__)

}  // namespace

std::optional<int64_t> ParseProcStatusVmRssBytes(absl::string_view contents) {
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    if (std::optional<int64_t> vm_rss_bytes =
            ParseKbLineToBytes(line, "VmRSS")) {
      return *vm_rss_bytes;
    }
  }
  return std::nullopt;
}

std::optional<int64_t> ParseSmapsRollupRssBytes(absl::string_view contents) {
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    if (std::optional<int64_t> rss_bytes = ParseKbLineToBytes(line, "Rss")) {
      return *rss_bytes;
    }
  }
  return std::nullopt;
}

AndroidGpuMappings ParseProcMapsAndroidGpu(absl::string_view contents) {
  AndroidGpuMappings result;
  result.maps_accessible = true;
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    if (line.empty()) continue;
    std::vector<absl::string_view> fields =
        absl::StrSplit(line, absl::MaxSplits(' ', 5), absl::SkipEmpty());
    if (fields.size() < 5) continue;
    std::vector<absl::string_view> range = absl::StrSplit(fields[0], '-');
    uint64_t start = 0;
    uint64_t end = 0;
    if (range.size() != 2 || !absl::SimpleHexAtoi(range[0], &start) ||
        !absl::SimpleHexAtoi(range[1], &end) || end < start) {
      continue;
    }
    if (fields.size() < 6) continue;
    const absl::string_view path = absl::StripAsciiWhitespace(fields[5]);
    const int64_t size_bytes = static_cast<int64_t>(end - start);
    if (absl::StrContains(path, "kgsl-3d0")) {
      result.kgsl_bytes += size_bytes;
    } else if (absl::StartsWith(path, "/dev/dri/renderD")) {
      result.drm_bytes += size_bytes;
    } else if (absl::StartsWith(path, "/dev/mali")) {
      result.has_mali = true;
    }
  }
  return result;
}

absl::flat_hash_map<std::string, std::string> ParseFdinfoFields(
    absl::string_view contents) {
  absl::flat_hash_map<std::string, std::string> fields;
  for (absl::string_view line : absl::StrSplit(contents, '\n')) {
    const size_t colon = line.find(':');
    if (colon == absl::string_view::npos) continue;
    fields[std::string(absl::StripAsciiWhitespace(line.substr(0, colon)))] =
        std::string(absl::StripAsciiWhitespace(line.substr(colon + 1)));
  }
  return fields;
}

std::optional<DrmClientInfo> ParseDrmFdinfoClient(
    const absl::flat_hash_map<std::string, std::string>& fields) {
  const auto driver_it = fields.find("drm-driver");
  if (driver_it == fields.end() || driver_it->second.empty()) {
    return std::nullopt;
  }
  DrmClientInfo client;
  client.driver = driver_it->second;
  if (const auto it = fields.find("drm-pdev"); it != fields.end()) {
    client.pdev = it->second;
  }
  if (const auto it = fields.find("drm-client-id"); it != fields.end()) {
    client.client_id = it->second;
  }

  std::vector<std::pair<std::string, int64_t>> totals =
      ExtractDrmRegions(fields, "drm-total-");
  if (totals.empty()) {
    totals = ExtractDrmRegions(fields, "drm-memory-");
  }
  const std::vector<std::pair<std::string, int64_t>> residents =
      ExtractDrmRegions(fields, "drm-resident-");
  client.regions.reserve(totals.size());
  for (const auto& [name, total_bytes] : totals) {
    int64_t resident_bytes = total_bytes;
    const auto res_it =
        std::find_if(residents.begin(), residents.end(),
                     [&name = name](const auto& r) { return r.first == name; });
    if (res_it != residents.end()) {
      resident_bytes = res_it->second;
    }
    client.regions.push_back({.name = name, .resident_bytes = resident_bytes});
  }
  return client;
}

int64_t ParseDumpsysGpuMemPidBytes(absl::string_view dumpsys_output, int pid) {
  if (!absl::StrContains(dumpsys_output, "Memory snapshot for GPU")) {
    return GpuMemorySnapshot::kUnavailable;
  }
  const std::string proc_prefix = absl::StrCat("Proc ", pid, " total:");
  int64_t total_bytes = 0;
  for (absl::string_view line : absl::StrSplit(dumpsys_output, '\n')) {
    line = absl::StripAsciiWhitespace(line);
    if (!absl::ConsumePrefix(&line, proc_prefix)) continue;
    int64_t bytes = 0;
    if (absl::SimpleAtoi(absl::StripAsciiWhitespace(line), &bytes) &&
        bytes >= 0) {
      total_bytes += bytes;
    }
  }
  return total_bytes;
}

GpuMemorySnapshot CombineAndroidGpuMemory(
    std::optional<int64_t> smaps_rss_bytes, std::optional<int64_t> vm_rss_bytes,
    const AndroidGpuMappings& mappings, const DmaBufInfo& dmabuf,
    int64_t external_gpumem_bytes) {
  GpuMemorySnapshot snapshot;
  if (!mappings.maps_accessible && !dmabuf.fdinfo_accessible &&
      external_gpumem_bytes < 0) {
    snapshot.gpu_bytes = GpuMemorySnapshot::kUnavailable;
    return snapshot;
  }

  // On Mali (`mali_kbase`), unmapped driver pages are added directly to
  // `mm->rss_stat[MM_FILEPAGES]` (`VmRSS`), whereas `/proc/self/smaps_rollup`
  // (`Rss`) counts only VMA-backed CPU mappings.
  int64_t mali_bytes = 0;
  if (mappings.has_mali && vm_rss_bytes.has_value() &&
      smaps_rss_bytes.has_value() && *vm_rss_bytes > *smaps_rss_bytes) {
    mali_bytes = *vm_rss_bytes - *smaps_rss_bytes;
  }

  const int64_t proc_driver_bytes =
      mali_bytes + mappings.kgsl_bytes + mappings.drm_bytes;
  const int64_t proc_known_bytes = proc_driver_bytes + dmabuf.total_bytes;
  const int64_t unmapped_driver_bytes =
      external_gpumem_bytes > proc_known_bytes
          ? (external_gpumem_bytes - proc_known_bytes)
          : 0;

  if (mappings.kgsl_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("gpu_driver_kgsl", mappings.kgsl_bytes);
  }
  if (mappings.drm_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("gpu_driver_drm", mappings.drm_bytes);
  }
  if (mali_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("gpu_driver_mali", mali_bytes);
  }
  if (unmapped_driver_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("gpu_driver_unmapped",
                                        unmapped_driver_bytes);
  }
  for (const auto& [exp_name, bytes] : dmabuf.by_exporter) {
    if (bytes > 0) {
      snapshot.gpu_breakdown.emplace_back(absl::StrCat("dmabuf:", exp_name),
                                          bytes);
    }
  }
  SortBreakdownLargestFirst(&snapshot.gpu_breakdown);

  snapshot.gpu_bytes =
      proc_driver_bytes + unmapped_driver_bytes + dmabuf.total_bytes;
  return snapshot;
}

GpuMemorySnapshot CombineLinuxGpuMemory(
    const std::vector<DrmClientInfo>& drm_clients, bool fdinfo_accessible,
    std::optional<int64_t> nvml_process_bytes) {
  GpuMemorySnapshot snapshot;
  if (!fdinfo_accessible && !nvml_process_bytes.has_value()) {
    snapshot.gpu_bytes = GpuMemorySnapshot::kUnavailable;
    return snapshot;
  }

  absl::flat_hash_map<std::string, int64_t> region_bytes;
  bool any_drm_memory_keys = false;
  int64_t total_gpu_bytes = 0;

  for (const DrmClientInfo& client : drm_clients) {
    for (const DrmClientRegion& region : client.regions) {
      any_drm_memory_keys = true;
      total_gpu_bytes += region.resident_bytes;
      region_bytes[absl::StrCat("drm:", region.name)] += region.resident_bytes;
    }
  }

  // If DRM clients exist (e.g. pre-5.19 kernel or proprietary driver) but
  // expose no memory region keys and NVML is not available, report GPU memory
  // as unavailable rather than a misleading zero.
  if (!drm_clients.empty() && !any_drm_memory_keys &&
      !nvml_process_bytes.has_value()) {
    snapshot.gpu_bytes = GpuMemorySnapshot::kUnavailable;
    return snapshot;
  }

  if (nvml_process_bytes.has_value() && *nvml_process_bytes > 0) {
    total_gpu_bytes += *nvml_process_bytes;
    region_bytes["nvml:vram"] += *nvml_process_bytes;
  }

  for (const auto& [label, bytes] : region_bytes) {
    if (bytes > 0) {
      snapshot.gpu_breakdown.emplace_back(label, bytes);
    }
  }
  SortBreakdownLargestFirst(&snapshot.gpu_breakdown);
  snapshot.gpu_bytes = total_gpu_bytes;
  return snapshot;
}

GpuMemorySnapshot CombineAppleGpuMemory(
    std::optional<AppleGpuLedger> ledger,
    std::optional<int64_t> metal_allocated_bytes) {
  GpuMemorySnapshot snapshot;
  if (!ledger.has_value() && !metal_allocated_bytes.has_value()) {
    snapshot.gpu_bytes = GpuMemorySnapshot::kUnavailable;
    return snapshot;
  }

  const AppleGpuLedger l = ledger.value_or(AppleGpuLedger{});
  const int64_t graphics_bytes =
      std::max<int64_t>(0, l.graphics_bytes) +
      std::max<int64_t>(0, l.graphics_compressed_bytes);
  const int64_t neural_bytes = std::max<int64_t>(0, l.neural_bytes) +
                               std::max<int64_t>(0, l.neural_compressed_bytes);
  const int64_t charged_bytes = graphics_bytes + neural_bytes;
  const int64_t metal_bytes =
      std::max<int64_t>(0, metal_allocated_bytes.value_or(0));
  const int64_t mapped_weights_bytes =
      metal_bytes > charged_bytes ? (metal_bytes - charged_bytes) : 0;

  if (graphics_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("graphics", graphics_bytes);
  }
  if (neural_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("neural", neural_bytes);
  }
  if (mapped_weights_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("mapped_weights", mapped_weights_bytes);
  }
  SortBreakdownLargestFirst(&snapshot.gpu_breakdown);

  snapshot.gpu_bytes = graphics_bytes + neural_bytes + mapped_weights_bytes;
  return snapshot;
}

GpuMemorySnapshot CombineWindowsGpuMemory(
    const std::vector<WindowsAdapterVideoMemory>& adapters) {
  GpuMemorySnapshot snapshot;
  if (adapters.empty()) {
    snapshot.gpu_bytes = GpuMemorySnapshot::kUnavailable;
    return snapshot;
  }

  int64_t local_bytes = 0;
  int64_t non_local_bytes = 0;
  for (const WindowsAdapterVideoMemory& adapter : adapters) {
    local_bytes += std::max<int64_t>(0, adapter.local_bytes);
    non_local_bytes += std::max<int64_t>(0, adapter.non_local_bytes);
  }

  if (local_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("dxgi:local", local_bytes);
  }
  if (non_local_bytes > 0) {
    snapshot.gpu_breakdown.emplace_back("dxgi:non_local", non_local_bytes);
  }
  SortBreakdownLargestFirst(&snapshot.gpu_breakdown);

  snapshot.gpu_bytes = local_bytes + non_local_bytes;
  return snapshot;
}

#if defined(__linux__)
DmaBufInfo ReadAndroidDmaBufInfo(absl::string_view proc_self_dir) {
  DmaBufInfo info;
  const std::string fdinfo_dir_path = absl::StrCat(proc_self_dir, "/fdinfo");
  DIR* dir = opendir(fdinfo_dir_path.c_str());
  if (dir == nullptr) {
    return info;
  }
  info.fdinfo_accessible = true;

  struct DmaBufEntry {
    int64_t size_bytes = 0;
    std::string exp_name;
  };
  absl::flat_hash_map<uint64_t, DmaBufEntry> unique_buffers;
  uint64_t fallback_id = 0;

  while (const struct dirent* entry = readdir(dir)) {
    if (entry->d_name[0] == '.') continue;
    const std::optional<std::string> contents =
        ReadWholeProcFile(absl::StrCat(fdinfo_dir_path, "/", entry->d_name));
    if (!contents.has_value()) continue;
    const absl::flat_hash_map<std::string, std::string> fields =
        ParseFdinfoFields(*contents);
    const auto size_it = fields.find("size");
    const auto exp_it = fields.find("exp_name");
    if (size_it == fields.end() || exp_it == fields.end() ||
        exp_it->second.empty()) {
      continue;
    }
    int64_t size_bytes = 0;
    if (!absl::SimpleAtoi(size_it->second, &size_bytes) || size_bytes <= 0) {
      continue;
    }

    uint64_t inode = 0;
    if (const auto ino_it = fields.find("ino"); ino_it != fields.end()) {
      (void)absl::SimpleAtoi(ino_it->second, &inode);
    }
    if (inode == 0) {
      struct stat st;
      if (stat(absl::StrCat(proc_self_dir, "/fd/", entry->d_name).c_str(),
               &st) == 0 &&
          st.st_ino != 0) {
        inode = static_cast<uint64_t>(st.st_ino);
      } else {
        inode = (uint64_t{1} << 63) | (++fallback_id);
      }
    }
    unique_buffers[inode] =
        DmaBufEntry{.size_bytes = size_bytes, .exp_name = exp_it->second};
  }
  closedir(dir);

  absl::flat_hash_map<std::string, int64_t> exporter_totals;
  for (const auto& [_, buf] : unique_buffers) {
    info.total_bytes += buf.size_bytes;
    exporter_totals[buf.exp_name] += buf.size_bytes;
  }
  info.by_exporter.assign(exporter_totals.begin(), exporter_totals.end());
  SortBreakdownLargestFirst(&info.by_exporter);
  return info;
}

std::vector<DrmClientInfo> ReadLinuxDrmClients(absl::string_view proc_self_dir,
                                               bool* fdinfo_accessible) {
  std::vector<DrmClientInfo> clients;
  const std::string fdinfo_dir_path = absl::StrCat(proc_self_dir, "/fdinfo");
  DIR* dir = opendir(fdinfo_dir_path.c_str());
  if (dir == nullptr) {
    if (fdinfo_accessible != nullptr) *fdinfo_accessible = false;
    return clients;
  }
  if (fdinfo_accessible != nullptr) *fdinfo_accessible = true;

  absl::flat_hash_set<std::string> seen_clients;
  while (const struct dirent* entry = readdir(dir)) {
    if (entry->d_name[0] == '.') continue;
    const std::optional<std::string> contents =
        ReadWholeProcFile(absl::StrCat(fdinfo_dir_path, "/", entry->d_name));
    if (!contents.has_value()) continue;
    std::optional<DrmClientInfo> client =
        ParseDrmFdinfoClient(ParseFdinfoFields(*contents));
    if (!client.has_value()) continue;
    const std::string client_key = absl::StrCat(
        client->driver, "\n", client->pdev, "\n",
        client->client_id.empty() ? entry->d_name : client->client_id);
    if (seen_clients.insert(client_key).second) {
      clients.push_back(*std::move(client));
    }
  }
  closedir(dir);
  return clients;
}

GpuMemorySnapshot ReadAndroidProcGpuMemorySnapshot(
    absl::string_view proc_self_dir, int64_t external_gpumem_bytes) {
  const std::optional<std::string> maps_contents =
      ReadWholeProcFile(absl::StrCat(proc_self_dir, "/maps"));
  const AndroidGpuMappings mappings =
      maps_contents.has_value() ? ParseProcMapsAndroidGpu(*maps_contents)
                                : AndroidGpuMappings{};

  std::optional<int64_t> vm_rss;
  std::optional<int64_t> smaps_rss;
  if (mappings.has_mali) {
    if (const std::optional<std::string> status_contents =
            ReadWholeProcFile(absl::StrCat(proc_self_dir, "/status"))) {
      vm_rss = ParseProcStatusVmRssBytes(*status_contents);
    }
    if (const std::optional<std::string> smaps_contents =
            ReadWholeProcFile(absl::StrCat(proc_self_dir, "/smaps_rollup"))) {
      smaps_rss = ParseSmapsRollupRssBytes(*smaps_contents);
    }
  }
  const DmaBufInfo dmabuf = ReadAndroidDmaBufInfo(proc_self_dir);

  return CombineAndroidGpuMemory(smaps_rss, vm_rss, mappings, dmabuf,
                                 external_gpumem_bytes);
}

GpuMemorySnapshot ReadLinuxProcGpuMemorySnapshot(
    absl::string_view proc_self_dir,
    std::optional<int64_t> nvml_process_bytes) {
  bool fdinfo_accessible = false;
  const std::vector<DrmClientInfo> drm_clients =
      ReadLinuxDrmClients(proc_self_dir, &fdinfo_accessible);
  return CombineLinuxGpuMemory(drm_clients, fdinfo_accessible,
                               nvml_process_bytes);
}
#endif  // defined(__linux__)

#if defined(__APPLE__)
std::optional<AppleGpuLedger> ReadAppleGpuLedger() {
  task_vm_info_data_t info = {};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return std::nullopt;
  }
  if (count < TASK_VM_INFO_REV3_COUNT) {
    return std::nullopt;
  }
  const auto non_negative = [](int64_t bytes) -> int64_t {
    return bytes > 0 ? bytes : 0;
  };
  AppleGpuLedger ledger;
  ledger.graphics_bytes = non_negative(info.ledger_tag_graphics_footprint);
  ledger.graphics_compressed_bytes =
      non_negative(info.ledger_tag_graphics_footprint_compressed);
  ledger.neural_bytes = non_negative(info.ledger_tag_neural_footprint);
  ledger.neural_compressed_bytes =
      non_negative(info.ledger_tag_neural_footprint_compressed);
  return ledger;
}

std::optional<int64_t> QueryAppleMetalAllocatedBytes() {
  using Sel = void*;
  using RegisterNameFn = Sel (*)(const char*);
  using MsgSendUsizeFn = size_t (*)(void*, Sel);
  using MsgSendIdxFn = void* (*)(void*, Sel, size_t);
  using CopyAllDevicesFn = void* (*)();
  using CreateDefaultDeviceFn = void* (*)();

  struct MetalApi {
    MsgSendUsizeFn msg_send_usize = nullptr;
    Sel current_allocated_size_sel = nullptr;
    std::vector<void*> devices;
    bool available() const {
      return msg_send_usize != nullptr &&
             current_allocated_size_sel != nullptr && !devices.empty();
    }
  };

  static const MetalApi* const api = []() -> MetalApi* {
    auto* loaded = new MetalApi();
    void* objc_lib = dlopen("/usr/lib/libobjc.A.dylib", RTLD_LAZY | RTLD_LOCAL);
    void* metal_lib = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                             RTLD_LAZY | RTLD_LOCAL);
    if (objc_lib == nullptr || metal_lib == nullptr) return loaded;

    auto register_name =
        reinterpret_cast<RegisterNameFn>(dlsym(objc_lib, "sel_registerName"));
    void* msg_send_sym = dlsym(objc_lib, "objc_msgSend");
    if (register_name == nullptr || msg_send_sym == nullptr) return loaded;

    loaded->msg_send_usize = reinterpret_cast<MsgSendUsizeFn>(msg_send_sym);
    loaded->current_allocated_size_sel = register_name("currentAllocatedSize");

    auto copy_all = reinterpret_cast<CopyAllDevicesFn>(
        dlsym(metal_lib, "MTLCopyAllDevices"));
    if (copy_all != nullptr) {
      if (void* array = copy_all()) {
        Sel count_sel = register_name("count");
        Sel object_at_index_sel = register_name("objectAtIndex:");
        auto msg_send_idx = reinterpret_cast<MsgSendIdxFn>(msg_send_sym);
        const size_t count = loaded->msg_send_usize(array, count_sel);
        for (size_t i = 0; i < count; ++i) {
          if (void* dev = msg_send_idx(array, object_at_index_sel, i)) {
            loaded->devices.push_back(dev);
          }
        }
      }
    }
    if (loaded->devices.empty()) {
      auto create_default = reinterpret_cast<CreateDefaultDeviceFn>(
          dlsym(metal_lib, "MTLCreateSystemDefaultDevice"));
      if (create_default != nullptr) {
        if (void* dev = create_default()) {
          loaded->devices.push_back(dev);
        }
      }
    }
    return loaded;
  }();

  if (!api->available()) return std::nullopt;
  int64_t total_bytes = 0;
  for (void* dev : api->devices) {
    total_bytes += static_cast<int64_t>(
        api->msg_send_usize(dev, api->current_allocated_size_sel));
  }
  return total_bytes;
}

GpuMemorySnapshot ReadAppleGpuMemorySnapshot() {
  return CombineAppleGpuMemory(ReadAppleGpuLedger(),
                               QueryAppleMetalAllocatedBytes());
}
#endif  // defined(__APPLE__)

#if defined(_WIN32)
std::vector<WindowsAdapterVideoMemory> ReadWindowsAdapterVideoMemory() {
  using Microsoft::WRL::ComPtr;
  using CreateDXGIFactory1Fn = HRESULT(WINAPI*)(REFIID, void**);

  struct DxgiAdapterEntry {
    ComPtr<IDXGIAdapter3> adapter;
    std::string description;
  };

  // Intentionally leaked so COM objects are not released during static
  // destruction after dxgi.dll may have been unloaded.
  static const std::vector<DxgiAdapterEntry>* const adapters = []() {
    auto* list = new std::vector<DxgiAdapterEntry>();
    HMODULE dxgi_module = LoadLibraryA("dxgi.dll");
    if (dxgi_module == nullptr) return list;
    auto create_factory = reinterpret_cast<CreateDXGIFactory1Fn>(
        GetProcAddress(dxgi_module, "CreateDXGIFactory1"));
    if (create_factory == nullptr) return list;

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(create_factory(IID_PPV_ARGS(&factory)))) return list;

    ComPtr<IDXGIAdapter1> adapter1;
    for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter1)); ++i) {
      ComPtr<IDXGIAdapter3> adapter3;
      if (FAILED(adapter1.As(&adapter3))) continue;
      DxgiAdapterEntry entry;
      entry.adapter = std::move(adapter3);
      DXGI_ADAPTER_DESC1 desc = {};
      if (SUCCEEDED(adapter1->GetDesc1(&desc))) {
        const int utf8_len = WideCharToMultiByte(
            CP_UTF8, 0, desc.Description, -1, nullptr, 0, nullptr, nullptr);
        if (utf8_len > 1) {
          entry.description.resize(static_cast<size_t>(utf8_len - 1));
          WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                              entry.description.data(), utf8_len, nullptr,
                              nullptr);
        }
      }
      if (entry.description.empty()) {
        entry.description = absl::StrCat("Adapter ", i);
      }
      list->push_back(std::move(entry));
    }
    return list;
  }();

  std::vector<WindowsAdapterVideoMemory> result;
  for (const DxgiAdapterEntry& entry : *adapters) {
    DXGI_QUERY_VIDEO_MEMORY_INFO local_info = {};
    DXGI_QUERY_VIDEO_MEMORY_INFO non_local_info = {};
    const bool local_ok = SUCCEEDED(entry.adapter->QueryVideoMemoryInfo(
        /*NodeIndex=*/0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info));
    const bool non_local_ok = SUCCEEDED(entry.adapter->QueryVideoMemoryInfo(
        /*NodeIndex=*/0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local_info));
    if (!local_ok && !non_local_ok) continue;
    result.push_back(WindowsAdapterVideoMemory{
        .description = entry.description,
        .local_bytes =
            local_ok ? static_cast<int64_t>(local_info.CurrentUsage) : 0,
        .non_local_bytes =
            non_local_ok ? static_cast<int64_t>(non_local_info.CurrentUsage)
                         : 0,
    });
  }
  return result;
}

GpuMemorySnapshot ReadWindowsGpuMemorySnapshot() {
  return CombineWindowsGpuMemory(ReadWindowsAdapterVideoMemory());
}
#endif  // defined(_WIN32)

}  // namespace internal

bool GpuMemorySnapshot::IsSupported() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  return true;
#else
  return false;
#endif
}

GpuMemorySnapshot GetGpuMemorySnapshot(bool include_external_tools) {
#if defined(__ANDROID__)
  const int64_t external_gpumem_bytes =
      include_external_tools ? internal::QueryAndroidDumpsysGpuMemBytes()
                             : GpuMemorySnapshot::kUnavailable;
  return internal::ReadAndroidProcGpuMemorySnapshot("/proc/self",
                                                    external_gpumem_bytes);
#elif defined(__linux__)
  (void)include_external_tools;
  return internal::ReadLinuxProcGpuMemorySnapshot(
      "/proc/self", internal::QueryLinuxNvmlProcessBytes());
#elif defined(__APPLE__)
  (void)include_external_tools;
  return internal::ReadAppleGpuMemorySnapshot();
#elif defined(_WIN32)
  (void)include_external_tools;
  return internal::ReadWindowsGpuMemorySnapshot();
#else
  (void)include_external_tools;
  return GpuMemorySnapshot{};
#endif
}

}  // namespace benchmark
}  // namespace tflite
