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
#include "tensorflow/lite/profiling/memory_info.h"

#include <stddef.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <string>
#include <unordered_set>

#ifdef __linux__
#include <dirent.h>
#include <malloc.h>
#include <stdio.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#elif defined(_WIN32)
#include <windows.h>
// psapi must be included after windows.h.
#include <psapi.h>
#endif

namespace tflite {
namespace profiling {
namespace memory {

const size_t MemoryUsage::kValueNotSet = 0;

namespace {

#if defined(__linux__)
// Returns the current VM swap in kilobytes on Linux.
int64_t GetCurrentVmSwapKb() {
  FILE* status_file = fopen("/proc/self/status", "r");
  if (!status_file) {
    return -1;
  }
  char line[256];
  while (fgets(line, sizeof(line), status_file)) {
    if (strncmp(line, "VmSwap:", 7) == 0) {
      int64_t value_kb = 0;
      if (sscanf(line + 7, "%" SCNd64, &value_kb) == 1) {
        fclose(status_file);
        return value_kb;
      } else {
        fclose(status_file);
        return -1;  // Indicate parsing error
      }
    }
  }
  fclose(status_file);
  // If the VmSwap line is not found, it means 0 swap is being used.
  return 0;
}

#if !defined(__ANDROID__)
int64_t ParseDrmMemoryValueKb(const char* line) {
  const char* colon = strchr(line, ':');
  if (colon == nullptr) {
    return 0;
  }
  int64_t value = 0;
  char unit[16] = {0};
  int matched = sscanf(colon + 1, "%" SCNd64 " %15s", &value, unit);
  if (matched < 1 || value <= 0) {
    return 0;
  }
  if (strcmp(unit, "KiB") == 0 || strcmp(unit, "kB") == 0) {
    return value;
  }
  if (strcmp(unit, "MiB") == 0 || strcmp(unit, "MB") == 0) {
    return value * 1024;
  }
  return value / 1024;
}
#endif  // !defined(__ANDROID__)
#endif  // defined(__linux__)

}  // namespace

int64_t GetCurrentCpuRssKb() {
#if defined(__linux__)
  FILE* rollup_file = fopen("/proc/self/smaps_rollup", "r");
  if (!rollup_file) {
    return -1;
  }
  char line[256];
  while (fgets(line, sizeof(line), rollup_file)) {
    if (strncmp(line, "Rss:", 4) == 0) {
      int64_t value_kb = -1;
      if (sscanf(line + 4, "%" SCNd64, &value_kb) == 1) {
        fclose(rollup_file);
        return value_kb;
      }
      fclose(rollup_file);
      return -1;
    }
  }
  fclose(rollup_file);
#endif  // defined(__linux__)
  return -1;
}

int64_t GetCurrentGpuMemoryKb() {
#if defined(__linux__)
#if defined(__ANDROID__)
  if (access("/system/bin/dumpsys", X_OK) == 0) {
    FILE* pipe = popen("/system/bin/dumpsys gpu --gpumem 2>/dev/null", "r");
    if (pipe != nullptr) {
      const std::string proc_prefix =
          "Proc " + std::to_string(getpid()) + " total:";
      char buf[256];
      int64_t gpu_kb = 0;
      while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        const char* pos = strstr(buf, proc_prefix.c_str());
        if (pos != nullptr) {
          int64_t bytes = 0;
          if (sscanf(pos + proc_prefix.size(), "%" SCNd64, &bytes) == 1 &&
              bytes >= 0) {
            gpu_kb = bytes / 1024;
          }
          break;
        }
      }
      pclose(pipe);
      return gpu_kb;
    }
  }
#else
  DIR* fdinfo_dir = opendir("/proc/self/fdinfo");
  if (fdinfo_dir != nullptr) {
    std::unordered_set<std::string> seen_drm_clients;
    int64_t drm_total_kb = 0;
    struct dirent* entry = nullptr;
    while ((entry = readdir(fdinfo_dir)) != nullptr) {
      if (entry->d_name[0] == '.') {
        continue;
      }
      std::string path =
          std::string("/proc/self/fdinfo/") + std::string(entry->d_name);
      FILE* fdinfo_file = fopen(path.c_str(), "r");
      if (!fdinfo_file) {
        continue;
      }
      std::string client_id;
      std::string pdev;
      int64_t resident_kb = 0;
      int64_t memory_kb = 0;
      bool has_resident_tag = false;
      char line[256];
      while (fgets(line, sizeof(line), fdinfo_file)) {
        if (strncmp(line, "drm-client-id:", 14) == 0) {
          client_id = line + 14;
        } else if (strncmp(line, "drm-pdev:", 9) == 0) {
          pdev = line + 9;
        } else if (strncmp(line, "drm-resident-", 13) == 0) {
          has_resident_tag = true;
          resident_kb += ParseDrmMemoryValueKb(line);
        } else if (strncmp(line, "drm-memory-", 11) == 0) {
          memory_kb += ParseDrmMemoryValueKb(line);
        }
      }
      fclose(fdinfo_file);
      if (!client_id.empty()) {
        std::string client_key = pdev + ":" + client_id;
        if (seen_drm_clients.insert(client_key).second) {
          drm_total_kb += has_resident_tag ? resident_kb : memory_kb;
        }
      }
    }
    closedir(fdinfo_dir);
    return drm_total_kb;
  }
#endif  // defined(__ANDROID__)
  return 0;
#else
  return -1;
#endif  // defined(__linux__)
}

bool MemoryUsage::IsSupported() {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
  return true;
#endif
  return false;
}

MemoryUsage GetMemoryUsage() {
  MemoryUsage result;
#ifdef __linux__
  rusage res;
  if (getrusage(RUSAGE_SELF, &res) == 0) {
    result.mem_footprint_kb = res.ru_maxrss;
    int64_t vm_swap_kb = GetCurrentVmSwapKb();
    if (vm_swap_kb >= 0) {
      result.private_footprint_bytes = (vm_swap_kb + res.ru_maxrss) * 1024;
    }
  }
#if defined(__NO_MALLINFO__) || !defined(__GLIBC__) ||         \
    defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
  result.total_allocated_bytes = -1;
  result.in_use_allocated_bytes = -1;
#elif __GLIBC_MINOR__ >= 33
  const auto mem = mallinfo2();
  result.total_allocated_bytes = mem.arena;
  result.in_use_allocated_bytes = mem.uordblks;
#else
  const auto mem = mallinfo();
  result.total_allocated_bytes = mem.arena;
  result.in_use_allocated_bytes = mem.uordblks;
#endif  // defined(__NO_MALLINFO__) || !defined(__GLIBC__) || \
        // defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) ||
        // defined(THREAD_SANITIZER)
#elif defined(__APPLE__)
  struct task_vm_info vm_info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  auto status = task_info(mach_task_self(), TASK_VM_INFO,
                          reinterpret_cast<task_info_t>(&vm_info), &count);
  if (status == KERN_SUCCESS) {
    result.mem_footprint_kb =
        static_cast<int64_t>(vm_info.phys_footprint / 1024.0);
    // TODO: b/421171145 - Consider subtracting shared_resident_kb.
    result.private_footprint_bytes = vm_info.phys_footprint;
  }
  struct mstats stats = mstats();
  result.total_allocated_bytes = stats.bytes_total;
  result.in_use_allocated_bytes = stats.bytes_used;
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX process_memory_counters;
  HANDLE process_handle = GetCurrentProcess();
  if (process_handle != nullptr &&
      GetProcessMemoryInfo(process_handle,
                           (PROCESS_MEMORY_COUNTERS*)&process_memory_counters,
                           sizeof(process_memory_counters))) {
    result.mem_footprint_kb = process_memory_counters.WorkingSetSize / 1024;
    result.private_footprint_bytes = process_memory_counters.PrivateUsage;
  } else {
    result.mem_footprint_kb = -1;
    result.private_footprint_bytes = -1;
  }
  CloseHandle(process_handle);
  result.total_allocated_bytes = -1;
  result.in_use_allocated_bytes = -1;
#ifdef USE_WIN32_HEAP_SUMMARY
  HANDLE process_heap = GetProcessHeap();
  if (process_heap != nullptr) {
    HEAP_SUMMARY heap_summary;
    heap_summary.cb = sizeof(heap_summary);
    if (HeapSummary(process_heap, 0, &heap_summary)) {
      result.total_allocated_bytes = heap_summary.cbCommitted;
      result.in_use_allocated_bytes = heap_summary.cbAllocated;
    }
  }
#endif  // USE_WIN32_HEAP_SUMMARY
#endif  // __linux__
  return result;
}

void MemoryUsage::AllStatsToStream(std::ostream* stream) const {
  *stream << "max resident set size/physical footprint = "
          << mem_footprint_kb / 1000.0 << " MB, total non-mmapped heap size = "
          << total_allocated_bytes / 1000.0 / 1000.0
          << " MB, in-use heap size = "
          << in_use_allocated_bytes / 1000.0 / 1000.0
          << " MB, private footprint = "
          << private_footprint_bytes / 1000.0 / 1000.0 << " MB";
}

}  // namespace memory
}  // namespace profiling
}  // namespace tflite
