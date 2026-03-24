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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#ifdef __linux__
#include <malloc.h>
#include <sys/resource.h>
#include <sys/time.h>
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
  std::ifstream status_file("/proc/self/status");
  if (!status_file.is_open()) {
    return -1;
  }
  std::string line;
  while (std::getline(status_file, line)) {
    if (line.rfind("VmSwap:", 0) == 0) {
      std::stringstream ss(line);
      std::string key;
      int64_t value_kb;
      // The line format is "VmSwap:    1234 kB"
      // We can extract the key ("VmSwap:") and the numeric value ("1234").
      ss >> key >> value_kb;
      if (!ss.fail()) {
        return value_kb;
      } else {
        return -1;  // Indicate parsing error
      }
    }
  }
  // If the VmSwap line is not found, it means 0 swap is being used.
  return 0;
}
#endif

}  // namespace

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
