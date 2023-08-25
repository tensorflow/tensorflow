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

#ifdef __linux__
#include <malloc.h>
#include <sys/resource.h>
#include <sys/time.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#endif

namespace tflite {
namespace profiling {
namespace memory {

const size_t MemoryUsage::kValueNotSet = 0;

bool MemoryUsage::IsSupported() {
#if defined(__linux__) || defined(__APPLE__)
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
  }
#if defined(__NO_MALLINFO__)
  result.total_allocated_bytes = -1;
  result.in_use_allocated_bytes = -1;
#elif defined(__GLIBC__) && __GLIBC_MINOR__ >= 33
  const auto mem = mallinfo2();
  result.total_allocated_bytes = mem.arena;
  result.in_use_allocated_bytes = mem.uordblks;
#else
  const auto mem = mallinfo();
  result.total_allocated_bytes = mem.arena;
  result.in_use_allocated_bytes = mem.uordblks;
#endif  // defined(__NO_MALLINFO__)
#elif defined(__APPLE__)
  struct task_vm_info vm_info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  auto status = task_info(mach_task_self(), TASK_VM_INFO,
                          reinterpret_cast<task_info_t>(&vm_info), &count);
  if (status == KERN_SUCCESS) {
    result.mem_footprint_kb =
        static_cast<int64_t>(vm_info.phys_footprint / 1024.0);
  }
  struct mstats stats = mstats();
  result.total_allocated_bytes = stats.bytes_total;
  result.in_use_allocated_bytes = stats.bytes_used;
#endif  // __linux__
  return result;
}

void MemoryUsage::AllStatsToStream(std::ostream* stream) const {
  *stream << "max resident set size/physical footprint = "
          << mem_footprint_kb / 1024.0 << " MB, total malloc-ed size = "
          << total_allocated_bytes / 1024.0 / 1024.0
          << " MB, in-use allocated/mmapped size = "
          << in_use_allocated_bytes / 1024.0 / 1024.0 << " MB";
}

}  // namespace memory
}  // namespace profiling
}  // namespace tflite
