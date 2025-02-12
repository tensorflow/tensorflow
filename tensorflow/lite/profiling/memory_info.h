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
#ifndef TENSORFLOW_LITE_PROFILING_MEMORY_INFO_H_
#define TENSORFLOW_LITE_PROFILING_MEMORY_INFO_H_

#include <stddef.h>

#include <cstdint>
#include <ostream>
#include <sstream>

namespace tflite {
namespace profiling {
namespace memory {

struct MemoryUsage {
  static const size_t kValueNotSet;

  // Indicates whether obtaining memory usage is supported on the platform, thus
  // indicating whether the values defined in this struct make sense or not.
  static bool IsSupported();

  MemoryUsage()
      : mem_footprint_kb(kValueNotSet),
        total_allocated_bytes(kValueNotSet),
        in_use_allocated_bytes(kValueNotSet) {}

  // The memory footprint (in kilobytes).
  //
  // For Linux:
  // This is the maximum memory size (in kilobytes) occupied by an OS process
  // that is held in main memory (RAM). Such memory usage information is
  // generally referred as resident set size (rss). This is an alias to
  // rusage::ru_maxrss.
  //
  // For Mac:
  // This is the physical memory footprint (in kilobytes). This is an alias to
  // task_vm_info::phys_footprint.
  // Per kern/task.c, physical footprint is the sum of:
  //    + (internal - alternate_accounting)
  //    + (internal_compressed - alternate_accounting_compressed)
  //    + iokit_mapped
  //    + purgeable_nonvolatile
  //    + purgeable_nonvolatile_compressed
  //    + page_table
  int64_t mem_footprint_kb;

  // Total non-mmapped heap space allocated from system in bytes.
  // For Linux, this is an alias to mallinfo::arena.
  // For Mac, this is an alias to mstats::bytes_total
  //
  // This does not count mmapped heap space, nor does it count non-heap
  // uses of memory such as other mmapped space, thread stacks, globals,
  // code, etc.
  size_t total_allocated_bytes;

  // Total allocated (including mmapped) heap bytes that are in use
  // (i.e. excluding those have been freed).
  // For Linux, this is an alias to mallinfo::uordblks.
  // For Mac, this is an alias to mstats::bytes_used
  //
  // This does not count non-heap uses of mmap, nor does it count other
  // non-heap uses of memory such as thread stacks, globals, code, etc.
  size_t in_use_allocated_bytes;

  MemoryUsage operator+(MemoryUsage const& obj) const {
    MemoryUsage res;
    res.mem_footprint_kb = mem_footprint_kb + obj.mem_footprint_kb;
    res.total_allocated_bytes =
        total_allocated_bytes + obj.total_allocated_bytes;
    res.in_use_allocated_bytes =
        in_use_allocated_bytes + obj.in_use_allocated_bytes;
    return res;
  }

  MemoryUsage operator-(MemoryUsage const& obj) const {
    MemoryUsage res;
    res.mem_footprint_kb = mem_footprint_kb - obj.mem_footprint_kb;
    res.total_allocated_bytes =
        total_allocated_bytes - obj.total_allocated_bytes;
    res.in_use_allocated_bytes =
        in_use_allocated_bytes - obj.in_use_allocated_bytes;
    return res;
  }

  void AllStatsToStream(std::ostream* stream) const;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const MemoryUsage& obj) {
    obj.AllStatsToStream(&stream);
    return stream;
  }
};

// Return the memory usage from the system.
// Note: this currently only works on Linux-based and Apple systems.
MemoryUsage GetMemoryUsage();

}  // namespace memory
}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_MEMORY_INFO_H_
