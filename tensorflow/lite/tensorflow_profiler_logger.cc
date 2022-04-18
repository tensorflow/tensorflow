/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tensorflow_profiler_logger.h"

#include <malloc.h>

#include <algorithm>
#include <string>

#include "tensorflow/core/profiler/lib/traceme.h"

namespace {

struct Statistics {
  uint64_t total_bytes_allocated = 0LL;
  uint64_t peak_bytes_in_use = 0LL;
};
Statistics g_stat;

std::string AddTraceMeInternal(bool is_allocating,
                               const std::string& allocator_name,
                               int64_t tensor_id, const std::string& name,
                               const std::string& dims,
                               int64_t allocation_bytes,
                               int64_t requested_bytes) {
  if (is_allocating) {
    g_stat.total_bytes_allocated += allocation_bytes;
  } else {
    g_stat.total_bytes_allocated -= allocation_bytes;
  }
  g_stat.peak_bytes_in_use =
      std::max(g_stat.peak_bytes_in_use, g_stat.total_bytes_allocated);
  int64_t total_bytes_allocated = g_stat.total_bytes_allocated;
  int64_t peak_bytes_in_use = g_stat.peak_bytes_in_use;

  std::string res = tensorflow::profiler::TraceMeEncode(
      is_allocating ? "MemoryAllocation" : "MemoryDeallocation",
      // Note that all of these fields are necessary for profiling UI.
      {{"allocator_name", allocator_name},  // name shown on 'Memory ID'
       {"bytes_allocated", total_bytes_allocated},
       {"peak_bytes_in_use", peak_bytes_in_use},
       {"requested_bytes", requested_bytes},
       {"allocation_bytes", allocation_bytes},
       // Note: addr is used as a key to match alloc and dealloc.
       {"addr", tensor_id},
       // Note that we're using tensor.name not op name here.
       {"tf_op", name},
       {"shape", dims}});
  // Note: bytes_reserved, fragmentation, data_type, region_type, id
  // can be potentially useful but not added.
  return res;
}

}  // namespace

void AddTraceMe(bool is_allocating, int64_t allocation_bytes, int64_t tensor_id,
                const std::string& name, const std::string& dims) {
  if (is_allocating && allocation_bytes == 0) return;
  int64_t requested_bytes = is_allocating ? allocation_bytes : 0;
  const std::string allocator_name = "_tflite_native_dynamic";

  tensorflow::profiler::TraceMe::InstantActivity(
      [is_allocating, allocator_name, tensor_id, name, dims, allocation_bytes,
       requested_bytes]() {
        return AddTraceMeInternal(is_allocating, allocator_name, tensor_id,
                                  name, dims, allocation_bytes,
                                  requested_bytes);
      },
      /*level=*/tensorflow::profiler::TraceMeLevel::kInfo);
}
