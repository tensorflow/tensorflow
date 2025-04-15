/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_
#define XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_

#include <stdint.h>

#include "unsupported/Eigen/CXX11/Tensor"

extern "C" {

// Each entry in 'values' represents a 3-dimensional shape with dimensions
// [a, b, c]. The 'b' dimension of each shape is sorted into ascending order
// according to the results of comparisons using the provided 'less_than'
// function. 'values_count' must be > 0 and specifies the number of entries in
// 'values' and 'values_primitive_type_size_in_bytes'. The size of the primitive
// type of the i-th shape has exactly 'values_primitive_type_size_in_bytes[i]'
// bytes. 'is_stable' specifies whether the sorting should be stable.
// 'run_options' and 'prof_counters' are passed through to the less-than
// function, which expects the following arguments:
// - pointer to the return value buffer (char*)
// - xla::ExecutableRunOptions = 'run_options' (char*)
// - pointers to the parameter buffers (char**)
// - pointers to the buffer tables = nullptr for thread local functions (char**)
// - profile counters = 'prof_counters' (int64_t*)
extern void __xla_cpu_runtime_KeyValueSort(
    int64_t a, int64_t b, int64_t c, char** values, int32_t values_count,
    int32_t* values_primitive_type_size_in_bytes, bool is_stable,
    char* run_options, int64_t* prof_counters,
    void (*less_than)(char*, char*, char**, char**, int64_t*));
}

#endif  // XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_
