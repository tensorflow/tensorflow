/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FORK_JOIN_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FORK_JOIN_H_

#include "tensorflow/core/platform/types.h"

extern "C" {

// Dispatches 'num_partitions' parallel calls to 'function_ptr' and joins
// threads before returning. See comments in runtime_fork_join.cc for details.
extern void __xla_cpu_runtime_ParallelForkJoin(
    void* result_ptr, const void* run_options_ptr, const void** params,
    void** buffer_table, tensorflow::uint64* prof_counters,
    int32_t num_partitions, int64_t* partitions, int32_t num_partitioned_dims,
    void* function_ptr);

}  // extern "C"

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FORK_JOIN_H_
