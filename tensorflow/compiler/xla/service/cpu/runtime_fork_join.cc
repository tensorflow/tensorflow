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

#include "tensorflow/compiler/xla/service/cpu/runtime_fork_join.h"

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::int32;
using tensorflow::int64;
using tensorflow::uint64;

using ComputeFunctionType = void (*)(void*, const void*, const void**, void**,
                                     int64*, uint64*);

// Dispatches 'num_partitions - 1' calls to 'function_ptr' in parallel.
// Calls 'function_ptr' for first partition inline.
// Uses blocking counter to synchronize threads after parallel calls complete.
//
// The 'partitions' array has a total number of elements equal to
// 'num_partitions * num_partitioned_dims * 2' (the '2' is necessary to specify
// dimension start and limit indices).
//
// The 'partitions' array layout stores array elements in memory with dimension
// start limit as the most-minor dimension, followed by dimension, then
// partition.
//
// EX: Layout of 'partitions' array with 'num_partitions = 2', and
//     'num_partitioned_dims = 3'
//
//   [partition0_dim0_start]
//   [partition0_dim0_limit]
//   [partition0_dim1_start]
//   [partition0_dim1_limit]
//   [partition0_dim2_start]
//   [partition0_dim2_limit]
//   [partition1_dim0_start]
//   [partition1_dim0_limit]
//   [partition1_dim1_start]
//   [partition1_dim1_limit]
//   [partition1_dim2_start]
//   [partition1_dim2_limit]
//
TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ParallelForkJoin(
    void* result_ptr, const void* run_options_ptr, const void** params,
    void** buffer_table, uint64* prof_counters, int32 num_partitions,
    int64* partitions, int32 num_partitioned_dims, void* function_ptr) {
  VLOG(2) << "ParallelForkJoin ENTRY"
          << " num_partitions: " << num_partitions
          << " num_partitioned_dims: " << num_partitioned_dims;
  CHECK_EQ(params, nullptr);
  CHECK_GT(num_partitions, 1);
  CHECK_GT(num_partitioned_dims, 0);
  CHECK_NE(function_ptr, nullptr);
  CHECK_NE(partitions, nullptr);
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  CHECK_NE(run_options, nullptr);
  CHECK_NE(run_options->intra_op_thread_pool(), nullptr);

  ComputeFunctionType function =
      reinterpret_cast<ComputeFunctionType>(function_ptr);
  // Compute partition stride in 'partitions' array.
  const int64 stride = 2 * num_partitioned_dims;

  // Dispatch 'num_partitions - 1' compute functions to run in parallel.
  tensorflow::BlockingCounter bc(num_partitions - 1);
  for (int32 i = 1; i < num_partitions; ++i) {
    const int64 offset = i * stride;
    run_options->intra_op_thread_pool()->enqueueNoNotification(
        [i, function, result_ptr, run_options_ptr, buffer_table, prof_counters,
         partitions, offset, &bc]() {
          function(result_ptr, run_options_ptr, nullptr, buffer_table,
                   &partitions[offset], prof_counters);
          bc.DecrementCount();
          VLOG(3) << "ParallelForkJoin partition " << i << " done.";
        });
  }

  // Call first compute function inline.
  function(result_ptr, run_options_ptr, params, buffer_table, &partitions[0],
           prof_counters);
  VLOG(3) << "ParallelForkJoin partition 0 done.";
  bc.Wait();
  VLOG(2) << "ParallelForkJoin EXIT";
}
