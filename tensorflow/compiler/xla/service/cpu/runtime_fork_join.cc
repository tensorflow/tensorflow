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

#include "absl/base/dynamic_annotations.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"

using ComputeFunctionType = void (*)(void*, const void*, const void**, void**,
                                     void*, int64_t*, uint64_t*);

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
ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ParallelForkJoin(
    void* result_ptr, const void* run_options_ptr, const void** params,
    void** buffer_table, void* status, uint64_t* prof_counters,
    int32_t num_partitions, int64_t* partitions, int32_t num_partitioned_dims,
    void* function_ptr) {
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
  const int64_t stride = 2 * num_partitioned_dims;

  std::vector<XlaCustomCallStatus> statuses(num_partitions);

  // Dispatch 'num_partitions - 1' compute functions to run in parallel.
  tensorflow::BlockingCounter bc(num_partitions - 1);
  for (int32_t i = 1; i < num_partitions; ++i) {
    const int64_t offset = i * stride;
    run_options->intra_op_thread_pool()->enqueueNoNotification(
        [i, function, result_ptr, run_options_ptr, buffer_table, prof_counters,
         partitions, offset, &bc, &statuses]() {
          function(result_ptr, run_options_ptr, nullptr, buffer_table,
                   &statuses[i], &partitions[offset], prof_counters);
          bc.DecrementCount();
          VLOG(3) << "ParallelForkJoin partition " << i << " done.";
        });
  }

  // Call first compute function inline.
  function(result_ptr, run_options_ptr, params, buffer_table, &statuses[0],
           &partitions[0], prof_counters);
  VLOG(3) << "ParallelForkJoin partition 0 done.";
  bc.Wait();

  // Collect all error messages (if any).
  std::vector<std::pair<int32_t, absl::string_view>> error_messages;
  for (int32_t i = 0; i < num_partitions; ++i) {
    std::optional<absl::string_view> msg =
        xla::CustomCallStatusGetMessage(&statuses[i]);
    if (msg) {
      error_messages.emplace_back(i, *msg);
    }
  }

  if (!error_messages.empty()) {
    // Join all error messages into a single string to serve as the message for
    // the returned status.
    std::string error_message = absl::StrJoin(
        error_messages, "\n",
        [](std::string* out, std::pair<int32_t, absl::string_view> p) {
          int32_t idx = p.first;
          absl::string_view msg = p.second;
          absl::StrAppend(out,
                          absl::StrFormat("Partition %d error: %s", idx, msg));
        });
    XlaCustomCallStatusSetFailure(
        reinterpret_cast<XlaCustomCallStatus*>(status), error_message.data(),
        error_message.length());
  }
  VLOG(2) << "ParallelForkJoin EXIT";
}
