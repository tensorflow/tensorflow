/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_SINGLE_THREADED_EXECUTOR_H_
#define TENSORFLOW_CORE_KERNELS_DATA_SINGLE_THREADED_EXECUTOR_H_

#include "tensorflow/core/common_runtime/executor.h"

namespace tensorflow {
namespace data {

// Creates a new `Executor` for executing `graph` synchronously on the caller
// thread.
//
// NOTE(mrry): The returned executor is optimized to impose low overhead on
// graphs that perform a small amount of work (e.g. <15us of work per graph on
// present architectures). It eschews concurrency, because issuing work to
// multiple threads can dominate the cost of executing small ops synchronously,
// and because contention in the executor data structures can reduce throughput
// (in terms of ops executed per unit time).
//
// However, the current implementation has the following limitations:
//
// 1. Reference-typed tensors are not supported and will not be supported in
//    future.
// 2. Graphs with control flow (containing "Switch" and "Merge" nodes) are not
//    currently supported. The current plan is to extend support to "functional"
//    control flow after the TensorFlow APIs transition to building graphs in
//    that form (e.g. `tf.cond_v2()`).
// 3. Partitioned graphs (containing "_Recv" nodes) are not currently supported.
//    The present implementation executes kernels one at a time in topological
//    order, and cannot currently distinguish between disconnected subgraphs
//    that are logically connected by subgraphs on a different device.
// 4. Memory logging is not currently supported.
// 5. Allocation forwarding is not currently supported.
// 6. Non-default device contexts are not currently supported. In effect, this
//    limits the executor to CPU devices.
// 7. Ops that rely on `OpKernelContext::slice_reader_cache()` being non-null
//    are not currently supported.
//
// The single-threaded executor is primarily suitable for executing simple
// TensorFlow functions, such as one might find in a `tf.data` pipeline.
Status NewSingleThreadedExecutor(const LocalExecutorParams& params,
                                 const Graph& graph, Executor** executor);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_SINGLE_THREADED_EXECUTOR_H_
