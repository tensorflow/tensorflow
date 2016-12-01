/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_UTIL_WORK_SHARDER_H_
#define TENSORFLOW_UTIL_WORK_SHARDER_H_

#include <functional>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Shards the "total" unit of work assuming each unit of work having
// roughly "cost_per_unit". Each unit of work is indexed 0, 1, ...,
// total - 1. Each shard contains 1 or more units of work and the
// total cost of each shard is roughly the same. The calling thread and the
// "workers" are used to compute each shard (calling work(start,
// limit). A common configuration is that "workers" is a thread pool
// with at least "max_parallelism" threads.
//
// "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
// if not CPU-bound) to complete a unit of work. Overestimating creates too
// many shards and CPU time will be dominated by per-shard overhead, such as
// Context creation. Underestimating may not fully make use of the specified
// parallelism.
//
// "work" should be a callable taking (int64, int64) arguments.
// work(start, limit) computes the work units from [start,
// limit), i.e., [start, limit) is a shard.
//
// REQUIRES: max_parallelism >= 0
// REQUIRES: workers != nullptr
// REQUIRES: total >= 0
// REQUIRES: cost_per_unit >= 0
void Shard(int max_parallelism, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work);

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_WORK_SHARDER_H_
