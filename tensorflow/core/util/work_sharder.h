#ifndef TENSORFLOW_UTIL_WORK_SHARDER_H_
#define TENSORFLOW_UTIL_WORK_SHARDER_H_

#include <functional>

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

// Shards the "total" unit of work assuming each unit of work having
// roughly "cost_per_unit". Each unit of work is indexed 0, 1, ...,
// total - 1. Each shard contains 1 or more units of work and the
// total cost of each shard is roughly the same. The total number of
// shards is no more than num_workers. The calling thread and the
// "workers" are used to compute each shard (calling work(start,
// limit). A common configuration is that "workers" is a thread pool
// with "num_workers" threads.
//
// "work" should be a callable taking (int64, int64) arguments.
// work(start, limit) computes the work units from [start,
// limit), i.e., [start, limit) is a shard.
//
// REQUIRES: num_workers >= 0
// REQUIRES: workers != nullptr
// REQUIRES: total >= 0
// REQUIRES: cost_per_unit >= 0
void Shard(int num_workers, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work);

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_WORK_SHARDER_H_
