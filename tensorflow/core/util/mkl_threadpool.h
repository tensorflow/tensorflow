
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
#define TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
#ifdef INTEL_MKL

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mkldnn.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/threadpool.h"
#define EIGEN_USE_THREADS
#ifdef ENABLE_MKLDNN_THREADPOOL
using dnnl::stream_attr;
using dnnl::threadpool_iface;

namespace tensorflow {

// Divide 'n' units of work equally among 'teams' threads. If 'n' is not
// divisible by 'teams' and has a remainder 'r', the first 'r' teams have one
// unit of work more than the rest. Returns the range of work that belongs to
// the team 'tid'.
// Parameters
//   n        Total number of jobs.
//   team     Number of workers.
//   tid      Current thread_id.
//   n_start  start of range operated by the thread.
//   n_end    end of the range operated by the thread.

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T* n_start, T* n_end) {
  if (team <= 1 || n == 0) {
    *n_start = 0;
    *n_end = n;
    return;
  }
  T min_per_team = n / team;
  T remainder = n - min_per_team * team;  // i.e., n % teams.
  *n_start = tid * min_per_team + std::min(tid, remainder);
  *n_end = *n_start + min_per_team + (tid < remainder);
}

struct MklDnnThreadPool : public dnnl::threadpool_iface {
  MklDnnThreadPool() = default;

  MklDnnThreadPool(OpKernelContext* ctx)
      : eigen_interface_(ctx->device()
                             ->tensorflow_cpu_worker_threads()
                             ->workers->AsEigenThreadPool()) {}
  virtual int get_num_threads() override {
    return eigen_interface_->NumThreads();
  }
  virtual bool get_in_parallel() override {
    return (eigen_interface_->CurrentThreadId() != -1) ? true : false;
  }
  virtual uint64_t get_flags() override { return ASYNCHRONOUS; }
  virtual void parallel_for(int n,
                            const std::function<void(int, int)>& fn) override {
    // Should never happen (handled by DNNL)
    if (n == 0) return;

    // Should never happen (handled by DNNL)
    if (n == 1) {
      fn(0, 1);
      return;
    }

    int nthr = get_num_threads();
    int njobs = std::min(n, nthr);
    for (int i = 0; i < njobs; i++) {
      eigen_interface_->ScheduleWithHint(
          [i, n, njobs, fn]() {
            int start, end;
            balance211(n, njobs, i, &start, &end);
            for (int j = start; j < end; j++) fn(j, n);
          },
          i, i + 1);
    }
  }
  ~MklDnnThreadPool() {}

 private:
  Eigen::ThreadPoolInterface* eigen_interface_ = nullptr;
};

class MklDnnThreadPoolWrapper {
 public:
  static MklDnnThreadPoolWrapper& GetInstance() {
    static MklDnnThreadPoolWrapper instance_;
    return instance_;
  }
  MklDnnThreadPool* CreateThreadPoolPtr(OpKernelContext* ctx) {
    if (threadpool_map_.empty() ||
        threadpool_map_.find(ctx->device()) == threadpool_map_.end()) {
      auto tp_iface = new MklDnnThreadPool(ctx);
      threadpool_map_.emplace(std::make_pair(ctx->device(), tp_iface));
      return tp_iface;
    } else {
      auto entry = threadpool_map_.find(ctx->device());
      return entry->second;
    }
  }

 private:
  std::unordered_map<DeviceBase*, MklDnnThreadPool*> threadpool_map_;
  MklDnnThreadPoolWrapper() {}
  MklDnnThreadPoolWrapper(const MklDnnThreadPoolWrapper&) = delete;
  MklDnnThreadPoolWrapper& operator=(const MklDnnThreadPoolWrapper&) = delete;
  ~MklDnnThreadPoolWrapper() {
    for (auto& tp : threadpool_map_) {
      delete tp.second;
    }
  }
};

}  // namespace tensorflow
#endif  // ENABLE_MKLDNN_THREADPOOL
#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
