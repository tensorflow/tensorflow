
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

#ifndef TENSORFLOW_TSL_UTIL_ONEDNN_THREADPOOL_H_
#define TENSORFLOW_TSL_UTIL_ONEDNN_THREADPOOL_H_
#ifdef INTEL_MKL

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "dnnl_threadpool.hpp"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/threadpool.h"

namespace tsl {

#ifndef ENABLE_ONEDNN_OPENMP
using dnnl::threadpool_interop::threadpool_iface;

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

inline void run_jobs(bool balance, int i, int n, int njobs,
                     const std::function<void(int, int)>& fn) {
  if (balance) {
    int start, end;
    balance211(n, njobs, i, &start, &end);
    for (int j = start; j < end; j++) fn(j, n);
  } else {
    fn(i, n);
  }
}

class OneDnnThreadPool : public threadpool_iface {
 public:
  OneDnnThreadPool() = default;

  OneDnnThreadPool(Eigen::ThreadPoolInterface* eigen_interface,
                   int num_threads = -1)
      : eigen_interface_(eigen_interface) {
    set_num_and_max_threads(num_threads);
  }
  OneDnnThreadPool(Eigen::ThreadPoolInterface* eigen_interface,
                   bool can_use_caller_thread, int num_threads = -1)
      : eigen_interface_(eigen_interface),
        can_use_caller_thread_(can_use_caller_thread) {
    set_num_and_max_threads(num_threads);
  }
  virtual int get_num_threads() const override { return num_threads_; }
  virtual bool get_in_parallel() const override {
    return (eigen_interface_->CurrentThreadId() != -1) ? true : false;
  }
  virtual uint64_t get_flags() const override { return ASYNCHRONOUS; }
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
    bool balance = (nthr < n);

    // If use_caller_thread, schedule njobs-1 jobs to thread pool and run last
    // job directly.
    const bool use_caller_thread =
        can_use_caller_thread_ && nthr == port::NumSchedulableCPUs();
    const int njobs_to_schedule = use_caller_thread ? njobs - 1 : njobs;

    if (use_caller_thread) {
      for (int i = 0; i < njobs_to_schedule; i++) {
        eigen_interface_->ScheduleWithHint(
            [balance, i, n, njobs, fn]() {
              run_jobs(balance, i, n, njobs, fn);
            },
            i, i + 1);
      }
      run_jobs(balance, njobs_to_schedule, n, njobs, fn);
    } else {
      tsl::BlockingCounter counter(njobs);
      std::function<void(int, int)> handle_range = [=, &handle_range, &counter](
                                                       int first, int last) {
        while (last - first > 1) {
          const auto mid = first + (last - first) / 2;
          // Find something near the midpoint which is a multiple of block size.
          eigen_interface_->ScheduleWithHint([=]() { handle_range(mid, last); },
                                             mid, mid + 1);
          last = mid;
        }
        counter.DecrementCount();
        run_jobs(balance, first, n, njobs, fn);
      };

      // Eigen avoids a thread hop by running the root of the tree on the main
      // thread. We have disabled this because it actually slows things down
      // relative to base because base cheats and uses n threads while letting
      // main continue doing other work
      eigen_interface_->ScheduleWithHint([=]() { handle_range(0, njobs); }, 0,
                                         1);

      counter.Wait();
    }
  }

  ~OneDnnThreadPool() {}

  static void set_onednn_max_threads(int num_threads) {
#if DNNL_VERSION_MAJOR >= 3 || \
    (DNNL_VERSION_MAJOR == 2 && DNNL_VERSION_MINOR >= 7)
#ifndef DNNL_AARCH64_USE_ACL
    dnnl_threadpool_interop_set_max_concurrency(num_threads);
#endif  // DNNL_AARCH64_USE_ACL
#endif  // DNNL_VERSION_MAJOR >= 3 ||
        // (DNNL_VERSION_MAJOR == 2 && DNNL_VERSION_MINOR >= 7)
  }

 private:
  Eigen::ThreadPoolInterface* eigen_interface_ = nullptr;
  int num_threads_ = 1;                 // Execute in caller thread.
  bool can_use_caller_thread_ = false;  // true if the user set the env variable
                                        // to use caller thread also.
  inline void set_num_and_max_threads(int num_threads) {
    num_threads_ =
        num_threads == -1 ? eigen_interface_->NumThreads() : num_threads;
    set_onednn_max_threads(num_threads_);
  }
};

#else

// This class was just added to enable successful OMP-based build.
class OneDnnThreadPool {
 public:
  OneDnnThreadPool() = default;
  OneDnnThreadPool(Eigen::ThreadPoolInterface* eigen_interface) {}
  OneDnnThreadPool(Eigen::ThreadPoolInterface* eigen_interface,
                   bool can_use_caller_thread, int num_threads = -1) {}
  static void set_onednn_max_threads(int num_threads) {}
};

#endif  // !ENABLE_ONEDNN_OPENMP

}  // namespace tsl

#endif  // INTEL_MKL
#endif  // TENSORFLOW_TSL_UTIL_ONEDNN_THREADPOOL_H_
