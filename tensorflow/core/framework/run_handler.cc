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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/run_handler.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/run_handler_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

// Contains the concrete implementation of the RunHandler.
// Externally visible RunHandler class simply forwards the work to this one.
class RunHandler::Impl {
 public:
  explicit Impl(RunHandlerPool::Impl* pool_impl) : pool_impl_(pool_impl) {
    Reset();
  }

  ~Impl() {}

  void set_inter_op_scheduling_range(std::uint_fast32_t start,
                                     std::uint_fast32_t limit) {
    inter_op_scheduling_range_.store(EncodePartition(start, limit),
                                     std::memory_order_release);
  }

  std::uint_fast32_t inter_op_scheduling_range() const {
    return inter_op_scheduling_range_.load(std::memory_order_acquire);
  }

  // Stores now time (in microseconds) since unix epoch when the handler is
  // requested via RunHandlerPool::Get().
  uint64 start_time_us() const { return start_time_us_; }

  void ScheduleInterOpClosure(std::function<void()> fn);

  void Reset();

  RunHandlerPool::Impl* pool_impl() { return pool_impl_; }

 private:
  // Encoding/decoding logic for storing [start, limit) into a single
  // uint_fast32_t int. We assume that pool_num_threads < (1 << 16).
  const int kMaxPartitionBits = 16;
  const int kMaxThreads = 1 << kMaxPartitionBits;

  std::uint_fast32_t EncodePartition(std::uint_fast32_t start,
                                     std::uint_fast32_t limit) {
    return (start << kMaxPartitionBits) | limit;
  }

  void DecodePartition(std::uint_fast32_t val, std::uint_fast32_t* start,
                       std::uint_fast32_t* limit) {
    *limit = val & (kMaxThreads - 1);
    val >>= kMaxPartitionBits;
    *start = val;
  }

  std::atomic_uint_fast32_t inter_op_scheduling_range_;
  RunHandlerPool::Impl* pool_impl_;  // NOT OWNED.
  uint64 start_time_us_;
};

// Contains shared state across all run handlers present in the pool. Also
// responsible for pool management decisions.
// This class is thread safe.
class RunHandlerPool::Impl {
 public:
  explicit Impl(int num_inter_op_threads)
      : max_handlers_(128),
        inter_op_thread_pool_(new thread::ThreadPool(
            Env::Default(), ThreadOptions(), "inter_op", num_inter_op_threads)),
        iterations_(0) {
    VLOG(1) << "Creating a RunHandlerPool with max handlers: " << max_handlers_;
    for (int i = 0; i < max_handlers_; ++i) {
      handlers_.emplace_back(new RunHandler::Impl(this));
      free_handlers_.push_back(handlers_.back().get());
    }
    // Set steal partitions to a fixed size steal domain of size 6 = 2 *
    // kMinThreadsPerRequest.
    std::vector<std::pair<unsigned, unsigned>> steal_partitions(
        num_inter_op_threads);
    int kStealDomainSize = std::min(6, num_inter_op_threads);
    unsigned steal_start = 0, steal_end = kStealDomainSize;
    for (int i = 0; i < num_inter_op_threads; ++i) {
      if (i > steal_start) {
        if (steal_end + kStealDomainSize < num_inter_op_threads) {
          steal_start = steal_end;
          steal_end += kStealDomainSize;
        } else {
          steal_end = num_inter_op_threads;
          steal_start = steal_end - kStealDomainSize;
        }
      }
      steal_partitions[i] = std::make_pair(steal_start, steal_end);
      VLOG(1) << "Steal partition i: " << i << " steal_start: " << steal_start
              << " steal_end: " << steal_end;
    }
    inter_op_thread_pool_->SetStealPartitions(steal_partitions);
  }

  ~Impl() {
    // Sanity check that all handlers have been returned back to the pool before
    // destruction.
    DCHECK_EQ(handlers_.size(), max_handlers_);
    DCHECK_EQ(free_handlers_.size(), handlers_.size());
    DCHECK_EQ(sorted_active_handlers_.size(), 0);
  }

  thread::ThreadPool* inter_op_thread_pool() const {
    return inter_op_thread_pool_.get();
  }

  std::unique_ptr<RunHandler> Get() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    while (free_handlers_.empty()) {
      one_handler_free_.wait(l);
    }
    // Remove the last entry from free_handlers_ and add to the end of
    // sorted_active_handlers_.
    auto* handler_impl = free_handlers_.back();
    handler_impl->Reset();
    // Sortedness isn't violated if we simply add at the end of the list, since
    // handlers are expected to be obtained in increasing order of time.
    sorted_active_handlers_.push_back(handler_impl);
    DCHECK_LE(sorted_active_handlers_.size(), max_handlers_);
    free_handlers_.pop_back();

    RecomputePoolStatsLocked();
    return WrapUnique<RunHandler>(new RunHandler(handler_impl));
  }

  void ReleaseHandler(RunHandler::Impl* handler) LOCKS_EXCLUDED(mu_) {
    {
      mutex_lock l(mu_);
      DCHECK_GT(sorted_active_handlers_.size(), 0);

      uint64 now = tensorflow::Env::Default()->NowMicros();
      double elapsed = (now - handler->start_time_us()) / 1000.0;
      time_hist_.Add(elapsed);

      // Erase from and update sorted_active_handlers_. Add it to the end of
      // free_handlers_.
      auto iter = std::find(sorted_active_handlers_.begin(),
                            sorted_active_handlers_.end(), handler);
      DCHECK(iter != sorted_active_handlers_.end())
          << "Unexpected handler: " << handler
          << " is being requested for release";

      // Remove this handler from this list and add it to the list of free
      // handlers.
      sorted_active_handlers_.erase(iter);
      free_handlers_.push_back(handler);
      DCHECK_LE(free_handlers_.size(), max_handlers_);

      RecomputePoolStatsLocked();
    }
    one_handler_free_.notify_one();
  }

 private:
  void RecomputePoolStatsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Maximum number of handlers pre-created during pool construction time. The
  // number has been chosen expecting each handler might at least want 1
  // inter-op thread for execution (during compute intensive workloads like
  // inference).
  const int max_handlers_;

  // Thread safe part.
  const std::unique_ptr<thread::ThreadPool> inter_op_thread_pool_;

  // Thread compatible part used only by lock under RunHandlerPool.
  // Handlers are sorted by start time.
  std::vector<RunHandler::Impl*> sorted_active_handlers_ GUARDED_BY(mu_);
  std::vector<RunHandler::Impl*> free_handlers_ GUARDED_BY(mu_);
  std::vector<std::unique_ptr<RunHandler::Impl>> handlers_ GUARDED_BY(mu_);
  // Histogram of elapsed runtime of every handler (in ms).
  histogram::Histogram time_hist_ GUARDED_BY(mu_);
  std::vector<std::uint_fast32_t> inter_op_start_ GUARDED_BY(mu_);
  std::vector<std::uint_fast32_t> inter_op_limit_ GUARDED_BY(mu_);
  int64 iterations_ GUARDED_BY(mu_);
  condition_variable one_handler_free_;
  mutex mu_;
};

void RunHandlerPool::Impl::RecomputePoolStatsLocked() {
  int num_active_requests = sorted_active_handlers_.size();
  if (num_active_requests == 0) return;

  int num_threads = inter_op_thread_pool_->NumThreads();

  inter_op_start_.resize(num_active_requests);
  inter_op_limit_.resize(num_active_requests);

  const int kMinThreadsPerRequest = 3;
  ComputeInterOpSchedulingRanges(num_active_requests, num_threads,
                                 kMinThreadsPerRequest, &inter_op_start_,
                                 &inter_op_limit_);

  for (int i = 0; i < num_active_requests; ++i) {
    sorted_active_handlers_[i]->set_inter_op_scheduling_range(
        inter_op_start_[i], inter_op_limit_[i]);
  }

  if (iterations_++ % 5000 == 0 && VLOG_IS_ON(1)) {
    VLOG(1) << "Printing time histogram: " << time_hist_.ToString();
    VLOG(1) << "Active session runs: " << num_active_requests;
    uint64 now = tensorflow::Env::Default()->NowMicros();
    string ranges_str = "";
    string times_str = "";
    for (int i = 0; i < num_active_requests; ++i) {
      if (i > 0) {
        times_str += " ";
        ranges_str += " ";
      }

      times_str += strings::StrCat(
          (now - sorted_active_handlers_[i]->start_time_us()) / 1000.0, " ms.");
      ranges_str += strings::StrCat("[", inter_op_start_[i], ", ",
                                    inter_op_limit_[i], ")");
    }
    VLOG(1) << "Elapsed times are: " << times_str;
    VLOG(1) << "Ranges are: " << ranges_str;
  }
}

void RunHandler::Impl::ScheduleInterOpClosure(std::function<void()> fn) {
  std::uint_fast32_t start = 0, limit = 0;
  DecodePartition(inter_op_scheduling_range(), &start, &limit);
  DCHECK_LT(start, limit);
  pool_impl_->inter_op_thread_pool()->ScheduleWithHint(std::move(fn), start,
                                                       limit);
}

void RunHandler::Impl::Reset() {
  set_inter_op_scheduling_range(
      0, pool_impl_->inter_op_thread_pool()->NumThreads());
  start_time_us_ = tensorflow::Env::Default()->NowMicros();
}

RunHandlerPool::RunHandlerPool(int num_inter_op_threads)
    : impl_(new Impl(num_inter_op_threads)) {}

RunHandlerPool::~RunHandlerPool() {}

std::unique_ptr<RunHandler> RunHandlerPool::Get() { return impl_->Get(); }

RunHandler::RunHandler(Impl* impl) : impl_(impl) {}

void RunHandler::ScheduleInterOpClosure(std::function<void()> fn) {
  impl_->ScheduleInterOpClosure(std::move(fn));
}

RunHandler::~RunHandler() { impl_->pool_impl()->ReleaseHandler(impl_); }
}  // namespace tensorflow
