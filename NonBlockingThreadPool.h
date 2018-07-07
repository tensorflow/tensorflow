// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
#define EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H

#include <iostream>

namespace Eigen {

template <typename Environment>
class NonBlockingThreadPoolTempl : public Eigen::ThreadPoolInterface {
 public:
  typedef typename Environment::Task Task;
  typedef RunQueue<Task, 1024> Queue;

  NonBlockingThreadPoolTempl(int num_threads, std::vector<int> &proc_set, Environment env = Environment())
      : NonBlockingThreadPoolTempl(num_threads, proc_set, true, env) {}

  NonBlockingThreadPoolTempl(int num_threads, std::vector<int> &proc_set, bool allow_spinning,
                             Environment env = Environment())
      : env_(env),
        num_threads_(num_threads),
        allow_spinning_(allow_spinning),
        threads_(num_threads),
        queues_(num_threads),
        coprimes_(num_threads),
        waiters_(num_threads),
        blocked_(0),
        spinning_(0),
        done_(false),
        cancelled_(false),
        ec_(waiters_) {
    if(proc_set.size() != 0) {
        proc_set_.insert(proc_set_.begin(), proc_set.begin(), proc_set.end());
        num_threads_ = proc_set_.size();
    }
 
    thread_bound_.resize(num_threads_);
    waiters_.resize(num_threads_);

    // Calculate coprimes of num_threads_.
    // Coprimes are used for a random walk over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a walk starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    for (int i = 1; i <= num_threads_; i++) {
      unsigned a = i;
      unsigned b = num_threads_;
      // If GCD(a, b) == 1, then a and b are coprimes.
      while (b != 0) {
        unsigned tmp = a;
        a = b;
        b = tmp % b;
      }
      if (a == 1) {
        coprimes_.push_back(i);
      }
    }
    for (int i = 0; i < num_threads_; i++) {
      queues_.push_back(new Queue());
    }
    for (int i = 0; i < num_threads_; i++) {
      threads_.push_back(env_.CreateThread([this, i]() { WorkerLoop(i); }));
    }
  }

  ~NonBlockingThreadPoolTempl() {
    if(proc_set_.size() != 0){
      std::cout << "NonBlockingThreadPool: " << (long) this << " bound to: ";
      for(int i=0; i<thread_bound_.size(); i++) 
        std::cout << thread_bound_[i] << " ";
      std::cout << "\n";
    }

    done_ = true;

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    if (!cancelled_) {
      ec_.Notify(true);
    } else {
      // Since we were cancelled, there might be entries in the queues.
      // Empty them to prevent their destructor from asserting.
      for (size_t i = 0; i < queues_.size(); i++) {
        queues_[i]->Flush();
      }
    }

    // Join threads explicitly to avoid destruction order issues.
    for (size_t i = 0; i < num_threads_; i++) delete threads_[i];
    for (size_t i = 0; i < num_threads_; i++) delete queues_[i];
  }

  void Schedule(std::function<void()> fn) {
    Task t = env_.CreateTask(std::move(fn));
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue* q = queues_[pt->thread_id];
      t = q->PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      Queue* q = queues_[Rand(&pt->rand) % queues_.size()];
      t = q->PushBack(std::move(t));
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t.f) {
      ec_.Notify(false);
    }
    else {
      env_.ExecuteTask(t);  // Push failed, execute directly.
    }
  }

  void Cancel() {
    cancelled_ = true;
    done_ = true;

    // Let each thread know it's been cancelled.
#ifdef EIGEN_THREAD_ENV_SUPPORTS_CANCELLATION
    for (size_t i = 0; i < threads_.size(); i++) {
      threads_[i]->OnCancel();
    }
#endif

    // Wake up the threads without work to let them exit on their own.
    ec_.Notify(true);
  }

  int NumThreads() const final {
    return num_threads_;
  }

  int CurrentThreadId() const final {
    const PerThread* pt =
        const_cast<NonBlockingThreadPoolTempl*>(this)->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

 private:
  typedef typename Environment::EnvThread Thread;

  struct PerThread {
    constexpr PerThread() : pool(NULL), rand(0), thread_id(-1) { }
    NonBlockingThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    uint64_t rand;  // Random generator state.
    int thread_id;  // Worker thread index in pool.
  };

  Environment env_;
  int num_threads_;
  const bool allow_spinning_;
  MaxSizeVector<Thread*> threads_;
  MaxSizeVector<Queue*> queues_;
  MaxSizeVector<unsigned> coprimes_;
  MaxSizeVector<EventCount::Waiter> waiters_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> spinning_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;
  EventCount ec_;
  std::vector<int> proc_set_;
  std::vector<int> thread_bound_;

  // Main worker thread loop.
  void WorkerLoop(int thread_id) {
    if(proc_set_.size() != 0) {
      //set the thread affinity
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(proc_set_[thread_id], &cpuset);
      pthread_t thread = pthread_self();

      int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
      if (s != 0) {
        std::cout << "NonBlockingThreadPool: Failed to set thread affinity\n";
        return;
      }

      /* Check the actual affinity mask assigned to the thread */
      s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
      if (s != 0)
        std::cout << "NonBlockingThreadPool: Failed to call pthread_getaffinity_np\n";
      else {
        for (int j = 0; j < CPU_SETSIZE; j++)
          if (CPU_ISSET(j, &cpuset))
            thread_bound_[thread_id] = j;
      }

      char buf[1024];
      sprintf(buf, "eigen worker %d", thread_id);
      pthread_setname_np(thread, buf);
    }

    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->rand = std::hash<std::thread::id>()(std::this_thread::get_id());
    pt->thread_id = thread_id;
    Queue* q = queues_[thread_id];
    EventCount::Waiter* waiter = &waiters_[thread_id];
    // TODO(dvyukov,rmlarsen): The time spent in Steal() is proportional
    // to num_threads_ and we assume that new work is scheduled at a
    // constant rate, so we set spin_count to 5000 / num_threads_. The
    // constant was picked based on a fair dice roll, tune it.
    const int spin_count =
        allow_spinning_ && num_threads_ > 0 ? 5000 / num_threads_ : 0;
    if (num_threads_ == 1) {
      // For num_threads_ == 1 there is no point in going through the expensive
      // steal loop. Moreover, since Steal() calls PopBack() on the victim
      // queues it might reverse the order in which ops are executed compared to
      // the order in which they are scheduled, which tends to be
      // counter-productive for the types of I/O workloads the single thread
      // pools tend to be used for.
      while (!cancelled_) {
        Task t = q->PopFront();
        for (int i = 0; i < spin_count && !t.f; i++) {
          if (!cancelled_.load(std::memory_order_relaxed)) {
            t = q->PopFront();
          }
        }
        if (!t.f) {
          if (!WaitForWork(waiter, &t)) {
            return;
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    } else {
      while (!cancelled_) {
        Task t = q->PopFront();
        if (!t.f) {
          t = Steal();
          if (!t.f) {
            // Leave one thread spinning. This reduces latency.
            if (allow_spinning_ && !spinning_ && !spinning_.exchange(true)) {
              for (int i = 0; i < spin_count && !t.f; i++) {
                if (!cancelled_.load(std::memory_order_relaxed)) {
                  t = Steal();
                } else {
                  return;
                }
              }
              spinning_ = false;
            }
            if (!t.f) {
              if (!WaitForWork(waiter, &t)) {
                return;
              }
            }
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    }
  }

  // Steal tries to steal work from other worker threads in best-effort manner.
  Task Steal() {
    PerThread* pt = GetPerThread();
    const size_t size = queues_.size();
    unsigned r = Rand(&pt->rand);
    unsigned inc = coprimes_[r % coprimes_.size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      Task t = queues_[victim]->PopBack();
      if (t.f) {
        return t;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return Task();
  }

  // WaitForWork blocks until new work is available (returns true), or if it is
  // time to exit (returns false). Can optionally return a task to execute in t
  // (in such case t.f != nullptr on return).
  bool WaitForWork(EventCount::Waiter* waiter, Task* t) {
    eigen_assert(!t->f);
    // We already did best-effort emptiness check in Steal, so prepare for
    // blocking.
    ec_.Prewait(waiter);
    // Now do a reliable emptiness check.
    int victim = NonEmptyQueueIndex();
    if (victim != -1) {
      ec_.CancelWait(waiter);
      if (cancelled_) {
        return false;
      } else {
        *t = queues_[victim]->PopBack();
        return true;
      }
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    blocked_++;
    if (done_ && blocked_ == num_threads_) {
      ec_.CancelWait(waiter);
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (NonEmptyQueueIndex() != -1) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        blocked_--;
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(true);
      return false;
    }
    ec_.CommitWait(waiter);
    blocked_--;
    return true;
  }

  int NonEmptyQueueIndex() {
    PerThread* pt = GetPerThread();
    const size_t size = queues_.size();
    unsigned r = Rand(&pt->rand);
    unsigned inc = coprimes_[r % coprimes_.size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      if (!queues_[victim]->Empty()) {
        return victim;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return -1;
  }

  static EIGEN_STRONG_INLINE PerThread* GetPerThread() {
    EIGEN_THREAD_LOCAL PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  static EIGEN_STRONG_INLINE unsigned Rand(uint64_t* state) {
    uint64_t current = *state;
    // Update the internal state
    *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
  }
};

typedef NonBlockingThreadPoolTempl<StlThreadEnvironment> NonBlockingThreadPool;

}  // namespace Eigen

#endif  // EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
