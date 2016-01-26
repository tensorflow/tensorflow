// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================

#ifndef TENSORFLOW_LIB_CORE_NONBLOCKING_THREADPOOL_H_
#define TENSORFLOW_LIB_CORE_NONBLOCKING_THREADPOOL_H_

#include <thread>
#include <vector>
#include <deque>
#include <unordered_set>
#include <atomic>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace thread {

//////////////////////////////////////////////////////////////////////////////
// NONBLOCKING THREAD POOL                                                   /
//////////////////////////////////////////////////////////////////////////////

// This is the internal implementation of the NonBlockingThreadPool class.
//
// It allows any thread to submit work across a set of pre-created worker
// threads.  The implementation is complicated by the fact that both thread
// pools and submitting threads may be created dynamically, in other words
// there is a many to many relationship between threads and thread pools.
//
// There are two kinds of threads: pool threads and external threads.
//
// Pool threads are those which are created as part of the thread pool, and
// whose sole function is to perform work.  These will continually scan for
// work, and once there is no work, go to sleep until there is more.  There
// is a fixed, known number of pool threads.  These will often steal work
// from other threads if they have none of their own.
//
// External threads may submit work, and may even contribute to getting work
// done while they are waiting for it to be finished.  Frequently we will
// end up with an external thread producing lots of work for the worker
// threads, but not being able to do much itself.  So the ability to
// handle lots of work being submitted by a given thread but not much being
// done by it is important.
//
// The implementation is based around work queues, described in this blog
// post:
// http://blog.molecular-matters.com/2015/08/24/job-system-2-0-lock-free-work-stealing-part-1-basics/
//
// Each thread has a work queue for each thread pool in which it is
// participating.  There is logic to allow a given thread to access its
// queue in the context of the current thread pool.
//
// WARNING: this thread pool does not guarantee any order of execution of
// submitted jobs, even with one single worker thread, so any ordering
// dependencies must be handled by submitting dependent jobs once the
// parent job is completed.  (In practice neither does any other thread
// pool implementation that has more than one thread, as either thread
// could pause for a long time before the job starts running).

class NonBlockingThreadPool {
 public:
  typedef std::function<void()> ThreadJob;

  NonBlockingThreadPool(Env* env, const string& name,
                        const ThreadOptions& thread_options, int num_threads)
      : jobs_stolen_(0),
        jobs_with_full_queue_(0),
        jobs_run_locally_(0),
        shutdown_(0),
        threads_sleeping_(0),
        thread_creation_epoch_(0),
        queues_(new Queues(thread_creation_epoch_)) {
    jobs_.submitted_ = 0;
    jobs_.finished_ = 0;

    CHECK_GE(num_threads, 1);
    string name_prefix = "tf_" + name;
    for (int i = 0; i < num_threads; i++) {
      workers_.push_back(env->StartThread(thread_options, name_prefix,
                                          [this, i]() { RunWorker(i); }));
    }
  }

  ~NonBlockingThreadPool() {
    // A requirement of TF thread pools is that all work be done
    // in the destructor if they are destroyed with work outstanding.
    WaitForAll();

    {
      mutex_lock guard(queues_mutex_);
      this->shutdown_ = 1;
    }

    wakeup_condition_variable_.notify_all();

    // Wait for threads to finish.
    for (auto t : workers_) {
      delete t;
    }
  }

  // Return the number of jobs running.  If there are more than
  // 2^31 jobs running, this may give the wrong answer.
  uint32_t jobs_running() const {
    uint64_t val = started_finished_;

    union {
      Jobs jobs;
      uint64_t started_finished;
    };

    started_finished = val;

    // Note: unsigned so we wrap around properly
    return uint32_t(jobs_.submitted_ - jobs_.finished_);
  }

  // Return the number of jobs submitted.  Wraps around at UINT_MAX.
  uint32_t jobs_submitted() const { return jobs_.submitted_; }

  // Return the number of jobs finished.  Wraps around at UINT_MAX.
  uint32_t jobs_finished() const { return jobs_.finished_; }

  // Schedule a new job to be run.
  void Schedule(ThreadJob job) {
    ++jobs_.submitted_;

    // Since we pass by value, we need to create a heap-allocated pointer
    // to be able to store a pointer
    std::unique_ptr<ThreadJob> overflow =
        GetEntry()->queue_->Push(new ThreadJob(std::move(job)));

    // There is a possible race condition here: if we add this
    // job just before the first thread goes to sleep,
    // then it will miss the wakeup.  We deal with this by having
    // the threads never sleep for too long, so that if we do
    // miss a wakeup it won't be the end of the world.
    if (threads_sleeping_) {
      wakeup_condition_variable_.notify_one();
    }

    if (overflow) {
      // The queue was full.  Do the work here, hopefully someone
      // will steal some work in the meantime.
      ++jobs_with_full_queue_;
      RunJob(*overflow);
    }
  }

 private:
  friend class Int64ThreadQueue;  // for unit testing

  // This is the basic deque used as the queue for each thread.  It's a
  // fixed size lock-free deck with the ability to push to one end and
  // pop from both ends.
  //
  // There is an implicit owning thread, which is allowed to call more
  // methods than other threads.  The thread ownership requirements
  // aren't checked but must be respected.

  struct ThreadQueue {
    static constexpr std::int_fast32_t QUEUE_LENGTH_LOG_2 = 10;
    static_assert(QUEUE_LENGTH_LOG_2 > 0, "Thread pool queue can't be empty");
    static_assert(QUEUE_LENGTH_LOG_2 < 31, "Thread pool queue is too long");

    // Length of the queue, 2^QUEUE_LENGTH_LOG_2
    static constexpr std::int_fast32_t QUEUE_LENGTH = 1 << QUEUE_LENGTH_LOG_2;

    // Mask to convert an index to a queue position.  It's due
    // to this mask that we require the length to be a power of two.
    static constexpr std::int_fast32_t MASK = QUEUE_LENGTH - 1u;

    // Element number for the bottom of the queue.  Note that a signed
    // type is required.
    std::atomic_int_fast32_t bottom_;

    // Element number for the top of the queue.  Note that a signed
    // type is required.
    std::atomic_int_fast32_t top_;

    // Total number of elements in the queue.  Used to ensure that
    // we don't wrap around.  It doesn't need to be synchronous with
    // the other elements, since only Push() reads it and at worst
    // it will think there are more queued than there really are,
    // leading to a harmless spurious overflow.
    std::atomic_int_fast32_t num_queued_;

    // Queue entries
    std::atomic<ThreadJob*> jobs_[QUEUE_LENGTH];

    ThreadQueue() : bottom_(0), top_(0), num_queued_(0) {
      std::fill(begin(jobs_), end(jobs_), nullptr);
    }

    // Push a new job onto the bottom of the queue.  This should only be
    // called by the owning thread.
    //
    // Will return true iff the push was successful.  A false return
    // indicates that the queue was full.
    std::unique_ptr<ThreadJob> Push(ThreadJob* job) {
      CHECK(job);

      if (num_queued_ == QUEUE_LENGTH) {
        return std::unique_ptr<ThreadJob>(job);
      }

      // Relaxed memory order, since we're the only thread that could
      // have written it.
      std::int_fast32_t b = bottom_.load(std::memory_order_relaxed);

      // Switch the new job in.
      jobs_[b & MASK] = job;

      // ensure the job is written before b+1 is published to other
      // threads.
      bottom_.store(b + 1, std::memory_order_release);

      // One more job is queued
      ++num_queued_;

      return nullptr;
    }

    // Steal an job from the top of the queue.  This can be called by any
    // thread; it exhibits FIFO functionality so older jobs will be
    // returned first.
    //
    // Will return the job if stolen, or nullptr if none was available.
    std::unique_ptr<ThreadJob> Steal() {
      std::int_fast32_t t = top_.load(std::memory_order_acquire);

      // ensure that top is always read before bottom.
      std::int_fast32_t b = bottom_.load(std::memory_order_acquire);

      if (t >= b) {
        return nullptr;  // no work to be taken
      }

      // non-empty queue
      ThreadJob* ptr = jobs_[t & MASK];

      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_acq_rel)) {
        // a concurrent steal or pop operation removed an element in
        // the meantime.
        return false;
      }

      CHECK(ptr);
      --num_queued_;
      return std::unique_ptr<ThreadJob>(ptr);
    }

    // Pop from the bottom_ of the queue, with LIFO semantics to improve
    // cache coherency.  This can only be called by the owning thread.
    //
    // Guaranteed that a race between pops and steals will result
    // in at least one entry being returned, which is handled by
    // retrying pops until we know that the queue is empty (which it
    // will remain, since the thread calling pop is the only thread
    // which can push elements).
    //
    // The path parameter, if non-null, will be used to return the
    // execution path of the operation as follows: each iteration
    // through the loop will add a single decimal digit, describing
    // which case was encountered, with earlier iterations in the most
    // significant digit.  See the code for the actual path codes.
    // This is used to debug atomicity failures with the queue, by
    // recording which path led to an issue.  For normal use, it will
    // be null and no path will be recorded.
    //
    // Path zero means that there was nothing queued (exits with
    // failure).
    // Path one is for when there was more than one element free,
    // and the bottom one was popped (exits with success).
    // Path two is for when we raced against a steal operation,
    // and lost (retries).
    // Path three is for when we raced against a steal operation, and
    // won (exits with success).
    // Path four is for when the queue was non-empty, but a steal
    // stole the entry before we could even race for it (retries).
    std::unique_ptr<ThreadJob> Pop(int* path = nullptr) {
      if (path) *path = 0;

      // Fast fail, since we're the same thread as the one which would
      // increment num_queued_, so if it's zero it really is zero
      // We need a while loop, as a race between pop and steal
      // can result in no element being returned from either
      while (num_queued_.load(std::memory_order_relaxed)) {
        if (path) *path *= 10;

        // Pre-emptively reserve the element at the bottom for us
        // We'll undo the decrement if it turns out we didn't
        // manage to reserve it.
        std::int_fast32_t b1 = bottom_;
        std::int_fast32_t b = b1 - 1;
        bottom_.exchange(b, std::memory_order_acq_rel);
        std::int_fast32_t t = top_.load(std::memory_order_acquire);

        if (t <= b) {
          // non-empty queue
          ThreadJob* ptr = jobs_[b & MASK];
          if (t != b) {
            if (path) *path += 1;
            // there's still more than one job left in the queue
            CHECK(ptr);
            --num_queued_;
            return std::unique_ptr<ThreadJob>(ptr);
          }

          // this is the last job in the queue
          if (!top_.compare_exchange_strong(t, t + 1,
                                            std::memory_order_acq_rel)) {
            if (path) *path += 2;
            // failed race against steal operation

            bottom_ = b1;
            continue;
          }

          bottom_.store(t + 1, std::memory_order_relaxed);

          CHECK(ptr);
          --num_queued_;
          if (path) *path += 3;
          return std::unique_ptr<ThreadJob>(ptr);
        } else {
          if (path) *path += 4;
          // already empty
          bottom_ = b1;
          continue;
        }
      }

      return nullptr;
    }
  };

  // A thread's local copy of the list of queues that may have work in
  // them, including an epoch number.
  struct Queues : public std::vector<std::shared_ptr<ThreadQueue> > {
    Queues(uint64_t epoch) : epoch_(epoch) {}

    // Epoch number for this set of queues.  By comparing with the
    // thread pool's epoch number, we can see if the list is out of
    // date or not.
    uint64_t epoch_;
  };

  // Data structure used to store information for each thread
  // associated with each thread pool.  Note that it may store
  // information about both the worker threads, as well as other
  // threads not managed by the thread pool that interract with
  // it by submitting jobs.
  class ThreadEntry {
   public:
    ThreadEntry(NonBlockingThreadPool* owner = nullptr, int worker_num = -1)
        : owner_(owner),
          worker_num_(worker_num),
          queue_(new ThreadQueue()),
          queues_(new Queues(0)) {}

    ~ThreadEntry() {
      // If we were never associated, we have nothing to do
      if (!owner_) {
        return;
      }

      // Thread is being shutdown.  But what to do with
      // its work?  If it's a shutdown due to an exception,
      // OK.  But otherwise it should have waited for it
      // to be done.
      owner_->UnpublishThread(this);
    }

    // The NonBlockingThreadPool we're owned by
    NonBlockingThreadPool* owner_;

    // Our thread number; used to choose a starting point in the
    // list of available queues and avoid races.  This is only
    // set for worker threads; for others it will be -1 as they
    // don't normally scavenge for work to do.
    int worker_num_;

    // Our reference to our work queue.  It's a shared pointer
    // because others may continue to reference it even after
    // our thread has been destroyed, and allowing this avoids
    // a lot of synchronization and locking.
    std::shared_ptr<ThreadQueue> queue_;

    // The list of queues that we know about over all threads.
    // This is a cached copy that we occasionally check to see
    // if it needs to be updated.
    std::shared_ptr<const Queues> queues_;
  };

  // A standard thread_local variable allows us to store one item per
  // thread.  However, we need one queue per thread *per thread pool*,
  // and thread pools can come and go.  This class allows us to have
  // an instance per thread per thread pool, ie it's like a non-static
  // thread_local variable.
  //
  // Note that while this class has several locks, they're only grabbed
  // when an instance is created, destroyed or first accessed. Past
  // the first access, reads equate to a deque probe.
  class ThreadSpecificInstanceInfo {
   public:
    // If there are lots of threads being created (1000s per second),
    // then we may consider changing this to a spinlock.
    typedef mutex Lock;

    struct Value {
      Value() : object_(nullptr) {}
      ~Value() {
        ThreadSpecificInstanceInfo* oldObject = Destruct();
        if (!oldObject) {
          return;
        }

        mutex_lock guard(oldObject->free_set_lock_);
        oldObject->free_set_.erase(this);
      }

      ThreadSpecificInstanceInfo* Destruct() {
        mutex_lock guard(destruct_lock_);
        if (!object_) {
          return nullptr;
        }

        storage_.value_.~ThreadEntry();
        auto oldObject = object_;
        object_ = nullptr;

        return oldObject;
      }

      // This can't raise with either object destruction or thread
      // destruction so no locks are needed.
      void Construct(ThreadSpecificInstanceInfo* newObject) {
        new (&storage_.value_) ThreadEntry();
        object_ = newObject;
      }

      // The odd setup is to prevent spurious calls to the ThreadEntry
      // constructor and destructor when we construct our parent class
      // Value.
      //
      // Note that using a union is a well defined type-puning construct
      // in gcc while reinterpret_cast<> could cause problems when used
      // with strict-aliasing (I think). Feel free to simplify it if
      // I'm wrong.
      union Storage {
        Storage() {}
        ~Storage() {}

        ThreadEntry value_;
        uint8_t unused_[sizeof(ThreadEntry)];
      } storage_;

      Lock destruct_lock_;
      ThreadSpecificInstanceInfo* object_;
    };

    typedef std::deque<Value> PerThreadInfo;

    ThreadSpecificInstanceInfo() {
      mutex_lock guard(free_index_lock);

      if (!free_indexes.empty()) {
        index_ = free_indexes.front();
        free_indexes.pop_front();
      } else {
        index_ = ++next_index;
      }
    }

    ~ThreadSpecificInstanceInfo() {
      // We don't want to be holding the free_set_ lock when calling
      // destruct because thread destruction will also attempt to
      // lock our free_set_ lock which is a recipe for deadlocks.
      std::unordered_set<Value*> free_set_copy;
      {
        mutex_lock guard(free_set_lock_);
        free_set_copy = std::move(free_set_);
      }

      for (Value* toFree : free_set_copy) {
        toFree->Destruct();
      }

      mutex_lock guard(free_index_lock);
      free_indexes.push_back(index_);
    }

    // Return the data for this thread for this instance of the class.
    ThreadEntry* get() const {
      PerThreadInfo* info = static_info.get();
      return load(info);
    }

   private:
    ThreadEntry* load(PerThreadInfo* info) const {
      while ((int64_t)info->size() <= index_) {
        info->emplace_back();
      }

      Value& val = (*info)[index_];

      if (!val.object_) {
        val.Construct(const_cast<ThreadSpecificInstanceInfo*>(this));
        mutex_lock guard(free_set_lock_);
        free_set_.insert(&val);
      }

      return &val.storage_.value_;
    }

    // Per-thread information (shared across all thread pools) for
    // each thread
    static thread_local std::unique_ptr<PerThreadInfo> static_info;

    // Mutex that protects the list of free indexes.  We use this to
    // maintain thread indexes as small integers so that lookups are
    // a simple vector probe.
    static mutex free_index_lock;

    // List of free indexes; protected by free_index_lock.
    static std::deque<size_t> free_indexes;

    // Next index, for when there are none free
    static uint32_t next_index;

    // The index of this particular thread specific information block.
    // For any thread, this is the index in that thread's information
    // of our instance's information.
    int32_t index_;

    // Mutex protecting free_set_
    mutable mutex free_set_lock_;

    // Set of objects that need to be freed once this object is
    // destroyed.  Protected with a mutex to reduce complexity;
    // otherwise we would require a concurrent garbage collection
    // mechanism.
    mutable std::unordered_set<Value*> free_set_;
  };

  // This allows us to have one threadEntry per thread.  It maintains
  // access to one ThreadEntry per thread per thread pool.
  ThreadSpecificInstanceInfo thread_entries_;

  // Our internal worker threads
  std::vector<Thread*> workers_;

  // Job statistics.  This is designed to allow for a single atomic
  // access to the full 64 bits to allow determination if all
  // jobs have been terminated at a given point in time.
  struct Jobs {
    std::atomic<int32_t> submitted_;
    std::atomic<int32_t> finished_;
  };

  union {
    Jobs jobs_;
    std::atomic<uint64_t> started_finished_;
  };

  // Statistics counters for debugging and information
  std::atomic<uint64_t> jobs_stolen_;
  std::atomic<uint64_t> jobs_with_full_queue_;
  std::atomic<uint64_t> jobs_run_locally_;

  // Non-zero when we're shutting down.
  std::atomic_bool shutdown_;

  // Number of sleeping threads.  This is used to
  // help triggering wakeups if there are no sleeping threads
  // to be woken.
  std::atomic_int_fast32_t threads_sleeping_;

  // Mutex for the wakeup condition variable
  mutex wakeup_mutex_;

  // Wakeup condition variable
  condition_variable wakeup_condition_variable_;

  // Epoch number for thread creation.  We can use this to
  // tell if a set of queues is current, by checking for
  // matching epoch numbers.
  std::atomic<uint64_t> thread_creation_epoch_;

  // List of all the queues that could contain work to steal.
  // Protected by queues_mutex_, since in C++11 shared_ptrs
  // can't be atomically modified.
  std::shared_ptr<const Queues> queues_;

  // Mutex to protect modification to the list of queues.  This is
  // needed since std::shared_ptr doesn't have an atomic compare
  // and exchange operation. The lock will only need to be taken
  // when new threads are being created, so it doesn't introduce a
  // mutex into the job queueing or execution paths.
  mutex queues_mutex_;

  // Runs as much work as possible in this thread's queue.  Returns
  // true if some work was obtained.
  bool RunMyJobs(ThreadEntry* entry) {
    CHECK(entry);
    bool result = false;

    // First, do all of our work
    std::unique_ptr<ThreadJob> job;
    while ((job = entry->queue_->Pop())) {
      result = true;
      ++jobs_run_locally_;
      RunJob(*job);
    }

    return result;
  }

  // Steal one bit of work from another thread and run it.  Returns
  // true if some work was obtained.
  bool StealWork(ThreadEntry* entry) {
    CHECK(entry);
    bool foundWork = false;

    // Check if we have the latest list of queues, by looking at
    // the epoch number.
    if (thread_creation_epoch_.load() != entry->queues_->epoch_) {
      // A thread has been created or destroyed, and so our list
      // of queues is out of date.  Refresh them if we can obtain
      // the mutex.  If we can't refresh it's not a big deal, since
      // at worst we have references to queues that are no longer
      // replenished or we're missing some work.  When we run out of
      // work to do, we'll try again.
      mutex_lock guard(queues_mutex_, std::try_to_lock);

      if (guard) {
        // We successfully locked the mutex.  Now we can read queues
        // and take a reference to it.
        entry->queues_ = queues_;
        CHECK_EQ(entry->queues_->epoch_, thread_creation_epoch_);
      }
    }

    for (unsigned i = 0; i < entry->queues_->size() && !shutdown_; ++i) {
      // Try to avoid all threads starting looking for work at the
      // same place.
      int n = (entry->worker_num_ + i) % entry->queues_->size();

      const std::shared_ptr<ThreadQueue>& q = entry->queues_->at(n);

      if (q == entry->queue_) {
        continue;  // our own thread
      }

      std::unique_ptr<ThreadJob> job;
      while ((job = q->Steal())) {
        ++jobs_stolen_;
        RunJob(*job);

        // If that job submitted anything, we run it now
        RunMyJobs(entry);
      }
    }

    return foundWork;
  }

  // Run a job we successfully dequeued from a queue somewhere.
  void RunJob(const ThreadJob& job) {
    job();
    ++jobs_.finished_;
  }

  // Wait for all work in all threads to be done, and return when it
  // is.
  void WaitForAll() {
    ThreadEntry* entry = GetEntry();

    while (!shutdown_ && jobs_running() > 0) {
      if (!RunMyJobs(entry)) {
        StealWork(entry);
      }
    }
  }

  // Return the entry for this thread for this thread pool
  // The worker_num is used when initializing to set the worker
  // number.
  ThreadEntry* GetEntry(int worker_num = -1) {
    ThreadEntry* threadEntry = thread_entries_.get();
    CHECK(threadEntry);

    // If it's not initialized yet, this is the first time we've
    // seen this thread.  So we initialize the thread's entry and
    // publish its queue to the list of queues.
    if (!threadEntry->owner_) {
      threadEntry->owner_ = this;
      threadEntry->worker_num_ = worker_num;
      PublishThread(threadEntry);
    }

    return threadEntry;
  }

  // Run a worker thread.  This does any work in its own queue, and
  // then looks to steal work from another queue.  If there is nothing
  // to do for a while, it will sleep for a millisecond and start
  // again.
  void RunWorker(int worker_num) {
    ThreadEntry* entry = GetEntry(worker_num);

    int iterations_with_no_work = 0;

    while (!shutdown_.load()) {
      if (!RunMyJobs(entry)) {
        if (!StealWork(entry)) {
          // Nothing to do, for now.  Wait for something to
          // wake us up.  We try 10 times, and if there is
          // nothing to do then we go to sleep and wait for
          // some more work to come.
          ++iterations_with_no_work;
          if (iterations_with_no_work == 10) {
            ++threads_sleeping_;
            mutex_lock guard(wakeup_mutex_);

            // We can't sleep forever, since we allow for
            // wakeups to be missed for efficiency reasons,
            // and so we need to poll every now and again.
            wakeup_condition_variable_.wait_for(guard,
                                                std::chrono::milliseconds(1));

            --threads_sleeping_;
            iterations_with_no_work = 0;
          } else {
            // We didn't find any work, but it's not yet time
            // to give up on it.  We wait a small amount of
            // time and try again.
            // std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        }
      }
    }
  }

  // A new thread has made itself known to this thread pool.  Publish
  // its queue in the list of known queues so that other threads may
  // steal work from it.
  void PublishThread(ThreadEntry* thread) {
    if (shutdown_) {
      return;
    }
    CHECK(thread);
    mutex_lock guard(queues_mutex_);
    if (shutdown_) {
      return;
    }

    std::shared_ptr<Queues> new_queues(new Queues(*queues_));

    // Don't allow epoch zero to be used on a wraparound, as its
    // reserved for the empty set in the constructor.
    do {
      new_queues->epoch_ = thread_creation_epoch_.fetch_add(1) + 1;
    } while (new_queues->epoch_ == 0);

    new_queues->emplace_back(thread->queue_);
    queues_ = new_queues;
  }

  // A thread has exited and so its queue is no longer available for
  // work stealing.  Publish the reduced list of queues.
  void UnpublishThread(ThreadEntry* thread) {
    if (shutdown_) {
      return;
    }
    CHECK(thread);
    mutex_lock guard(queues_mutex_);
    if (shutdown_) {
      return;
    }

    std::shared_ptr<Queues> new_queues(new Queues(*queues_));

    // Don't allow epoch zero to be used on a wraparound, as its
    // reserved for the empty set in the constructor.
    do {
      new_queues->epoch_ = thread_creation_epoch_.fetch_add(1) + 1;
    } while (new_queues->epoch_ == 0);

    // Note: std::find triggers a compiler bug in GCC 4.8, so
    // we unroll it explicitly here.
    bool found_thread_to_unpublish = false;
    for (auto it = new_queues->begin(), end = new_queues->end();
         !found_thread_to_unpublish && it != end; ++it) {
      if (*it == thread->queue_) {
        new_queues->erase(it);
        found_thread_to_unpublish = true;
      }
    }
    CHECK(found_thread_to_unpublish);

    queues_ = new_queues;
  }
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_NONBLOCKING_THREADPOOL_H_
