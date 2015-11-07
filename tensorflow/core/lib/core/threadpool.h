#ifndef TENSORFLOW_LIB_CORE_THREADPOOL_H_
#define TENSORFLOW_LIB_CORE_THREADPOOL_H_

#include <deque>
#include <functional>
#include <thread>
#include <vector>
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {
namespace thread {

class ThreadPool {
 public:
  // Construct a pool that contains "num_threads" threads with specified "name".
  // env->StartThread() is used to create individual threads.
  //
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const string& name, int num_threads);

  // Construct a pool that contains "num_threads" threads with specified "name".
  // env->StartThread() is used to create individual threads.
  //
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options, const string& name,
             int num_threads);

  // Wait until all scheduled work has finished and then destroy the
  // set of threads.
  virtual ~ThreadPool();

  // Schedule fn() for execution in the pool of threads.
  virtual void Schedule(std::function<void()> fn);

  virtual bool HasPendingClosures() const;

 private:
  struct Waiter;
  struct Item {
    std::function<void()> fn;
    uint64 id;
  };

  void WorkerLoop();

  const string name_;
  mutable mutex mu_;
  std::vector<Thread*> threads_;  // All threads
  std::vector<Waiter*> waiters_;  // Stack of waiting threads.
  std::deque<Item> pending_;      // Queue of pending work

  TF_DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_THREADPOOL_H_
