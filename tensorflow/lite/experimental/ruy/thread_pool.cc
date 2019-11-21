/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/thread_pool.h"

#include <atomic>
#include <chrono>              // NOLINT(build/c++11)
#include <condition_variable>  // NOLINT(build/c++11)
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>               // NOLINT(build/c++11)
#include <thread>              // NOLINT(build/c++11)

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/wait.h"

namespace ruy {

// A worker thread.
class Thread {
 public:
  enum class State {
    Startup,  // The initial state before the thread main loop runs.
    Ready,    // Is not working, has not yet received new work to do.
    HasWork,  // Has work to do.
    ExitAsSoonAsPossible  // Should exit at earliest convenience.
  };

  explicit Thread(BlockingCounter* counter_to_decrement_when_ready)
      : task_(nullptr),
        state_(State::Startup),
        counter_to_decrement_when_ready_(counter_to_decrement_when_ready) {
    thread_.reset(new std::thread(ThreadFunc, this));
  }

  ~Thread() {
    ChangeState(State::ExitAsSoonAsPossible);
    thread_->join();
  }

  // Changes State; may be called from either the worker thread
  // or the master thread; however, not all state transitions are legal,
  // which is guarded by assertions.
  //
  // The Task argument is to be used only with new_state==HasWork.
  // It specifies the Task being handed to this Thread.
  void ChangeState(State new_state, Task* task = nullptr) {
    state_mutex_.lock();
    State old_state = state_.load(std::memory_order_relaxed);
    RUY_DCHECK_NE(old_state, new_state);
    switch (old_state) {
      case State::Startup:
        RUY_DCHECK_EQ(new_state, State::Ready);
        break;
      case State::Ready:
        RUY_DCHECK(new_state == State::HasWork ||
                   new_state == State::ExitAsSoonAsPossible);
        break;
      case State::HasWork:
        RUY_DCHECK(new_state == State::Ready ||
                   new_state == State::ExitAsSoonAsPossible);
        break;
      default:
        abort();
    }
    switch (new_state) {
      case State::Ready:
        if (task_) {
          // Doing work is part of reverting to 'ready' state.
          task_->Run();
          task_ = nullptr;
        }
        break;
      case State::HasWork:
        RUY_DCHECK(!task_);
        task_ = task;
        break;
      default:
        break;
    }
    state_.store(new_state, std::memory_order_relaxed);
    state_cond_.notify_all();
    state_mutex_.unlock();
    if (new_state == State::Ready) {
      counter_to_decrement_when_ready_->DecrementCount();
    }
  }

  static void ThreadFunc(Thread* arg) { arg->ThreadFuncImpl(); }

  // Called by the master thead to give this thread work to do.
  void StartWork(Task* task) { ChangeState(State::HasWork, task); }

 private:
  // Thread entry point.
  void ThreadFuncImpl() {
    ChangeState(State::Ready);

    // Thread main loop
    while (true) {
      // In the 'Ready' state, we have nothing to do but to wait until
      // we switch to another state.
      const auto& condition = [this]() {
        return state_.load(std::memory_order_acquire) != State::Ready;
      };
      WaitUntil(condition, &state_cond_, &state_mutex_);

      // Act on new state.
      switch (state_.load(std::memory_order_acquire)) {
        case State::HasWork:
          // Got work to do! So do it, and then revert to 'Ready' state.
          ChangeState(State::Ready);
          break;
        case State::ExitAsSoonAsPossible:
          return;
        default:
          abort();
      }
    }
  }

  // The underlying thread.
  std::unique_ptr<std::thread> thread_;

  // The task to be worked on.
  Task* task_;

  // The condition variable and mutex guarding state changes.
  std::condition_variable state_cond_;
  std::mutex state_mutex_;

  // The state enum tells if we're currently working, waiting for work, etc.
  // Its concurrent accesses by the thread and main threads are guarded by
  // state_mutex_, and can thus use memory_order_relaxed. This still needs
  // to be a std::atomic because we use WaitForVariableChange.
  std::atomic<State> state_;

  // pointer to the master's thread BlockingCounter object, to notify the
  // master thread of when this thread switches to the 'Ready' state.
  BlockingCounter* const counter_to_decrement_when_ready_;
};

void ThreadPool::ExecuteImpl(int task_count, int stride, Task* tasks) {
  RUY_DCHECK_GE(task_count, 1);

  // Case of 1 thread: just run the single task on the current thread.
  if (task_count == 1) {
    (tasks + 0)->Run();
    return;
  }

  // Task #0 will be run on the current thread.
  CreateThreads(task_count - 1);
  counter_to_decrement_when_ready_.Reset(task_count - 1);
  for (int i = 1; i < task_count; i++) {
    auto task_address = reinterpret_cast<std::uintptr_t>(tasks) + i * stride;
    threads_[i - 1]->StartWork(reinterpret_cast<Task*>(task_address));
  }

  // Execute task #0 immediately on the current thread.
  (tasks + 0)->Run();

  // Wait for the threads submitted above to finish.
  counter_to_decrement_when_ready_.Wait();
}

// Ensures that the pool has at least the given count of threads.
// If any new thread has to be created, this function waits for it to
// be ready.
void ThreadPool::CreateThreads(int threads_count) {
  if (threads_.size() >= threads_count) {
    return;
  }
  counter_to_decrement_when_ready_.Reset(threads_count - threads_.size());
  while (threads_.size() < threads_count) {
    threads_.push_back(new Thread(&counter_to_decrement_when_ready_));
  }
  counter_to_decrement_when_ready_.Wait();
}

ThreadPool::~ThreadPool() {
  for (auto w : threads_) {
    delete w;
  }
}

}  // end namespace ruy
