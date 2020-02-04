/* Copyright 2020 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_INSTRUMENTATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_INSTRUMENTATION_H_

#ifdef RUY_PROFILER
#include <cstdio>
#include <mutex>
#include <vector>
#endif

namespace ruy {
namespace profiler {

#ifdef RUY_PROFILER

// A label is how a code scope is annotated to appear in profiles.
// The stacks that are sampled by the profiler are stacks of such labels.
// A label consists of a literal string, plus optional integer arguments.
class Label {
 public:
  Label() {}
  template <typename... Args>
  explicit Label(Args... args) {
    Set(args...);
  }
  void Set(const char* format) {
    format_ = format;
    args_count_ = 0;
  }
  template <typename... Args>
  void Set(const char* format, Args... args) {
    format_ = format;
    args_count_ = sizeof...(args);
    SetArgs(0, args...);
  }

  void operator=(const Label& other);

  bool operator==(const Label& other) const;

  std::string Formatted() const;
  const char* format() const { return format_; }

 private:
  void SetArgs(int position, int arg0) { args_[position] = arg0; }

  template <typename... Args>
  void SetArgs(int position, int arg0, Args... args) {
    SetArgs(position, arg0);
    SetArgs(position + 1, args...);
  }

  static constexpr int kMaxArgs = 4;
  const char* format_ = nullptr;
  int args_count_ = 0;
  int args_[kMaxArgs];
};

namespace detail {

// Forward-declaration, see class ThreadStack below.
class ThreadStack;

bool& GlobalIsProfilerRunning();

// Returns the global vector of pointers to all stacks, there being one stack
// per thread executing instrumented code.
std::vector<ThreadStack*>* GlobalAllThreadStacks();

// Returns the mutex to be locked around any access to GlobalAllThreadStacks().
std::mutex* GlobalsMutex();

// Returns the thread-local stack, specific to the current thread.
ThreadStack* ThreadLocalThreadStack();

// This 'stack' is what may be more appropriately called a 'pseudostack':
// It contains Label entries that are 'manually' entered by instrumentation
// code. It's unrelated to real call stacks.
struct Stack {
  std::uint32_t id = 0;
  static constexpr int kMaxSize = 64;
  int size = 0;
  Label labels[kMaxSize];
};

// Returns the buffer byte size required by CopyToSample.
int GetBufferSize(const Stack& stack);

// Copies this Stack into a byte buffer, called a 'sample'.
void CopyToBuffer(const Stack& stack, char* dst);

// Populates this Stack from an existing sample buffer, typically
// produced by CopyToSample.
void ReadFromBuffer(const char* src, Stack* stack);

// ThreadStack is meant to be used as a thread-local singleton, assigning to
// each thread a Stack object holding its pseudo-stack of profile labels,
// plus a mutex allowing to synchronize accesses to this pseudo-stack between
// this thread and a possible profiler thread sampling it.
class ThreadStack {
 public:
  ThreadStack();
  ~ThreadStack();

  const Stack& stack() const { return stack_; }

  // Returns the mutex to lock around any access to this stack. Each stack is
  // accessed by potentially two threads: the thread that it belongs to
  // (which calls Push and Pop) and the profiler thread during profiling
  // (which calls CopyToSample).
  std::mutex& Mutex() const { return mutex_; }

  // Pushes a new label on the top of this Stack.
  template <typename... Args>
  void Push(Args... args) {
    // This mutex locking is needed to guard against race conditions as both
    // the current thread and the profiler thread may be concurrently accessing
    // this stack. In addition to that, this mutex locking also serves the other
    // purpose of acting as a barrier (of compiler code reordering, of runtime
    // CPU instruction reordering, and of memory access reordering), which
    // gives a measure of correctness to this profiler. The downside is some
    // latency. As this lock will be uncontended most of the times, the cost
    // should be roughly that of an sequentially-consistent atomic access,
    // comparable to an access to the level of CPU data cache that is shared
    // among all cores, typically 60 cycles on current ARM CPUs, plus side
    // effects from barrier instructions.
    std::lock_guard<std::mutex> lock(mutex_);
    // Avoid overrunning the stack, even in 'release' builds. This profiling
    // instrumentation code should not ship in release builds anyway, the
    // overhead of this check is negligible, and overrunning a stack array would
    // be bad.
    if (stack_.size >= Stack::kMaxSize) {
      abort();
    }
    stack_.labels[stack_.size++].Set(args...);
  }

  // Pops the top-most label from this Stack.
  void Pop() {
    // See the comment in Push about this lock. While it would be tempting to
    // try to remove this lock and just atomically decrement size_ with a
    // store-release, that would not necessarily be a substitute for all of the
    // purposes that this lock serves, or if it was done carefully to serve all
    // of the same purposes, then that wouldn't be faster than this (mostly
    // uncontended) lock.
    std::lock_guard<std::mutex> lock(mutex_);
    stack_.size--;
  }

 private:
  mutable std::mutex mutex_;
  Stack stack_;
};

}  // namespace detail

// RAII user-facing way to construct Labels associated with their life scope
// and get them pushed to / popped from the current thread stack.
class ScopeLabel {
 public:
  template <typename... Args>
  ScopeLabel(Args... args) : thread_stack_(detail::ThreadLocalThreadStack()) {
    thread_stack_->Push(args...);
  }

  ~ScopeLabel() { thread_stack_->Pop(); }

 private:
  detail::ThreadStack* thread_stack_;
};

#else  // no RUY_PROFILER

class ScopeLabel {
 public:
  template <typename... Args>
  explicit ScopeLabel(Args...) {}

  // This destructor is needed to consistently silence clang's -Wunused-variable
  // which seems to trigger semi-randomly.
  ~ScopeLabel() {}
};

#endif

}  // namespace profiler
}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PROFILER_INSTRUMENTATION_H_
