#include "tensorflow/core/lib/core/threadpool.h"

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace thread {

struct ThreadPool::Waiter {
  condition_variable cv;
  bool ready;
};

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : name_(name) {
  CHECK_GE(num_threads, 1);
  string name_prefix = "tf_" + name_;
  for (int i = 0; i < num_threads; i++) {
    threads_.push_back(env->StartThread(thread_options, name_prefix,
                                        [this]() { WorkerLoop(); }));
  }
}

ThreadPool::~ThreadPool() {
  {
    // Wait for all work to get done.
    mutex_lock l(mu_);

    // Inform every thread to exit.
    for (size_t i = 0; i < threads_.size(); ++i) {
      pending_.push_back({nullptr, 0});
    }

    // Wakeup all waiters.
    for (auto w : waiters_) {
      w->ready = true;
      w->cv.notify_one();
    }
  }

  // Wait for threads to finish.
  for (auto t : threads_) {
    delete t;
  }
}

bool ThreadPool::HasPendingClosures() const {
  mutex_lock l(mu_);
  return pending_.size() != 0;
}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  uint64 id = 0;
  if (port::Tracing::IsActive()) {
    id = port::Tracing::UniqueId();
    port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
                               id);
  }

  mutex_lock l(mu_);
  pending_.push_back({fn, id});
  if (!waiters_.empty()) {
    Waiter* w = waiters_.back();
    waiters_.pop_back();
    w->ready = true;
    w->cv.notify_one();
  }
}

void ThreadPool::WorkerLoop() {
  port::Tracing::RegisterCurrentThread(name_.c_str());
  mutex_lock l(mu_);
  Waiter w;
  while (true) {
    while (pending_.empty()) {
      // Wait for work to be assigned to me
      w.ready = false;
      waiters_.push_back(&w);
      while (!w.ready) {
        w.cv.wait(l);
      }
    }
    // Pick up pending work
    Item item = pending_.front();
    pending_.pop_front();
    if (item.fn == nullptr) {
      break;
    }
    mu_.unlock();
    if (item.id != 0) {
      port::Tracing::ScopedActivity region(
          port::Tracing::EventCategory::kRunClosure, item.id);
      item.fn();
    } else {
      item.fn();
    }
    mu_.lock();
  }
}

}  // namespace thread
}  // namespace tensorflow
