#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_DEFAULT_MUTEX_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_DEFAULT_MUTEX_H_

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT

#include "tensorflow/stream_executor/platform/port.h"

// std::shared_timed_mutex is a C++14 feature.
#if (__cplusplus >= 201402L)
#define STREAM_EXECUTOR_USE_SHARED_MUTEX
#endif  // __cplusplus >= 201402L

#ifdef STREAM_EXECUTOR_USE_SHARED_MUTEX
#include <shared_mutex>  // NOLINT
#else
#include <mutex>  // NOLINT
#endif

namespace perftools {
namespace gputools {

enum ConditionResult { kCond_Timeout, kCond_MaybeNotified };

#ifdef STREAM_EXECUTOR_USE_SHARED_MUTEX
typedef std::shared_timed_mutex BaseMutex;
#else
typedef std::mutex BaseMutex;
#endif

// A class that wraps around the std::mutex implementation, only adding an
// additional LinkerInitialized constructor interface.
class mutex : public BaseMutex {
 public:
  mutex() {}
  // The default implementation of std::mutex is safe to use after the linker
  // initializations
  explicit mutex(LinkerInitialized x) {}
};

typedef std::unique_lock<BaseMutex> mutex_lock;

#ifdef STREAM_EXECUTOR_USE_SHARED_MUTEX
typedef std::shared_lock<BaseMutex> shared_lock;
#else
typedef std::unique_lock<BaseMutex> shared_lock;
#endif

using std::condition_variable;

inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                           condition_variable* cv, int64 ms) {
  std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
  return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
}

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLATFORM_DEFAULT_MUTEX_H_
