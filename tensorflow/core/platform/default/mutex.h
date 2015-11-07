#ifndef TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_
#define TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace tensorflow {

enum LinkerInitialized { LINKER_INITIALIZED };

// A class that wraps around the std::mutex implementation, only adding an
// additional LinkerInitialized constructor interface.
class mutex : public std::mutex {
 public:
  mutex() {}
  // The default implementation of std::mutex is safe to use after the linker
  // initializations
  explicit mutex(LinkerInitialized x) {}
};

using std::condition_variable;
typedef std::unique_lock<std::mutex> mutex_lock;

inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                           condition_variable* cv, int64 ms) {
  std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
  return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_
