#ifndef TENSORFLOW_STREAM_EXECUTOR_TIMER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TIMER_H_

#include <memory>

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

namespace internal {
class TimerInterface;
}  // namespace internal

class StreamExecutor;

// An interval timer, suitable for use in timing the operations which occur in
// streams.
//
// Thread-hostile: CUDA associates a CUDA-context with a particular thread in
// the system. Any operation that a user attempts to perform by using a Timer
// on a thread not-associated with the CUDA-context has unknown behavior at the
// current time; see b/13176597
class Timer {
 public:
  // Instantiate a timer tied to parent as a platform executor.
  explicit Timer(StreamExecutor *parent);

  // Deallocates any timer resources that the parent StreamExecutor has bestowed
  // upon this object.
  ~Timer();

  // Returns the elapsed number of microseconds for a completed timer.
  // Completed means has been through a start/stop lifecycle.
  uint64 Microseconds() const;

  // Returns the elapsed number of nanoseconds for a completed timer.
  // Completed means has been through a start/stop lifecycle.
  uint64 Nanoseconds() const;

  // Returns the (opaque) backing platform ITimer instance. Ownership is
  // not transferred to the caller.
  internal::TimerInterface *implementation() { return implementation_.get(); }

 private:
  // Platform-dependent implementation of the timer internals for the underlying
  // platform. This class just delegates to this opaque instance.
  std::unique_ptr<internal::TimerInterface> implementation_;

  // The StreamExecutor that manages the platform-specific internals for this
  // timer.
  StreamExecutor *parent_;

  SE_DISALLOW_COPY_AND_ASSIGN(Timer);
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_TIMER_H_
