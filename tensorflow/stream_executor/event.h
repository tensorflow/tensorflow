#ifndef TENSORFLOW_STREAM_EXECUTOR_EVENT_H_
#define TENSORFLOW_STREAM_EXECUTOR_EVENT_H_

#include <memory>

namespace perftools {
namespace gputools {

namespace internal {
class EventInterface;
}

class Stream;
class StreamExecutor;

// The Event class, when supported by a platform, enables low-overhead status
// reporting for a Stream. An Event is inserted at a location in a stream via
// the Stream::ThenRecordEvent() API. From then on, the Event's status can be
// monitored via the nonblocking Event::PollForStatus() call.
class Event {
 public:
  // Potential states for an Event. If PollForStatus() returns anything aside
  // from kPending or kComplete, an error has occurred; kUnknown is a bad state.
  // Not all implementations are able to return all enumeration values. Refer to
  // the platform-specific implementation for details.
  enum class Status {
    kUnknown,
    kError,
    kPending,
    kComplete,
  };

  explicit Event(StreamExecutor* stream_exec);  // NOLINT

  // Releases any resources held by the Event object.
  ~Event();

  // Performs any platform-specific or potentially error-generating
  // initialization.
  bool Init();

  // Returns the current Status for the event.
  Status PollForStatus();

  // Returns a pointer to the underlying platform-specific implementation.
  internal::EventInterface* implementation() { return implementation_.get(); }

 private:
  friend class Stream;

  // Pointer to the platform-specific EventInterface implementation underlying
  // the object. Owned.
  std::unique_ptr<internal::EventInterface> implementation_;

  // Pointer to the StreamExecutor interface used to create this object.
  // Not owned.
  StreamExecutor* stream_exec_;
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_EVENT_H_
