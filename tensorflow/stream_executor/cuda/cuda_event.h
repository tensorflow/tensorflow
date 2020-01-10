#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_

#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace perftools {
namespace gputools {
namespace cuda {

// CUDAEvent wraps a CUevent in the platform-independent EventInterface
// interface.
class CUDAEvent : public internal::EventInterface {
 public:
  explicit CUDAEvent(CUDAExecutor* parent);

  ~CUDAEvent() override;

  // Populates the CUDA-platform-specific elements of this object.
  port::Status Init();

  // Deallocates any platform-specific elements of this object. This is broken
  // out (not part of the destructor) to allow for error reporting.
  port::Status Destroy();

  // Inserts the event at the current position into the specified stream.
  port::Status Record(CUDAStream* stream);

  // Polls the CUDA platform for the event's current status.
  Event::Status PollForStatus();

  // The underyling CUDA event element.
  const CUevent& cuda_event();

 private:
  // The Executor used to which this object and CUevent are bound.
  CUDAExecutor* parent_;

  // The underlying CUDA event element.
  CUevent cuda_event_;
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_EVENT_H_
