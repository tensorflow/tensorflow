// Defines the CUDAStream type - the CUDA-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_

#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {
namespace cuda {

class CUDAExecutor;

// Wraps a CUstream in order to satisfy the platform-independent
// StreamInterface.
//
// Thread-safe post-initialization.
class CUDAStream : public internal::StreamInterface {
 public:
  explicit CUDAStream(CUDAExecutor *parent)
      : parent_(parent), cuda_stream_(nullptr), completed_event_(nullptr) {}

  // Note: teardown is handled by a parent's call to DeallocateStream.
  ~CUDAStream() override {}

  void *CudaStreamHack() override { return cuda_stream_; }
  void **CudaStreamMemberHack() override {
    return reinterpret_cast<void **>(&cuda_stream_);
  }

  // Explicitly initialize the CUDA resources associated with this stream, used
  // by StreamExecutor::AllocateStream().
  bool Init();

  // Explicitly destroy the CUDA resources associated with this stream, used by
  // StreamExecutor::DeallocateStream().
  void Destroy();

  // Returns true if no work is pending or executing on the stream.
  bool IsIdle() const;

  // Retrieves an event which indicates that all work enqueued into the stream
  // has completed. Ownership of the event is not transferred to the caller, the
  // event is owned by this stream.
  bool GetOrCreateCompletedEvent(CUevent *completed_event);

  // Returns the CUstream value for passing to the CUDA API.
  //
  // Precond: this CUDAStream has been allocated (otherwise passing a nullptr
  // into the NVIDIA library causes difficult-to-understand faults).
  CUstream cuda_stream() const {
    DCHECK(cuda_stream_ != nullptr);
    return const_cast<CUstream>(cuda_stream_);
  }

  CUDAExecutor *parent() const { return parent_; }

 private:
  mutex mu_;              // mutex that guards the completion event.
  CUDAExecutor *parent_;  // Executor that spawned this stream.
  CUstream cuda_stream_;  // Wrapped CUDA stream handle.

  // Event that indicates this stream has completed.
  CUevent completed_event_ GUARDED_BY(mu_);
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_
