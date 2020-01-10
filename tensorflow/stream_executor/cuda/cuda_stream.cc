#include "tensorflow/stream_executor/cuda/cuda_stream.h"

#include "tensorflow/stream_executor/lib/status.h"

namespace perftools {
namespace gputools {
namespace cuda {

bool CUDAStream::Init() {
  return CUDADriver::CreateStream(parent_->cuda_context(), &cuda_stream_);
}

void CUDAStream::Destroy() {
  {
    mutex_lock lock{mu_};
    if (completed_event_ != nullptr) {
      port::Status status =
          CUDADriver::DestroyEvent(parent_->cuda_context(), &completed_event_);
      if (!status.ok()) {
        LOG(ERROR) << status.error_message();
      }
    }
  }

  CUDADriver::DestroyStream(parent_->cuda_context(), &cuda_stream_);
}

bool CUDAStream::IsIdle() const {
  return CUDADriver::IsStreamIdle(parent_->cuda_context(), cuda_stream_);
}

bool CUDAStream::GetOrCreateCompletedEvent(CUevent *completed_event) {
  mutex_lock lock{mu_};
  if (completed_event_ != nullptr) {
    *completed_event = completed_event_;
    return true;
  }

  if (!CUDADriver::CreateEvent(parent_->cuda_context(), &completed_event_,
                               CUDADriver::EventFlags::kDisableTiming)
           .ok()) {
    return false;
  }

  *completed_event = completed_event_;
  return true;
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
