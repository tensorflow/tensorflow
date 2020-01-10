#include "tensorflow/stream_executor/cuda/cuda_event.h"

#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace perftools {
namespace gputools {
namespace cuda {

CUDAEvent::CUDAEvent(CUDAExecutor* parent)
    : parent_(parent), cuda_event_(nullptr) {}

CUDAEvent::~CUDAEvent() {}

port::Status CUDAEvent::Init() {
  return CUDADriver::CreateEvent(parent_->cuda_context(), &cuda_event_,
                                 CUDADriver::EventFlags::kDisableTiming);
}

port::Status CUDAEvent::Destroy() {
  return CUDADriver::DestroyEvent(parent_->cuda_context(), &cuda_event_);
}

port::Status CUDAEvent::Record(CUDAStream* stream) {
  return CUDADriver::RecordEvent(parent_->cuda_context(), cuda_event_,
                                 stream->cuda_stream());
}

Event::Status CUDAEvent::PollForStatus() {
  port::StatusOr<CUresult> status =
      CUDADriver::QueryEvent(parent_->cuda_context(), cuda_event_);
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().error_message();
    return Event::Status::kError;
  }

  switch (status.ValueOrDie()) {
    case CUDA_SUCCESS:
      return Event::Status::kComplete;
    case CUDA_ERROR_NOT_READY:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.ValueOrDie();
      return Event::Status::kError;
  }
}

const CUevent& CUDAEvent::cuda_event() {
  return cuda_event_;
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
