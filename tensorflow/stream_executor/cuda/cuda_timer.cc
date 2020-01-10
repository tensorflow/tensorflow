#include "tensorflow/stream_executor/cuda/cuda_timer.h"

#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace perftools {
namespace gputools {
namespace cuda {

bool CUDATimer::Init() {
  CHECK(start_event_ == nullptr && stop_event_ == nullptr);
  CUcontext context = parent_->cuda_context();
  if (!CUDADriver::CreateEvent(context, &start_event_,
                               CUDADriver::EventFlags::kDefault)
           .ok()) {
    return false;
  }

  if (!CUDADriver::CreateEvent(context, &stop_event_,
                               CUDADriver::EventFlags::kDefault)
           .ok()) {
    port::Status status = CUDADriver::DestroyEvent(context, &start_event_);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
    return false;
  }

  CHECK(start_event_ != nullptr && stop_event_ != nullptr);
  return true;
}

void CUDATimer::Destroy() {
  CUcontext context = parent_->cuda_context();
  port::Status status = CUDADriver::DestroyEvent(context, &start_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  status = CUDADriver::DestroyEvent(context, &stop_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

float CUDATimer::GetElapsedMilliseconds() const {
  CHECK(start_event_ != nullptr && stop_event_ != nullptr);
  // TODO(leary) provide a way to query timer resolution?
  // CUDA docs say a resolution of about 0.5us
  float elapsed_milliseconds = NAN;
  (void)CUDADriver::GetEventElapsedTime(parent_->cuda_context(),
                                        &elapsed_milliseconds, start_event_,
                                        stop_event_);
  return elapsed_milliseconds;
}

bool CUDATimer::Start(CUDAStream *stream) {
  return CUDADriver::RecordEvent(parent_->cuda_context(), start_event_,
                                 stream->cuda_stream())
      .ok();
}

bool CUDATimer::Stop(CUDAStream *stream) {
  return CUDADriver::RecordEvent(parent_->cuda_context(), stop_event_,
                                 stream->cuda_stream())
      .ok();
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
