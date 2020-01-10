#include "tensorflow/stream_executor/timer.h"

#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {

static internal::TimerInterface *CreateTimerImplementation(
    StreamExecutor *parent) {
  PlatformKind platform_kind = parent->platform_kind();
  if (platform_kind == PlatformKind::kCuda) {
    return (*internal::MakeCUDATimerImplementation())(parent);
  } else if (platform_kind == PlatformKind::kOpenCL ||
             platform_kind == PlatformKind::kOpenCLAltera) {
    return (*internal::MakeOpenCLTimerImplementation())(parent);
  } else if (platform_kind == PlatformKind::kHost) {
    return internal::MakeHostTimerImplementation(parent);
  } else if (platform_kind == PlatformKind::kMock) {
    return nullptr;
  } else {
    LOG(FATAL) << "cannot create timer implementation for platform kind: "
               << PlatformKindString(platform_kind);
  }
}

Timer::Timer(StreamExecutor *parent)
    : implementation_(CreateTimerImplementation(parent)), parent_(parent) {}

Timer::~Timer() { parent_->DeallocateTimer(this); }

uint64 Timer::Microseconds() const { return implementation_->Microseconds(); }

uint64 Timer::Nanoseconds() const { return implementation_->Nanoseconds(); }

}  // namespace gputools
}  // namespace perftools
