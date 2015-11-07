#include "tensorflow/stream_executor/event.h"

#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/stream.h"

namespace perftools {
namespace gputools {

internal::EventInterface* CreateEventImplementation(
    StreamExecutor* stream_exec) {
  PlatformKind platform_kind = stream_exec->platform_kind();
  switch (platform_kind) {
    case PlatformKind::kCuda:
      return (*internal::MakeCUDAEventImplementation())(stream_exec);
    default:
      LOG(FATAL) << "Cannot create event implementation for platform kind: "
                 << PlatformKindString(platform_kind);
  }
}

Event::Event(StreamExecutor* stream_exec)
    : implementation_(CreateEventImplementation(stream_exec)),
      stream_exec_(stream_exec) {}

Event::~Event() {
  auto status = stream_exec_->DeallocateEvent(this);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
}

bool Event::Init() {
  auto status = stream_exec_->AllocateEvent(this);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
    return false;
  }

  return true;
}

Event::Status Event::PollForStatus() {
  return stream_exec_->PollForEventStatus(this);
}

}  // namespace gputools
}  // namespace perftools
