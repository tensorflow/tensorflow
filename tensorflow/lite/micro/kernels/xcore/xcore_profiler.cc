// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_profiler.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {
namespace micro {
namespace xcore {

XCoreProfiler::XCoreProfiler() : event_count_(0) {}

void XCoreProfiler::Reset() { event_count_ = 0; }

uint32_t const* XCoreProfiler::GetTimes() { return event_times_; }

uint32_t XCoreProfiler::GetNumTimes() { return event_count_; }

uint32_t XCoreProfiler::BeginEvent(const char* tag, EventType event_type,
                                   int64_t event_metadata1,
                                   int64_t event_metadata2) {
  event_start_time_ = tflite::GetCurrentTimeTicks();
  TFLITE_DCHECK(tag != nullptr);
  event_tag_ = tag;
  return 0;
}

void XCoreProfiler::EndEvent(uint32_t event_handle) {
  uint32_t event_duration;
  int32_t event_end_time = tflite::GetCurrentTimeTicks();
  event_duration = event_end_time - event_start_time_;
  if (event_count_ < XCORE_PROFILER_MAX_LEVELS) {
    event_times_[event_count_] = event_duration;
    event_count_++;
  }
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite