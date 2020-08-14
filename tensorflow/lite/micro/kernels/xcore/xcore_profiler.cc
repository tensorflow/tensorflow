// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_profiler.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_time.h"

#ifdef XCORE
#include <platform.h>  // for PLATFORM_REFERENCE_MHZ
#endif

namespace tflite {
namespace micro {
namespace xcore {

XCoreProfiler::XCoreProfiler(tflite::ErrorReporter* reporter)
    : reporter_(reporter) {}

uint32_t XCoreProfiler::BeginEvent(const char* tag, EventType event_type,
                                   int64_t event_metadata1,
                                   int64_t event_metadata2) {
  start_time_ = tflite::GetCurrentTimeTicks();
  TFLITE_DCHECK(tag != nullptr);
  event_tag_ = tag;
  return 0;
}

void XCoreProfiler::EndEvent(uint32_t event_handle) {
  int32_t end_time = tflite::GetCurrentTimeTicks();
#ifdef XCORE
  TF_LITE_REPORT_ERROR(reporter_, "%s took %d microseconds", event_tag_,
                       (end_time - start_time_) / PLATFORM_REFERENCE_MHZ);
#else  // not XCORE
  TF_LITE_REPORT_ERROR(reporter_, "%s took %d cycles", event_tag_,
                       (end_time - start_time_));
#endif
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite