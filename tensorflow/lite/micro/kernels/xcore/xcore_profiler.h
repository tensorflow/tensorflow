// Copyright (c) 2019, XMOS Ltd, All rights reserved

#ifndef XCORE_PROFILER_H_
#define XCORE_PROFILER_H_

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/micro/compatibility.h"

#if !defined(XCORE_PROFILER_MAX_LEVELS)
#define XCORE_PROFILER_MAX_LEVELS (64)
#endif

namespace tflite {
namespace micro {
namespace xcore {

class XCoreProfiler : public tflite::Profiler {
 public:
  explicit XCoreProfiler();
  ~XCoreProfiler() override = default;

  // AddEvent is unused for TFLu.
  void AddEvent(const char* tag, EventType event_type, uint64_t start,
                uint64_t end, int64_t event_metadata1,
                int64_t event_metadata2) override{};

  // BeginEvent followed by code followed by EndEvent will profile the code
  // enclosed. Multiple concurrent events are unsupported, so the return value
  // is always 0. Event_metadata1 and event_metadata2 are unused. The tag
  // pointer must be valid until EndEvent is called.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  // Event_handle is ignored since TFLu does not support concurrent events.
  void EndEvent(uint32_t event_handle) override;

  void Reset();

  uint32_t const* GetTimes();
  uint32_t GetNumTimes();

 private:
  const char* event_tag_;
  uint32_t event_start_time_;
  uint32_t event_count_;
  uint32_t event_times_[XCORE_PROFILER_MAX_LEVELS];
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace xcore
}  // namespace micro
}  // namespace tflite

#endif  // XCORE_PROFILER_H_
