/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/profiling/signpost_profiler.h"

#import <Foundation/Foundation.h>
#import <os/log.h>
#import <os/signpost.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/lite/core/api/profiler.h"

namespace tflite {
namespace profiling {

class SignpostProfiler : public tflite::Profiler {
 public:
  SignpostProfiler()
      : log_(nullptr), msg_buf_(std::ios::out | std::ios::ate), last_event_handle_(0) {
    if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)) {
      log_ = os_log_create("org.tensorflow.lite", "Tracing");
    }
  }

  ~SignpostProfiler() override {
    if (log_) {
      os_release(log_);
    }
  }

  uint32_t BeginEvent(const char *tag, EventType event_type, int64_t event_metadata1,
                      int64_t event_metadata2) override {
    if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)) {
      if (!os_signpost_enabled(log_)) {
        return 0;
      }
      // We encode the signpost message as tag@event_metadata1/event_metadata2.
      // In case of OPERATOR_INVOKE_EVENT, the event message will be
      // op_name@node_index/subgraph_index. See the macro TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE
      // defined in tensorflow/lite/core/api/profiler.h for details.
      msg_buf_.str("");  // reset the buffer.
      msg_buf_ << tag << "@" << event_metadata1 << "/" << event_metadata2;
      std::string msg_str = msg_buf_.str();
      const char *msg = msg_str.c_str();

      os_signpost_id_t signpost_id = os_signpost_id_generate(log_);
      switch (event_type) {
        case EventType::DEFAULT:
          os_signpost_interval_begin(log_, signpost_id, "default", "%s", msg);
          break;
        case EventType::OPERATOR_INVOKE_EVENT:
          os_signpost_interval_begin(log_, signpost_id, "operator invoke", "%s", msg);
          break;
        case EventType::DELEGATE_OPERATOR_INVOKE_EVENT:
        case EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT:
          os_signpost_interval_begin(log_, signpost_id, "delegate operator invoke", "%s", msg);
          break;
        case EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT:
          os_signpost_interval_begin(log_, signpost_id, "runtime instrumentation", "%s", msg);
          break;
        default:
          os_signpost_interval_begin(log_, signpost_id, "unknown", "%s", msg);
      }

      uint32_t event_handle = ++last_event_handle_;
      saved_events_[event_handle] = std::make_pair(signpost_id, event_type);
      return event_handle;
    } else {
      return 0;
    }
  }

  void EndEvent(uint32_t event_handle) override {
    if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)) {
      if (!os_signpost_enabled(log_)) {
        return;
      }
      auto it = saved_events_.find(event_handle);
      if (it != saved_events_.end()) {
        auto signpost_id = it->second.first;
        auto event_type = it->second.second;
        switch (event_type) {
          case EventType::DEFAULT:
            os_signpost_interval_end(log_, signpost_id, "default");
            break;
          case EventType::OPERATOR_INVOKE_EVENT:
            os_signpost_interval_end(log_, signpost_id, "operator invoke");
            break;
          case EventType::DELEGATE_OPERATOR_INVOKE_EVENT:
          case EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT:
            os_signpost_interval_end(log_, signpost_id, "delegate operator invoke");
            break;
          case EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT:
            os_signpost_interval_end(log_, signpost_id, "runtime instrumentation");
            break;
          default:
            os_signpost_interval_end(log_, signpost_id, "unknown");
        }
        saved_events_.erase(it);
      }
    }
  }

 private:
  os_log_t log_;
  std::stringstream msg_buf_;
  uint32_t last_event_handle_;
  std::unordered_map<uint32_t, std::pair<os_signpost_id_t, EventType>> saved_events_;
};

std::unique_ptr<tflite::Profiler> MaybeCreateSignpostProfiler() {
#if defined(TFLITE_ENABLE_DEFAULT_PROFILER)
  return std::unique_ptr<tflite::Profiler>(new SignpostProfiler());
#else  // TFLITE_ENABLE_DEFAULT_PROFILER
  if ([[[NSProcessInfo processInfo] environment] objectForKey:@"debug.tflite.trace"]) {
    return std::unique_ptr<tflite::Profiler>(new SignpostProfiler());
  } else {
    return nullptr;
  }
#endif  // TFLITE_ENABLE_DEFAULT_PROFILER
}

}  // namespace profiling
}  // namespace tflite
