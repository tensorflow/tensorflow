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
#include "tensorflow/lite/profiling/atrace_profiler.h"

#include <dlfcn.h>

#include <type_traits>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace profiling {

// Profiler reporting to ATrace.
class ATraceProfiler : public tflite::Profiler {
 public:
  using FpIsEnabled = std::add_pointer<bool()>::type;
  using FpBeginSection = std::add_pointer<void(const char*)>::type;
  using FpEndSection = std::add_pointer<void()>::type;

  ATraceProfiler() {
    handle_ = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (handle_) {
      // Use dlsym() to prevent crashes on devices running Android 5.1
      // (API level 22) or lower.
      atrace_is_enabled_ =
          reinterpret_cast<FpIsEnabled>(dlsym(handle_, "ATrace_isEnabled"));
      atrace_begin_section_ = reinterpret_cast<FpBeginSection>(
          dlsym(handle_, "ATrace_beginSection"));
      atrace_end_section_ =
          reinterpret_cast<FpEndSection>(dlsym(handle_, "ATrace_endSection"));

      if (!atrace_is_enabled_ || !atrace_begin_section_ ||
          !atrace_end_section_) {
        dlclose(handle_);
        handle_ = nullptr;
      }
    }
  }

  ~ATraceProfiler() override {
    if (handle_) {
      dlclose(handle_);
    }
  }

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
    if (handle_ && atrace_is_enabled_()) {
      // Note: When recording an OPERATOR_INVOKE_EVENT, we have recorded the op
      // name
      // as tag, node index as event_metadata1 and subgraph index as
      // event_metadata2. See the macro TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE
      // defined in tensorflow/lite/core/api/profiler.h for details.
      // Regardless the 'event_type', we encode the perfetto event name as
      // tag@event_metadata1/event_metadata2. In case of OPERATOR_INVOKE_EVENT,
      // the perfetto event name will be op_name@node_index/subgraph_index
      std::string trace_event_tag =
          absl::StrCat(tag, "@", event_metadata1, "/", event_metadata2);
      atrace_begin_section_(trace_event_tag.c_str());
    }
    return 0;
  }

  void EndEvent(uint32_t event_handle) override {
    if (handle_) {
      atrace_end_section_();
    }
  }

 private:
  // Handle to libandroid.so library. Null if not supported.
  void* handle_;
  FpIsEnabled atrace_is_enabled_;
  FpBeginSection atrace_begin_section_;
  FpEndSection atrace_end_section_;
};

std::unique_ptr<tflite::Profiler> CreateATraceProfiler() {
  return std::unique_ptr<tflite::Profiler>(new ATraceProfiler());
}

}  // namespace profiling
}  // namespace tflite
